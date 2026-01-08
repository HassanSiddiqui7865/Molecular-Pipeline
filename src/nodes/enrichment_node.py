"""
Enrichment node for LangGraph - Enriches missing dosage information from drugs.com using Selenium.
Uses LlamaIndex for structured extraction.
Only processes entries where is_complete is False.
"""
import logging
import json
import re
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import quote_plus
from pathlib import Path
import time
import random
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import format_resistance_genes, get_icd_names_from_state, create_llm, retry_with_max_attempts, RetryError, chunk_text_custom, _get_overlap_text
from schemas import AntibioticMatchResult, DosageExtractionResult
from prompts import ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE_OPT, DOSAGE_EXTRACTION_PROMPT_TEMPLATE_OPT

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.chrome.options import Options
    SELENIUM_AVAILABLE = True
    logger.info("Selenium imported successfully")
except ImportError as e:
    SELENIUM_AVAILABLE = False
    logger.error(f"Selenium not available: {e}")
    logger.error("Please install with: pip install selenium")
except Exception as e:
    SELENIUM_AVAILABLE = False
    logger.error(f"Error importing Selenium ({type(e).__name__}): {e}")


def _get_selenium_driver() -> Optional[Any]:
    """
    Get a Selenium WebDriver instance with anti-detection settings.
    
    Returns:
        WebDriver instance or None if Selenium is not available
    """
    if not SELENIUM_AVAILABLE:
        return None
    
    try:
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Set reasonable timeouts - don't wait for full page load
        driver.set_page_load_timeout(10)  # Short timeout, we'll wait for specific elements instead
        driver.implicitly_wait(5)
        driver.set_script_timeout(10)
        
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            '''
        })
        
        return driver
    except Exception as e:
        logger.error(f"Error creating Selenium driver: {e}")
        return None


def _google_search_drugs_com_selenium(antibiotic_name: str, driver: Any) -> Optional[str]:
    """
    Search DuckDuckGo using Selenium to find drugs.com dosage URL.
    
    Args:
        antibiotic_name: Name of the antibiotic
        driver: Selenium WebDriver instance
        
    Returns:
        URL of first drugs.com result with 'dosage' in URL, or None if not found
    """
    if not driver:
        return None, []
    
    try:
        search_query = f"{antibiotic_name} dosage drug.com"
        logger.info(f"Searching DuckDuckGo for {antibiotic_name}...")
        
        # Navigate to DuckDuckGo - don't wait for full page load
        max_retries = 3
        retry_delay = 2.0
        navigation_success = False
        for attempt in range(max_retries):
            try:
                driver.get("https://duckduckgo.com/")
                navigation_success = True
                break
            except Exception as e:
                # Timeout is OK - we'll wait for search box instead
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.debug(f"DuckDuckGo page load timeout (attempt {attempt + 1}/{max_retries}), continuing to wait for search box...")
                    navigation_success = True  # Continue anyway
                    break
                elif attempt < max_retries - 1:
                    logger.warning(f"DuckDuckGo navigation failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"DuckDuckGo navigation failed after {max_retries} attempts: {e}")
                    return None
        
        if not navigation_success:
            return None
        
        # Retry waiting for search box
        search_box = None
        for attempt in range(max_retries):
            try:
                search_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "searchbox_input"))
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Search box not found (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    # Try fallback selector
                    try:
                        search_box = driver.find_element(By.CSS_SELECTOR, "input[name='q']")
                        break
                    except:
                        logger.warning(f"Search box not found after {max_retries} attempts: {e}")
                        return None
        
        if not search_box:
            return None
        
        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
            time.sleep(random.uniform(0.3, 0.7))
            
            search_box.click()
            time.sleep(random.uniform(0.2, 0.5))
            
            search_box.clear()
            for char in search_query:
                search_box.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))
            
            time.sleep(random.uniform(0.5, 1.2))
            search_box.send_keys(Keys.RETURN)
            time.sleep(random.uniform(2.5, 4.0))
            
        except Exception as e:
            logger.warning(f"Error during search interaction: {e}, trying direct URL as fallback")
            encoded_query = quote_plus(search_query)
            duckduckgo_url = f"https://duckduckgo.com/?q={encoded_query}"
            try:
                driver.get(duckduckgo_url)
            except Exception as nav_error:
                # Timeout is OK - we'll wait for results anyway
                if "timeout" not in str(nav_error).lower() and "timed out" not in str(nav_error).lower():
                    logger.warning(f"Fallback navigation failed: {nav_error}")
        
        try:
            time.sleep(3)
            
            result_selectors = [
                "a[data-testid='result-title-a']",
                "a.result__a",
                ".result a",
                "article a",
            ]
            
            found_urls = []
            
            for selector in result_selectors:
                try:
                    result_links = driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for link in result_links:
                        try:
                            href = link.get_attribute('href')
                            if not href:
                                continue
                            
                            href_lower = href.lower()
                            if ('duckduckgo.com' in href_lower or 
                                href.startswith('javascript:') or 
                                href.startswith('#') or
                                '?q=' in href_lower or
                                '/?q=' in href_lower):
                                continue
                            
                            if (href.startswith('http') and 
                                'drugs.com' in href_lower and 
                                'dosage' in href_lower):
                                
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(href)
                                    domain = parsed.netloc.lower()
                                    
                                    if 'drugs.com' in domain and 'dosage' in href_lower:
                                        url = href.split('?')[0] if '?' in href else href
                                        
                                        if ('drugs.com' in url.lower() and 
                                            'dosage' in url.lower() and 
                                            'duckduckgo' not in url.lower() and
                                            url.startswith('http')):
                                            found_urls.append(url)
                                            logger.info(f"Found drugs.com dosage URL for {antibiotic_name}: {url}")
                                            return url
                                except Exception as e:
                                    logger.debug(f"Error parsing URL {href}: {e}")
                                    continue
                        except Exception as e:
                            logger.debug(f"Error processing link: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if found_urls:
                logger.warning(f"Found {len(found_urls)} drugs.com URLs but none matched criteria: {found_urls[:3]}")
            else:
                logger.warning(f"Could not find any drugs.com URLs in DuckDuckGo results for {antibiotic_name}")
                
        except Exception as e:
            logger.warning(f"Error finding search results: {e}")
        
        logger.warning(f"No drugs.com dosage URL found for {antibiotic_name}")
        return None
        
    except Exception as e:
        logger.error(f"Error searching DuckDuckGo with Selenium for {antibiotic_name}: {e}")
        return None




def _validate_antibiotic_match(
    url: str,
    antibiotic_name: str,
    driver: Any
) -> bool:
    """
    Validate that the drugs.com page is for the same antibiotic we're searching for.
    Uses LlamaIndex to verify medical match based on page title.
    
    Args:
        url: URL of the drugs.com page
        antibiotic_name: Name of the antibiotic we're searching for
        driver: Selenium WebDriver instance
        
    Returns:
        True if the page matches the antibiotic, False otherwise
    """
    if not driver:
        return False
    
    llm = create_llm()
    if not llm:
        logger.warning("LLM not available for validation, allowing match")
        return True
    
    try:
        logger.info(f"Validating antibiotic match for {antibiotic_name} at {url}...")
        
        # Navigate to URL with retry logic
        def _navigate_to_url():
            try:
                driver.get(url)
                return True
            except Exception as e:
                # Timeout is OK - we'll get title anyway
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.debug(f"Page load timeout for {url}, continuing to get title...")
                    return True  # Continue anyway
                raise
        
        try:
            retry_with_max_attempts(
                operation=_navigate_to_url,
                operation_name=f"Navigation to {url}",
                max_attempts=5,
                retry_delay=2.0,
                should_retry_on_empty=False
            )
        except RetryError as e:
            logger.warning(f"Navigation failed after max attempts for {url}: {e}, continuing with title check")
            # Fail open - try to get title anyway
        
        # Get page title
        page_title = driver.title if driver.title else ""
        
        if not page_title:
            logger.warning(f"No page title found for {url}, skipping validation")
            return True  # Fail open if no title
        
        # Format prompt
        prompt = ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE_OPT.format(
            antibiotic_name=antibiotic_name,
            page_title=page_title
        )
        
        # Use LlamaIndex for structured extraction with retry
        def _perform_validation():
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=AntibioticMatchResult,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            result = program(input_str=prompt)
            if not result:
                return None
            return result
        
        try:
            result = retry_with_max_attempts(
                operation=_perform_validation,
                operation_name=f"Antibiotic match validation for {antibiotic_name}",
                max_attempts=5,
                retry_delay=2.0,
                should_retry_on_empty=False,
                empty_result_handler=lambda: AntibioticMatchResult(is_match=True, reason="Empty result, allowing match")
            )
        except RetryError as e:
            logger.warning(f"Validation failed after max attempts for {antibiotic_name}, allowing match: {e}")
            return True  # Fail open
        
        if not result:
            logger.warning(f"Empty validation result for {antibiotic_name}, allowing match")
            return True
        
        is_match = result.is_match
        reason = result.reason
        
        logger.info(f"Validation result for {antibiotic_name}: is_match={is_match}, reason={reason}")
        
        if is_match:
            logger.info(f"✓ Validated match for {antibiotic_name} (title: {page_title})")
            return True
        else:
            logger.warning(f"✗ No match for {antibiotic_name} (title: {page_title}): {reason}")
            return False
        
    except Exception as e:
        logger.warning(f"Error validating antibiotic match for {antibiotic_name}: {e}")
        # On error, allow it (fail open) but log warning
        return True


def _search_drugs_com_main_page(antibiotic_name: str, driver: Any) -> Optional[str]:
    """
    Search DuckDuckGo for the main drugs.com page for an antibiotic.
    Searches for "{antibiotic_name} drugs.com" and returns the first drugs.com link found.
    
    Args:
        antibiotic_name: Name of the antibiotic
        driver: Selenium WebDriver instance
        
    Returns:
        URL of first drugs.com result (main page, not dosage), or None if not found
    """
    if not driver:
        return None
    
    try:
        search_query = f"{antibiotic_name} drugs.com"
        logger.info(f"Searching DuckDuckGo for main drugs.com page: {search_query}...")
        
        # Navigate to DuckDuckGo
        max_retries = 3
        retry_delay = 2.0
        navigation_success = False
        for attempt in range(max_retries):
            try:
                driver.get("https://duckduckgo.com/")
                navigation_success = True
                break
            except Exception as e:
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.debug(f"DuckDuckGo page load timeout (attempt {attempt + 1}/{max_retries}), continuing...")
                    navigation_success = True
                    break
                elif attempt < max_retries - 1:
                    logger.warning(f"DuckDuckGo navigation failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"DuckDuckGo navigation failed after {max_retries} attempts: {e}")
                    return None
        
        if not navigation_success:
            return None
        
        # Wait for search box
        search_box = None
        for attempt in range(max_retries):
            try:
                search_box = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.ID, "searchbox_input"))
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Search box not found (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    try:
                        search_box = driver.find_element(By.CSS_SELECTOR, "input[name='q']")
                        break
                    except:
                        logger.warning(f"Search box not found after {max_retries} attempts: {e}")
                        return None
        
        if not search_box:
            return None
        
        try:
            driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
            time.sleep(random.uniform(0.3, 0.7))
            
            search_box.click()
            time.sleep(random.uniform(0.2, 0.5))
            
            search_box.clear()
            for char in search_query:
                search_box.send_keys(char)
                time.sleep(random.uniform(0.05, 0.15))
            
            time.sleep(random.uniform(0.5, 1.2))
            search_box.send_keys(Keys.RETURN)
            time.sleep(random.uniform(2.5, 4.0))
            
        except Exception as e:
            logger.warning(f"Error during search interaction: {e}, trying direct URL as fallback")
            encoded_query = quote_plus(search_query)
            duckduckgo_url = f"https://duckduckgo.com/?q={encoded_query}"
            try:
                driver.get(duckduckgo_url)
            except Exception as nav_error:
                if "timeout" not in str(nav_error).lower() and "timed out" not in str(nav_error).lower():
                    logger.warning(f"Fallback navigation failed: {nav_error}")
        
        try:
            time.sleep(3)
            
            result_selectors = [
                "a[data-testid='result-title-a']",
                "a.result__a",
                "a[href*='drugs.com']",
                "div.result a",
                "li[data-layout='organic'] a"
            ]
            
            found_urls = []
            for selector in result_selectors:
                try:
                    links = driver.find_elements(By.CSS_SELECTOR, selector)
                    for link in links:
                        try:
                            href = link.get_attribute('href')
                            if not href:
                                continue
                            
                            href_lower = href.lower()
                            
                            if (href.startswith('http') and 
                                'drugs.com' in href_lower and 
                                'duckduckgo' not in href_lower):
                                
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(href)
                                    domain = parsed.netloc.lower()
                                    
                                    if 'drugs.com' in domain:
                                        url = href.split('?')[0] if '?' in href else href
                                        
                                        if ('drugs.com' in url.lower() and
                                            'duckduckgo' not in url.lower() and
                                            url.startswith('http')):
                                            found_urls.append(url)
                                            logger.info(f"Found drugs.com main page URL for {antibiotic_name}: {url}")
                                            return url
                                except Exception as e:
                                    logger.debug(f"Error parsing URL {href}: {e}")
                                    continue
                        except Exception as e:
                            logger.debug(f"Error processing link: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if found_urls:
                logger.warning(f"Found {len(found_urls)} drugs.com URLs but none matched criteria: {found_urls[:3]}")
            else:
                logger.warning(f"No drugs.com URLs found for {antibiotic_name}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Error searching for drugs.com main page: {e}")
            return None
            
    except Exception as e:
        logger.warning(f"Error in _search_drugs_com_main_page: {e}")
        return None


def _extract_references_from_page(driver: Any) -> List[str]:
    """
    Extract references from drugs.com page by visiting the reference URL.
    Finds div.ddc-reference-list and extracts all <a> links from <ol><li> elements.
    
    Args:
        driver: Selenium WebDriver instance
        
    Returns:
        List of reference URLs (empty list if not found - no error thrown)
    """
    references = []
    try:
        # Try to find the references section - use find_elements (not find_element) to avoid exception
        reference_lists = driver.find_elements(By.CSS_SELECTOR, "div.ddc-reference-list")
        if reference_lists:
            reference_list = reference_lists[0]
            # Find all links in the ordered list
            links = reference_list.find_elements(By.CSS_SELECTOR, "ol li a")
            for link in links:
                href = link.get_attribute('href')
                if href:
                    references.append(href)
                    logger.debug(f"Found reference: {href}")
        else:
            # References section not found - that's OK, just continue
            logger.debug("No reference section found on page - continuing without references")
    except Exception as e:
        # Any error is fine - just log and continue
        logger.debug(f"Could not extract references (non-critical): {e}")
    
    return references


def _extract_references_from_reference_page(antibiotic_name: str, driver: Any) -> List[str]:
    """
    Extract references by searching DuckDuckGo for the main drugs.com page and extracting references.
    If references are not found, returns empty list and continues without error.
    
    Args:
        antibiotic_name: Name of the antibiotic to search for
        driver: Selenium WebDriver instance
    
    Returns:
        List of reference URLs extracted from div.ddc-reference-list (empty list if not found)
    """
    references = []
    try:
        # Search DuckDuckGo for the main drugs.com page
        reference_url = _search_drugs_com_main_page(antibiotic_name, driver)
        if not reference_url:
            logger.debug(f"Could not find drugs.com main page for {antibiotic_name} - continuing without references")
            return []
        
        logger.info(f"Extracting references from {reference_url}...")
        
        # Navigate to the reference URL - if it fails, just return empty list
        try:
            driver.get(reference_url)
        except Exception as e:
            # Navigation failed - that's OK, just continue without references
            logger.debug(f"Could not navigate to reference page {reference_url} (non-critical): {e}")
            return []
        
        # Wait a bit for the page to load
        time.sleep(2)
        
        # Extract references from the page - if not found, returns empty list (no error)
        references = _extract_references_from_page(driver)
        
        if references:
            logger.info(f"Extracted {len(references)} references from {reference_url}")
        else:
            logger.debug(f"No references found on {reference_url} - continuing without references")
        
    except Exception as e:
        # Any error is non-critical - just log and return empty list
        logger.debug(f"Error extracting references from reference page (non-critical): {e}")
    
    return references


def _scrape_drugs_com_page(url: str, antibiotic_name: str, driver: Any) -> tuple[Optional[str], List[str]]:
    """
    Scrape content from drugs.com page using Selenium.
    
    Args:
        url: URL of the drugs.com page
        antibiotic_name: Name of the antibiotic (for reference extraction)
        driver: Selenium WebDriver instance
    
    Returns:
        Tuple of (page_content, references_list) or (None, []) if error
    """
    if not driver:
        return None, []
    
    try:
        logger.info(f"Navigating directly to {url}...")
        
        # Navigate to URL with retry logic
        def _navigate_to_url():
            try:
                driver.get(url)
                return True
            except Exception as e:
                # Timeout is OK - we'll wait for specific element instead
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.debug(f"Page load timeout for {url}, continuing to wait for content element...")
                    return True  # Continue anyway
                raise
        
        try:
            retry_with_max_attempts(
                operation=_navigate_to_url,
                operation_name=f"Navigation to {url}",
                max_attempts=5,
                retry_delay=2.0,
                should_retry_on_empty=False
            )
        except RetryError as e:
            logger.error(f"Navigation failed after max attempts for {url}: {e}")
            return None, []
        
        # Wait for content element with retry logic
        def _wait_for_content():
            content_element = WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.ID, "content"))
            )
            return content_element
        
        try:
            content_element = retry_with_max_attempts(
                operation=_wait_for_content,
                operation_name=f"Waiting for content element at {url}",
                max_attempts=5,
                retry_delay=2.0,
                should_retry_on_empty=False
            )
        except RetryError as e:
            logger.warning(f"Content element not found after max attempts: {e}")
            content_element = None
        
        if not content_element:
            # Try fallback method
            logger.warning(f"Could not find #content element, trying fallback method...")
            try:
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, 'html.parser')
                
                for script in soup(["script", "style"]):
                    script.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                logger.info(f"Extracted {len(text)} characters (fallback method)")
                # Extract references from the main drug page
                references = _extract_references_from_reference_page(url, driver)
                return text, references
            except Exception as e:
                logger.error(f"Fallback extraction also failed: {e}")
                return None, []
        
        text = content_element.text
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        # Extract references by searching for main drugs.com page
        references = _extract_references_from_reference_page(antibiotic_name, driver)
        
        logger.info(f"Extracted {len(text)} characters from #content element, found {len(references)} references")
        return text, references
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None, []




def _extract_fields_with_llamaindex(
    page_content: str,
    medical_name: str,
    missing_fields: List[str],
    existing_data: Dict[str, Any],
    age: Optional[int],
    icd_code_names: Optional[str] = None,
    resistance_gene: Optional[str] = None,
    allergies: Optional[List[str]] = None,
    retry_delay: float = 2.0
) -> Dict[str, Optional[str]]:
    """
    Use LlamaIndex structured output to extract fields from drugs.com content.
    Blends extracted fields with existing data.
    
    Args:
        page_content: Extracted content from drugs.com page
        medical_name: Name of the antibiotic
        missing_fields: List of field names that are missing
        existing_data: Existing data for this antibiotic (to blend with)
        age: Patient age (optional)
        icd_code_names: Transformed ICD code names (comma-separated, optional)
        resistance_gene: Resistance gene name (optional)
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        Dictionary with extracted field values (blended with existing data)
    """
    if not LLAMAINDEX_AVAILABLE:
        logger.error("LlamaIndex not available for extraction")
        return {}
    
    llm = create_llm()
    if not llm:
        logger.error("LLM not available for extraction")
        return {}
    
    try:
        patient_age_str = f"{age} years" if age else "adult"
        missing_fields_str = ", ".join(missing_fields)
        icd_code_names_str = icd_code_names if icd_code_names else "none"
        
        # Build conditional resistance gene sections
        if resistance_gene:
            gene_context = f" | Gene={resistance_gene}"
            gene_matching = f", Gene: {resistance_gene}"
        else:
            gene_context = ""
            gene_matching = ""
        
        # Build conditional allergy sections
        from utils import format_allergies
        allergy_display = format_allergies(allergies) if allergies else None
        if allergy_display:
            allergy_context = f" | Allergies={allergy_display}"
        else:
            allergy_context = ""
        
        # Build existing data context
        existing_data_context = ""
        if existing_data:
            existing_fields = []
            # Emphasize route_of_administration if present - it's critical for dose_duration extraction
            if existing_data.get('route_of_administration'):
                existing_fields.append(f"route_of_administration={existing_data['route_of_administration']} [CRITICAL: Extract dose_duration matching this exact route]")
            if existing_data.get('dose_duration'):
                existing_fields.append(f"dose_duration={existing_data['dose_duration']}")
            if existing_data.get('coverage_for'):
                existing_fields.append(f"coverage_for={existing_data['coverage_for']}")
            if existing_data.get('renal_adjustment'):
                existing_fields.append(f"renal_adjustment={existing_data['renal_adjustment']}")
            if existing_data.get('general_considerations'):
                existing_fields.append(f"general_considerations={existing_data['general_considerations'][:100]}")
            
            if existing_fields:
                existing_data_context = "\n".join(existing_fields)
        
        # Chunk the content using custom token-based chunking
        chunk_size_tokens = 6000
        chunk_overlap_tokens = 200
        chunks = chunk_text_custom(page_content, chunk_size_tokens=chunk_size_tokens, chunk_overlap_tokens=chunk_overlap_tokens)
        
        from utils import _get_token_count
        total_tokens = _get_token_count(page_content)
        logger.info(f"Processing {len(chunks)} chunks for {medical_name} (total: {total_tokens} tokens, {len(page_content)} chars)")
        
        # Process each chunk and accumulate results
        all_results = {
            'dose_duration': [],
            'route_of_administration': [],
            'general_considerations': [],
            'coverage_for': [],
            'renal_adjustment': []
        }
        
        # Track previous chunk text for cross-chunk context
        previous_chunk_text = ""
        
        for i, chunk in enumerate(chunks):
            def _process_chunk():
                # Build cross-chunk context showing what's extracted and what's still missing
                cross_chunk_context = ""
                has_extracted_data = any(all_results.values())
                
                if has_extracted_data or i > 0:
                    extracted_summary = {}
                    still_missing = []
                    
                    for field in ['dose_duration', 'route_of_administration', 'coverage_for', 'renal_adjustment', 'general_considerations']:
                        if field in missing_fields:
                            if all_results[field]:
                                extracted_summary[field] = all_results[field][0]
                            else:
                                still_missing.append(field)
                    
                    context_parts = []
                    if extracted_summary:
                        summary_lines = [f"{k}={v[:80]}" for k, v in extracted_summary.items() if v]
                        if summary_lines:
                            context_parts.append("\n".join(summary_lines))
                    
                    if still_missing:
                        context_parts.append(f"Still missing: {', '.join(still_missing)}")
                    
                    if context_parts:
                        cross_chunk_context = "\n\n" + "\n".join(context_parts) + "\n\n"
                
                # Add overlap text from previous chunk for context
                overlap_context = ""
                if i > 0 and previous_chunk_text:
                    overlap_text = _get_overlap_text(previous_chunk_text, chunk_overlap_tokens)
                    if overlap_text:
                        overlap_context = f"\n\n{overlap_text}\n\n"
                
                # Format prompt with overlap and cross-chunk context
                prompt = DOSAGE_EXTRACTION_PROMPT_TEMPLATE_OPT.format(
                    medical_name=medical_name,
                    patient_age=patient_age_str,
                    icd_codes=icd_code_names_str,
                    gene_context=gene_context,
                    gene_matching=gene_matching,
                    allergy_context=allergy_context,
                    missing_fields=missing_fields_str,
                    existing_data=existing_data_context,
                    cross_chunk_context=cross_chunk_context,
                    chunk_num=i+1,
                    total_chunks=len(chunks),
                    chunk_content=overlap_context + chunk if overlap_context else chunk
                )
                
                # Use LlamaIndex for structured extraction
                program = LLMTextCompletionProgram.from_defaults(
                    output_cls=DosageExtractionResult,
                    llm=llm,
                    prompt_template_str="{input_str}",
                    verbose=False
                )
                
                result = program(input_str=prompt)
                if not result:
                    return None
                return result
            
            try:
                logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {medical_name}")
                result = retry_with_max_attempts(
                    operation=_process_chunk,
                    operation_name=f"LLM extraction from chunk {i+1} for {medical_name}",
                    max_attempts=5,
                    retry_delay=retry_delay,
                    should_retry_on_empty=True
                )
                
                # Store extracted fields from this chunk and merge into all_results
                if 'dose_duration' in missing_fields and result.dose_duration:
                    all_results['dose_duration'].append(result.dose_duration)
                if 'route_of_administration' in missing_fields and result.route_of_administration:
                    all_results['route_of_administration'].append(result.route_of_administration)
                if 'general_considerations' in missing_fields and result.general_considerations:
                    all_results['general_considerations'].append(result.general_considerations)
                if 'coverage_for' in missing_fields and result.coverage_for:
                    all_results['coverage_for'].append(result.coverage_for)
                if 'renal_adjustment' in missing_fields and result.renal_adjustment:
                    all_results['renal_adjustment'].append(result.renal_adjustment)
                
                # Store this chunk's text for overlap context in next chunk
                previous_chunk_text = chunk
                    
            except RetryError as e:
                logger.error(f"Chunk {i+1} extraction failed after max attempts for {medical_name}: {e}")
                raise
        
        # Merge results: blend with existing data, use latest extracted value for each field
        extracted = {}
        
        # Always preserve route_of_administration from existing data (should always be present from extraction)
        extracted['route_of_administration'] = existing_data.get('route_of_administration')
        if not extracted['route_of_administration']:
            logger.warning(f"route_of_administration is missing for {existing_data.get('medical_name', 'unknown')} - this should not happen")
        
        # For each field, use existing if present, otherwise use latest extracted value (from later chunks)
        for field in ['dose_duration', 'coverage_for', 'renal_adjustment']:
            if field in missing_fields:
                if all_results[field]:
                    if field == 'dose_duration':
                        # For dose_duration, prioritize shorter durations when multiple options exist
                        dosages = all_results[field]
                        # Sort by duration length (shorter first)
                        def get_duration_days(dosage_str):
                            """Extract duration in days from dosage string, return a sortable value."""
                            if not dosage_str:
                                return 9999
                            dosage_lower = dosage_str.lower()
                            if 'single' in dosage_lower or 'once' in dosage_lower:
                                return 0.1
                            day_match = re.search(r'for\s+(\d+)\s+days?', dosage_lower)
                            if day_match:
                                return int(day_match.group(1))
                            week_match = re.search(r'for\s+(\d+)\s+weeks?', dosage_lower)
                            if week_match:
                                return int(week_match.group(1)) * 7
                            month_match = re.search(r'for\s+(\d+)\s+months?', dosage_lower)
                            if month_match:
                                return int(month_match.group(1)) * 30
                            return 5000 + len(dosage_str)
                        
                        dosages_sorted = sorted(dosages, key=get_duration_days)
                        extracted[field] = dosages_sorted[0]
                        if len(dosages) > 1:
                            logger.info(f"Multiple dosages found for {existing_data.get('medical_name', 'unknown')}, selected shortest: {extracted[field]}")
                    else:
                        # Use latest extracted value (from later chunks)
                        extracted[field] = all_results[field][-1]
                elif existing_data.get(field):
                    extracted[field] = existing_data[field]
                else:
                    extracted[field] = None
            else:
                extracted[field] = existing_data.get(field)
        
      
        # renal_adjustment is handled by LLM during extraction - no programmatic logic needed
        
        # Special handling for general_considerations - use latest extracted, merge if multiple
        if 'general_considerations' in missing_fields:
            if all_results['general_considerations']:
                considerations = all_results['general_considerations']
                if existing_data.get('general_considerations'):
                    # Merge with existing
                    existing_cons = existing_data['general_considerations']
                    latest_cons = considerations[-1]
                    combined = f"{existing_cons}. {latest_cons}"
                    if len(combined) <= 300:
                        extracted['general_considerations'] = combined
                    else:
                        extracted['general_considerations'] = latest_cons
                else:
                    # Use latest extracted value
                    extracted['general_considerations'] = considerations[-1]
            elif existing_data.get('general_considerations'):
                extracted['general_considerations'] = existing_data['general_considerations']
            else:
                extracted['general_considerations'] = None
        else:
            extracted['general_considerations'] = existing_data.get('general_considerations')
        
        logger.info(f"Extracted fields for {medical_name}: {[k for k, v in extracted.items() if v and k in missing_fields]}")
        return extracted
        
    except Exception as e:
        logger.error(f"Error extracting fields for {medical_name}: {e}")
        return {}


def _scrape_antibiotic_page(
    antibiotic: Dict[str, Any],
    category: str,
    idx: int
) -> Tuple[str, int, Optional[str], List[str], bool, int, List[str]]:
    """
    Scrape page content for a single antibiotic in a separate thread with its own browser.
    
    Args:
        antibiotic: Antibiotic dictionary
        category: Category name (first_choice, second_choice, alternative_antibiotic)
        idx: Index of the antibiotic in the list
        
    Returns:
        Tuple of (category, idx, page_content, missing_fields, validation_failed, num_chunks, references) 
        where validation_failed=True means the drug name didn't match and should be removed
        num_chunks is the number of chunks the page content would be split into
        references is a list of reference URLs extracted from the page
    """
    medical_name = antibiotic.get('medical_name', '')
    if not medical_name:
        return (category, idx, None, [], False, 0, [])
    
    # Determine which fields are missing - check ALL fields that are null
    # NOTE: route_of_administration should always be present from extraction, so we don't check it here
    missing_fields = []
    if antibiotic.get('dose_duration') is None:
        missing_fields.append('dose_duration')
    # route_of_administration should always be there from extraction - skip it
    if antibiotic.get('coverage_for') is None:
        missing_fields.append('coverage_for')
    if antibiotic.get('renal_adjustment') is None:
        missing_fields.append('renal_adjustment')
    if antibiotic.get('general_considerations') is None:
        missing_fields.append('general_considerations')
    
    if not missing_fields:
        return (category, idx, None, [], False, 0, [])
    
    driver = None
    try:
        driver = _get_selenium_driver()
        if not driver:
            logger.error(f"[Thread] Could not create Selenium driver for {medical_name}")
            return (category, idx, None, missing_fields, False, 0, [])
        
        logger.info(f"[Thread] Scraping {medical_name} from drugs.com (missing: {', '.join(missing_fields)})...")
        
        drugs_com_url = _google_search_drugs_com_selenium(medical_name, driver)
        
        if drugs_com_url:
            # Validate that the page is about the same antibiotic before scraping
            is_valid = _validate_antibiotic_match(drugs_com_url, medical_name, driver)
            if not is_valid:
                logger.warning(f"[Thread] Page validation failed for {medical_name} - drug name doesn't match, will remove from result")
                return (category, idx, None, missing_fields, True, 0, [])  # validation_failed=True
            
            logger.info(f"[Thread] Navigating directly to {drugs_com_url}")
            page_content, references = _scrape_drugs_com_page(drugs_com_url, medical_name, driver)
            
            if page_content:
                # Calculate number of chunks using custom chunking
                chunks = chunk_text_custom(page_content, chunk_size_tokens=6000, chunk_overlap_tokens=200)
                num_chunks = len(chunks)
                from utils import _get_token_count
                total_tokens = _get_token_count(page_content)
                logger.info(f"[Thread] Successfully scraped page for {medical_name} ({num_chunks} chunks, {total_tokens} tokens, {len(references)} references)")
                return (category, idx, page_content, missing_fields, False, num_chunks, references)
            else:
                logger.warning(f"[Thread] Could not scrape page for {medical_name}")
                return (category, idx, None, missing_fields, False, 0, [])
        else:
            logger.warning(f"[Thread] Could not find drugs.com dosage URL for {medical_name}")
            return (category, idx, None, missing_fields, False, 0, [])
            
    except Exception as e:
        logger.error(f"[Thread] Error scraping {medical_name}: {e}", exc_info=True)
        return (category, idx, None, missing_fields, False, 0, [])
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"[Thread] Error closing Selenium driver for {medical_name}: {e}")


def enrichment_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrichment node that extracts content from drugs.com for antibiotics with missing fields.
    Only processes entries where is_complete is False.
    Uses LangGraph memory store to store previous chunk context to avoid randomness.
    Blends extracted fields with existing data.
    
    Args:
        state: Pipeline state dictionary (should have 'result' from synthesize_node)
        
    Returns:
        Updated state with enriched result
    """
    try:
        result = state.get('result', {})
        if not result:
            logger.warning("No result to enrich")
            return {'result': result}
        
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium is not available. Cannot perform enrichment.")
            logger.error("Please install selenium: pip install selenium")
            return {'result': result}
        
        if not LLAMAINDEX_AVAILABLE:
            logger.error("LlamaIndex is not available. Cannot perform enrichment.")
            logger.error("Please install: pip install llama-index llama-index-llms-ollama")
            return {'result': result}
        
        therapy_plan = result.get('antibiotic_therapy_plan', {})
        
        # Collect all antibiotics that need processing (only is_complete: false)
        antibiotics_to_process = []
        
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            antibiotics = therapy_plan.get(category, [])
            if not isinstance(antibiotics, list):
                continue
            
            for idx, antibiotic in enumerate(antibiotics):
                medical_name = antibiotic.get('medical_name', '')
                if not medical_name:
                    continue
                
                # Only process if is_complete is False
                is_complete = antibiotic.get('is_complete', True)  # Default to True if not set
                if is_complete:
                    logger.debug(f"Skipping {medical_name} (is_complete=True)")
                    continue
                
                # Check which fields are missing - extract ALL null fields from drugs.com
                # NOTE: route_of_administration should always be present from extraction, so we don't check it here
                missing_fields = []
                if antibiotic.get('dose_duration') is None:
                    missing_fields.append('dose_duration')
                # route_of_administration should always be there from extraction - skip it
                if antibiotic.get('coverage_for') is None:
                    missing_fields.append('coverage_for')
                if antibiotic.get('renal_adjustment') is None:
                    missing_fields.append('renal_adjustment')
                if antibiotic.get('general_considerations') is None:
                    missing_fields.append('general_considerations')
                
                # Only process if there are missing fields to extract
                if missing_fields:
                    antibiotics_to_process.append((antibiotic, category, idx, missing_fields))
        
        if not antibiotics_to_process:
            logger.info("No incomplete antibiotics need enrichment")
            return {'result': result}
        
        logger.info(f"Processing {len(antibiotics_to_process)} incomplete antibiotics...")
        
        input_params = state.get('input_parameters', {})
        
        # Get ICD code names from transformed state
        icd_code_names = get_icd_names_from_state(state)
        
        # Get resistance genes from input parameters and format
        from utils import get_resistance_genes_from_input, format_resistance_genes, get_allergies_from_input
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistance_gene = format_resistance_genes(resistant_genes)  # Returns None if empty
        allergies = get_allergies_from_input(input_params)
        
        age = input_params.get('age')
        
        # Separate antibiotics by category
        first_choice_ab = [(ab, cat, idx, mf) for ab, cat, idx, mf in antibiotics_to_process if cat == 'first_choice']
        second_choice_ab = [(ab, cat, idx, mf) for ab, cat, idx, mf in antibiotics_to_process if cat == 'second_choice']
        alternative_ab = [(ab, cat, idx, mf) for ab, cat, idx, mf in antibiotics_to_process if cat == 'alternative_antibiotic']
        
        # Track antibiotics to remove (validation failed)
        antibiotics_to_remove = {
            'first_choice': [],
            'second_choice': [],
            'alternative_antibiotic': []
        }
        
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        total_to_process = len(first_choice_ab) + len(second_choice_ab) + len(alternative_ab)
        processed_count = 0
        
        # Process first_choice and second_choice sequentially
        # For first_choice and second_choice: if extraction fails (no required fields), drop the antibiotic
        for antibiotic, category, idx, missing_fields in first_choice_ab + second_choice_ab:
            medical_name = antibiotic.get('medical_name', 'unknown')
            try:
                logger.info(f"Processing {medical_name}...")
                
                # Step 1: Scrape page content (with validation)
                logger.info(f"  [1/2] Scraping {medical_name} from drugs.com...")
                _, _, page_content, _, validation_failed, num_chunks, references = _scrape_antibiotic_page(antibiotic, category, idx)
                
                # If validation failed (name doesn't match), mark for removal
                if validation_failed:
                    logger.warning(f"  Validation failed for {medical_name} - removing from result")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                if not page_content:
                    logger.warning(f"  Could not scrape {medical_name}, removing from result (first_choice/second_choice require extraction)")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                # Append references to mentioned_in_sources
                if references:
                    existing_sources = antibiotic.get('mentioned_in_sources', [])
                    if not isinstance(existing_sources, list):
                        existing_sources = []
                    # Add references that aren't already in the list
                    for ref in references:
                        if ref not in existing_sources:
                            existing_sources.append(ref)
                    antibiotic['mentioned_in_sources'] = existing_sources
                    logger.info(f"  ✓ Added {len(references)} references to {medical_name}")
                
                # Step 2: Extract fields using LlamaIndex
                route_info = antibiotic.get('route_of_administration', 'not specified')
                if 'dose_duration' in missing_fields and route_info and route_info != 'not specified':
                    logger.info(f"  [2/2] Extracting fields for {medical_name} (route: {route_info} - will extract dose_duration matching this route)...")
                else:
                    logger.info(f"  [2/2] Extracting fields for {medical_name}...")
                extracted_fields = _extract_fields_with_llamaindex(
                    page_content=page_content,
                    medical_name=medical_name,
                    missing_fields=missing_fields,
                    existing_data=antibiotic,  # Pass existing data to blend with
                    age=age,
                    icd_code_names=icd_code_names,
                    resistance_gene=resistance_gene,
                    allergies=allergies
                )
                
                # Update antibiotic with extracted fields (only update missing fields)
                updated = False
                required_fields_updated = False
                for field in missing_fields:
                    if field in extracted_fields and extracted_fields[field]:
                        antibiotic[field] = extracted_fields[field]
                        updated = True
                        # Check if this is a required field
                        if field in ['coverage_for', 'dose_duration', 'route_of_administration', 'renal_adjustment', 'general_considerations']:
                            required_fields_updated = True
                        logger.info(f"  ✓ Updated {field} for {medical_name}")
                
                # Set defaults only if not extracted from enrichment
                if not antibiotic.get('renal_adjustment'):
                    antibiotic['renal_adjustment'] = "No Renal Adjustment"
                    logger.info(f"  ✓ Set default renal_adjustment for {medical_name}: No Renal Adjustment")
                    updated = True
                
                if not antibiotic.get('general_considerations'):
                    antibiotic['general_considerations'] = "Standard precautions apply; monitor for adverse effects"
                    logger.info(f"  ✓ Set default general_considerations for {medical_name}")
                    updated = True
                
                # Update is_complete status
                antibiotic['is_complete'] = (
                    antibiotic.get('medical_name') is not None and
                    antibiotic.get('coverage_for') is not None and
                    antibiotic.get('dose_duration') is not None and
                    antibiotic.get('route_of_administration') is not None and
                    antibiotic.get('renal_adjustment') is not None and
                    antibiotic.get('general_considerations') is not None
                )
                
                # For first_choice and second_choice: if no required fields were extracted, remove it
                if not updated:
                    logger.warning(f"  No fields extracted for {medical_name}, removing from {category} (first_choice/second_choice require extraction)")
                    antibiotics_to_remove[category].append(idx)
                elif missing_fields and not required_fields_updated:
                    # This shouldn't happen now since all fields are required, but keep as safety check
                    logger.warning(f"  No required fields extracted for {medical_name}, removing from {category} (first_choice/second_choice require extraction)")
                    antibiotics_to_remove[category].append(idx)
                    # Emit progress even if failed
                    if progress_callback and total_to_process > 0:
                        processed_count += 1
                        sub_progress = (processed_count / total_to_process) * 100.0
                        progress_callback('enrichment', sub_progress, f'Processed {processed_count}/{total_to_process} antibiotics')
                    continue
                
                if updated:
                    logger.info(f"  ✓ Completed enrichment for {medical_name} (is_complete={antibiotic['is_complete']})")
                else:
                    logger.warning(f"  No fields updated for {medical_name}")
                
                # Emit progress for this antibiotic
                if progress_callback and total_to_process > 0:
                    processed_count += 1
                    sub_progress = (processed_count / total_to_process) * 100.0
                    progress_callback('enrichment', sub_progress, f'Enriched {processed_count}/{total_to_process} antibiotics')
                
            except Exception as e:
                logger.error(f"Error processing {medical_name}: {e}", exc_info=True)
                # For first_choice/second_choice, remove on error
                antibiotics_to_remove[category].append(idx)
                continue
        
        # Process alternative_antibiotic: check how many are already complete, only process enough to reach 5 total
        if alternative_ab:
            # Count how many alternative_antibiotic are already complete
            all_alternative = therapy_plan.get('alternative_antibiotic', [])
            complete_count = sum(1 for ab in all_alternative if ab.get('is_complete', False))
            
            logger.info(f"Found {complete_count} complete alternative_antibiotic entries, {len(alternative_ab)} incomplete ones")
            
            # If we already have 5 or more complete, skip processing
            if complete_count >= 5:
                logger.info(f"Already have {complete_count} complete alternative_antibiotic (>= 5), skipping enrichment for incomplete ones")
            else:
                # Calculate how many we need to process
                needed = 5 - complete_count
                logger.info(f"Need {needed} more complete alternative_antibiotic to reach 5 total. Processing {len(alternative_ab)} incomplete ones...")
                
                # Step 1: Scrape all incomplete alternative_antibiotic concurrently
                scraped_alternative = {}  # {(category, idx): (page_content, missing_fields, antibiotic, num_chunks, validation_failed, references)}
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_antibiotic = {
                        executor.submit(_scrape_antibiotic_page, ab, cat, idx): (ab, cat, idx, mf)
                        for ab, cat, idx, mf in alternative_ab
                    }
                    
                    for future in as_completed(future_to_antibiotic):
                        try:
                            category, idx, page_content, missing_fields, validation_failed, num_chunks, references = future.result()
                            ab, cat, idx_orig, mf = future_to_antibiotic[future]
                            
                            if validation_failed:
                                # Mark for removal
                                antibiotics_to_remove[category].append(idx)
                                logger.warning(f"Validation failed for {ab.get('medical_name', 'unknown')} - will remove")
                            elif page_content:
                                # Store successful scrapes with chunk count and references
                                scraped_alternative[(category, idx)] = (page_content, missing_fields, ab, num_chunks, False, references)
                            else:
                                # Scraping failed, mark for removal
                                antibiotics_to_remove[category].append(idx)
                        except Exception as e:
                            logger.error(f"Error getting scraping result: {e}")
                            if future in future_to_antibiotic:
                                ab, cat, idx, mf = future_to_antibiotic[future]
                                antibiotics_to_remove[cat].append(idx)
                
                # Step 2: Select top N with fewer chunks (where N = needed, excluding validation failures)
                if scraped_alternative:
                    # Sort by number of chunks (ascending - fewer chunks first)
                    # Add secondary sort by medical_name for deterministic ordering
                    sorted_alternatives = sorted(
                        scraped_alternative.items(),
                        key=lambda x: (
                            x[1][3] if len(x[1]) > 3 else float('inf'),  # Sort by num_chunks (4th element)
                            x[1][2].get('medical_name', '') if len(x[1]) > 2 and isinstance(x[1][2], dict) else ''  # Secondary sort by name
                        )
                    )
                    
                    # Step 3: Process sequentially, trying to get enough complete ones
                    # For alternative_antibiotic: if extraction fails for one, we can try others
                    # Note: processed_count continues from first_choice/second_choice processing above
                    processed_indices = set()
                    alternative_completed = 0
                    
                    # Process enough to reach 5 total complete entries
                    for (category, idx), (page_content, missing_fields, antibiotic, num_chunks, _, references) in sorted_alternatives:
                        if alternative_completed >= needed:
                            break
                        
                        medical_name = antibiotic.get('medical_name', 'unknown')
                        try:
                            route_info = antibiotic.get('route_of_administration', 'not specified')
                            if 'dose_duration' in missing_fields and route_info and route_info != 'not specified':
                                logger.info(f"Processing {medical_name} (alternative_antibiotic, {num_chunks} chunks, route: {route_info} - will extract dose_duration matching this route)...")
                            else:
                                logger.info(f"Processing {medical_name} (alternative_antibiotic, {num_chunks} chunks)...")
                            
                            # Extract fields using LlamaIndex
                            extracted_fields = _extract_fields_with_llamaindex(
                                page_content=page_content,
                                medical_name=medical_name,
                                missing_fields=missing_fields,
                                existing_data=antibiotic,
                                age=age,
                                icd_code_names=icd_code_names,
                                resistance_gene=resistance_gene,
                                allergies=allergies
                            )
                            
                            # Update antibiotic with extracted fields
                            updated = False
                            for field in missing_fields:
                                if field in extracted_fields and extracted_fields[field]:
                                    antibiotic[field] = extracted_fields[field]
                                    updated = True
                                    logger.info(f"  ✓ Updated {field} for {medical_name}")
                            
                            # Fill missing fields with defaults if not extracted
                            if not antibiotic.get('renal_adjustment'):
                                antibiotic['renal_adjustment'] = "No Renal Adjustment"
                                logger.info(f"  ✓ Set default renal_adjustment for {medical_name}: No Renal Adjustment")
                                updated = True
                            
                            if not antibiotic.get('general_considerations'):
                                antibiotic['general_considerations'] = "Standard precautions apply; monitor for adverse effects"
                                logger.info(f"  ✓ Set default general_considerations for {medical_name}")
                                updated = True

                            # Update is_complete status
                            antibiotic['is_complete'] = (
                                antibiotic.get('medical_name') is not None and
                                antibiotic.get('coverage_for') is not None and
                                antibiotic.get('dose_duration') is not None and
                                antibiotic.get('route_of_administration') is not None and
                                antibiotic.get('renal_adjustment') is not None and
                                antibiotic.get('general_considerations') is not None
                            )
                            
                            processed_indices.add(idx)
                            
                            if antibiotic['is_complete']:
                                logger.info(f"  ✓ Completed enrichment for {medical_name} (is_complete=True)")
                            elif updated:
                                # Got some fields but not complete - keep it for alternative_antibiotic
                                logger.info(f"  ✓ Updated fields for {medical_name} but not yet complete (is_complete={antibiotic['is_complete']})")
                            else:
                                # Extraction failed completely - for alternative_antibiotic, we can try others, but remove this one
                                logger.warning(f"  No fields extracted for {medical_name}, will try others if needed")
                                antibiotics_to_remove[category].append(idx)
                            
                            # Emit progress for this antibiotic
                            if progress_callback and total_to_process > 0:
                                processed_count += 1
                                sub_progress = (processed_count / total_to_process) * 100.0
                                progress_callback('enrichment', sub_progress, f'Enriched {processed_count}/{total_to_process} antibiotics')
                            
                        except Exception as e:
                            logger.error(f"Error processing {medical_name}: {e}", exc_info=True)
                            # For alternative_antibiotic, continue trying others, but remove this one
                            processed_indices.add(idx)
                            antibiotics_to_remove[category].append(idx)
                            continue
                    
                    # Remove the alternative_antibiotic that weren't processed (because we already have enough complete ones)
                    for (category, idx), _ in scraped_alternative.items():
                        if idx not in processed_indices:
                            antibiotics_to_remove[category].append(idx)
                            ab_name = scraped_alternative[(category, idx)][2].get('medical_name', 'unknown') if (category, idx) in scraped_alternative and len(scraped_alternative[(category, idx)]) > 2 else 'unknown'
                            logger.info(f"Removed {ab_name} (not selected for processing - enough complete entries or lower priority)")
                else:
                    # No valid scraped results, mark all for removal
                    logger.warning("No valid alternative_antibiotic after scraping and validation")
                    for (category, idx), _ in scraped_alternative.items():
                        antibiotics_to_remove[category].append(idx)
        
        # Remove antibiotics that failed validation
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            antibiotics = therapy_plan.get(category, [])
            if isinstance(antibiotics, list) and antibiotics_to_remove[category]:
                # Remove in reverse order to maintain indices
                for idx in sorted(antibiotics_to_remove[category], reverse=True):
                    if 0 <= idx < len(antibiotics):
                        removed_ab = antibiotics.pop(idx)
                        ab_name = removed_ab.get('medical_name', 'unknown') if isinstance(removed_ab, dict) else 'unknown'
                        logger.info(f"Removed {ab_name} from {category} (validation failed - drug name doesn't match)")
        
        # Final cleanup AFTER all enrichment processing:
        # 1. Remove antibiotics with null critical fields (medical_name, dose_duration, coverage_for, route_of_administration)
        # 2. Apply default values for renal_adjustment and general_considerations if still null
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            antibiotics = therapy_plan.get(category, [])
            if isinstance(antibiotics, list):
                # First, collect indices to remove (antibiotics with null critical fields)
                indices_to_remove = []
                for idx, ab in enumerate(antibiotics):
                    medical_name = ab.get('medical_name', '')
                    
                    # Remove antibiotics with null critical fields
                    if (not medical_name or 
                        ab.get('dose_duration') is None or 
                        ab.get('coverage_for') is None or 
                        ab.get('route_of_administration') is None):
                        logger.warning(f"Removing {medical_name or 'unknown'} from {category}: critical fields are null")
                        indices_to_remove.append(idx)
                        continue
                    
                    # Set defaults for renal_adjustment and general_considerations if still null
                    if not ab.get('renal_adjustment'):
                        ab['renal_adjustment'] = "No Renal Adjustment"
                        logger.info(f"  ✓ Set default renal_adjustment for {medical_name}: No Renal Adjustment")
                    
                    if not ab.get('general_considerations'):
                        ab['general_considerations'] = "Standard precautions apply; monitor for adverse effects"
                        logger.info(f"  ✓ Set default general_considerations for {medical_name}")
                    
                    # Recalculate is_complete
                    ab['is_complete'] = (
                        ab.get('medical_name') is not None and
                        ab.get('coverage_for') is not None and
                        ab.get('dose_duration') is not None and
                        ab.get('route_of_administration') is not None and
                        ab.get('renal_adjustment') is not None and
                        ab.get('general_considerations') is not None
                    )
                
                # Remove antibiotics with null critical fields (in reverse order to maintain indices)
                for idx in sorted(indices_to_remove, reverse=True):
                    removed_ab = antibiotics.pop(idx)
                    logger.info(f"  ✗ Removed {removed_ab.get('medical_name', 'unknown')} from {category} (null critical fields)")
        
        # Group and unify antibiotics with the same name
        _group_and_unify_antibiotics(therapy_plan)
        
        logger.info("Enrichment complete")
        
        # Save results
        input_params = state.get('input_parameters', {})
        icd_transformation = state.get('icd_transformation', {})
        _save_enrichment_results(input_params, result, icd_transformation)
        
        return {'result': result}
        
    except RetryError as e:
        error_msg = f"Enrichment node failed: {e.operation_name} - {str(e)}"
        logger.error(error_msg)
        # Record error in state and stop pipeline
        errors = state.get('errors', [])
        errors.append(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        logger.error(f"Error in enrichment_node: {e}", exc_info=True)
        # Record error in state
        errors = state.get('errors', [])
        errors.append(f"Enrichment node error: {str(e)}")
        raise


def _group_and_unify_antibiotics(therapy_plan: Dict[str, Any]) -> None:
    """
    Group antibiotics by name and unify them:
    - route_of_administration: combine as "route1/route2"
    - dose_duration: combine with "or" instead of "/"
    - general_considerations: join them
    - renal_adjustment: if both are "No Renal Adjustment", keep it once; otherwise keep the one with a value
    """
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
        antibiotics = therapy_plan.get(category, [])
        if not isinstance(antibiotics, list):
            continue
        
        # Group antibiotics by medical_name
        # Skip antibiotics with null critical fields (safety check)
        grouped = {}
        for ab in antibiotics:
            if not isinstance(ab, dict):
                continue
            medical_name = ab.get('medical_name') or ''
            if isinstance(medical_name, str):
                medical_name = medical_name.strip()
            else:
                medical_name = ''
            
            # Skip antibiotics with null critical fields
            if (not medical_name or 
                ab.get('dose_duration') is None or 
                ab.get('coverage_for') is None or 
                ab.get('route_of_administration') is None):
                logger.warning(f"Skipping {medical_name or 'unknown'} in grouping: critical fields are null")
                continue
            
            if medical_name not in grouped:
                grouped[medical_name] = []
            grouped[medical_name].append(ab)
        
        # Unify groups with multiple entries
        unified_antibiotics = []
        for medical_name, ab_list in grouped.items():
            if len(ab_list) == 1:
                # Single entry, keep as is
                unified_antibiotics.append(ab_list[0])
            else:
                # Multiple entries with same name, unify them
                logger.info(f"Unifying {len(ab_list)} entries for {medical_name}")
                
                # Start with the first entry as base
                unified = ab_list[0].copy()
                
                # Collect unique routes
                routes = []
                for ab in ab_list:
                    route = ab.get('route_of_administration') or ''
                    if isinstance(route, str):
                        route = route.strip()
                    else:
                        route = ''
                    if route and route not in routes:
                        routes.append(route)
                
                # Combine routes with "/"
                if routes:
                    unified['route_of_administration'] = '/'.join(routes)
                else:
                    unified['route_of_administration'] = ab_list[0].get('route_of_administration') or 'N/A'
                
                # Collect unique dose_duration
                doses = []
                for ab in ab_list:
                    dose = ab.get('dose_duration') or ''
                    if isinstance(dose, str):
                        dose = dose.strip()
                    else:
                        dose = ''
                    if dose and dose not in doses:
                        doses.append(dose)
                
                # Combine doses with " or "
                if doses:
                    unified['dose_duration'] = ' or '.join(doses)
                else:
                    unified['dose_duration'] = ab_list[0].get('dose_duration') or 'N/A'
                
                # Join general_considerations
                considerations = []
                for ab in ab_list:
                    gc = ab.get('general_considerations') or ''
                    if isinstance(gc, str):
                        gc = gc.strip()
                    else:
                        gc = ''
                    if gc and gc not in considerations:
                        considerations.append(gc)
                
                if considerations:
                    # Join with semicolons, avoiding duplicates
                    unified['general_considerations'] = '; '.join(considerations)
                else:
                    unified['general_considerations'] = ab_list[0].get('general_considerations', 'N/A')
                
                # Handle renal_adjustment
                renal_adjustments = []
                for ab in ab_list:
                    ra = ab.get('renal_adjustment') or ''
                    if isinstance(ra, str):
                        ra = ra.strip()
                    else:
                        ra = ''
                    if ra:
                        renal_adjustments.append(ra)
                
                if renal_adjustments:
                    # If all are "No Renal Adjustment", keep it once
                    if all(ra == "No Renal Adjustment" for ra in renal_adjustments):
                        unified['renal_adjustment'] = "No Renal Adjustment"
                    else:
                        # Keep the one with a value (not "No Renal Adjustment")
                        non_default = [ra for ra in renal_adjustments if ra != "No Renal Adjustment"]
                        if non_default:
                            unified['renal_adjustment'] = non_default[0]  # Take first non-default
                        else:
                            unified['renal_adjustment'] = renal_adjustments[0]
                else:
                    unified['renal_adjustment'] = ab_list[0].get('renal_adjustment', 'N/A')
                
                # Keep other fields from first entry (coverage_for, category, etc.)
                # is_complete should be True if all required fields are present
                unified['is_complete'] = (
                    unified.get('medical_name') is not None and
                    unified.get('coverage_for') is not None and
                    unified.get('dose_duration') is not None and
                    unified.get('route_of_administration') is not None and
                    unified.get('renal_adjustment') is not None and
                    unified.get('general_considerations') is not None
                )
                
                unified_antibiotics.append(unified)
                logger.info(f"  ✓ Unified {medical_name}: route={unified.get('route_of_administration')}, dose={unified.get('dose_duration')[:50]}...")
        
        # Update the category with unified antibiotics
        therapy_plan[category] = unified_antibiotics


def _save_enrichment_results(input_params: Dict, result: Dict, icd_transformation: Dict = None) -> None:
    """Save enrichment results to file."""
    try:
        from config import get_output_config
        output_config = get_output_config()
        
        # Check if saving is enabled
        if not output_config.get('save_enabled', True):
            logger.debug("Saving enrichment results disabled (production mode)")
            return
        
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "enrichment_result.json"
        
        output_data = {
            'input_parameters': input_params,
            'result': result
        }
        
        # Include ICD transformation with formatted codes (Code (Name))
        if icd_transformation:
            from utils import get_icd_names_from_state
            # Create a temporary state dict to get formatted ICD codes
            temp_state = {'icd_transformation': icd_transformation}
            icd_codes_formatted = get_icd_names_from_state(temp_state)
            output_data['icd_transformation'] = {
                **icd_transformation,
                'icd_codes_formatted': icd_codes_formatted  # Add formatted string with names
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enrichment results saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save enrichment results: {e}")
