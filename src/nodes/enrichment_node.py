"""
Enrichment node for LangGraph - Enriches missing dosage information from drugs.com using Selenium.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import quote_plus, unquote
import time
import random
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import format_resistance_genes

logger = logging.getLogger(__name__)

try:
    import dspy
    from dspy import Signature, InputField, OutputField, Module
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available. Install with: pip install dspy-ai")

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
        # Run in visible mode (headless = False)
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute script to hide webdriver property
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
    Search DuckDuckGo using Selenium by mimicking human behavior:
    1. Go to DuckDuckGo homepage
    2. Type query in search bar
    3. Click search button
    4. Extract first drugs.com URL with 'dosage' in it.
    
    Args:
        antibiotic_name: Name of the antibiotic
        driver: Selenium WebDriver instance
        
    Returns:
        URL of first drugs.com result with 'dosage' in URL, or None if not found
    """
    if not driver:
        return None
    
    try:
        search_query = f"{antibiotic_name} dosage drug.com"
        
        logger.info(f"Searching DuckDuckGo for {antibiotic_name}...")
        
        # Step 1: Go to DuckDuckGo homepage
        driver.get("https://duckduckgo.com/")
        # Random wait to mimic human reading the page
        time.sleep(random.uniform(1.5, 3.0))
        
        # Step 2: Find search box and type query
        try:
            # DuckDuckGo search box selector
            search_box = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, "searchbox_input"))
            )
            
            if not search_box:
                # Fallback selector
                search_box = driver.find_element(By.CSS_SELECTOR, "input[name='q']")
            
            # Scroll to search box (human-like behavior)
            driver.execute_script("arguments[0].scrollIntoView(true);", search_box)
            time.sleep(random.uniform(0.3, 0.7))
            
            # Click on search box first (human-like)
            search_box.click()
            time.sleep(random.uniform(0.2, 0.5))
            
            # Type query character by character (human-like with variable speed)
            search_box.clear()
            for char in search_query:
                search_box.send_keys(char)
                # Random delay between keystrokes (0.05-0.15 seconds)
                time.sleep(random.uniform(0.05, 0.15))
            
            # Random pause before clicking (like human thinking)
            time.sleep(random.uniform(0.5, 1.2))
            
            # Step 3: Press Enter to search (DuckDuckGo doesn't always have a visible button)
            search_box.send_keys(Keys.RETURN)
            
            # Step 4: Wait for search results to load (with random delay)
            time.sleep(random.uniform(2.5, 4.0))  # Random wait for results
            
        except Exception as e:
            logger.warning(f"Error during search interaction: {e}, trying direct URL as fallback")
            # Fallback to direct URL if interaction fails
            encoded_query = quote_plus(search_query)
            duckduckgo_url = f"https://duckduckgo.com/?q={encoded_query}"
            driver.get(duckduckgo_url)
            time.sleep(2)
        
        # Step 5: Find all search result links from DuckDuckGo results
        try:
            # Wait for search results to appear
            time.sleep(3)  # Give more time for results to load
            
            # DuckDuckGo result links - try multiple selectors
            result_selectors = [
                "a[data-testid='result-title-a']",  # Modern DuckDuckGo
                "a.result__a",  # Classic DuckDuckGo
                ".result a",  # Any link in result container
                "article a",  # Links in article elements
            ]
            
            found_urls = []
            
            for selector in result_selectors:
                try:
                    result_links = driver.find_elements(By.CSS_SELECTOR, selector)
                    logger.debug(f"Found {len(result_links)} links with selector {selector}")
                    
                    for link in result_links:
                        try:
                            href = link.get_attribute('href')
                            if not href:
                                continue
                            
                            # Skip DuckDuckGo's own URLs, search URLs, and internal links
                            href_lower = href.lower()
                            if ('duckduckgo.com' in href_lower or 
                                href.startswith('javascript:') or 
                                href.startswith('#') or
                                '?q=' in href_lower or  # Skip search query URLs
                                '/?q=' in href_lower):  # Skip search query URLs
                                continue
                            
                            # Check if it's a drugs.com URL (must be in the domain, not query string)
                            # The URL must start with http and contain drugs.com as the domain
                            if (href.startswith('http') and 
                                'drugs.com' in href_lower and 
                                'dosage' in href_lower):
                                
                                # Extract domain to verify it's actually drugs.com
                                try:
                                    from urllib.parse import urlparse
                                    parsed = urlparse(href)
                                    domain = parsed.netloc.lower()
                                    
                                    # Must be drugs.com domain (not in query string)
                                    if 'drugs.com' in domain and 'dosage' in href_lower:
                                        # Clean the URL - remove any query parameters that might be added
                                        url = href.split('?')[0] if '?' in href else href
                                        
                                        # Final verification
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
            
            # If we found URLs but none matched, log them for debugging
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


def _scrape_drugs_com_page(url: str, driver: Any) -> Optional[str]:
    """
    Scrape content from drugs.com page using Selenium, extracting only from #content element.
    
    Args:
        url: URL of the drugs.com page
        driver: Selenium WebDriver instance
        
    Returns:
        Full text content from #content element, or None if error
    """
    if not driver:
        return None
    
    try:
        logger.info(f"Navigating directly to {url}...")
        driver.get(url)
        
        # Wait for page to load
        time.sleep(random.uniform(2.0, 3.5))
        
        # Find and extract content from #content element
        try:
            content_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "content"))
            )
            
            # Get text content from the #content element
            text = content_element.text
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Extracted {len(text)} characters from #content element")
            return text
            
        except Exception as e:
            logger.warning(f"Could not find #content element, trying fallback: {e}")
            # Fallback: get all page content if #content not found
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Extracted {len(text)} characters (fallback method)")
        return text
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None


def _chunk_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
    """
    Split text into chunks with overlap to avoid losing context at boundaries.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence boundary if possible
        if end < len(text):
            # Look for sentence endings near the end
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = chunk.rfind(punct)
                if last_punct > chunk_size * 0.7:  # If found in last 30% of chunk
                    chunk = chunk[:last_punct + 1]
                    end = start + len(chunk)
                    break
        
        chunks.append(chunk)
        start = end - overlap  # Overlap to maintain context
        
        if start >= len(text):
            break
    
    return chunks


# Configure DSPy once at module level to avoid thread-local issues
_dspy_configured = False
_dspy_lm = None

def _configure_dspy_once():
    """Configure DSPy once, thread-safe."""
    global _dspy_configured, _dspy_lm
    if not _dspy_configured and DSPY_AVAILABLE:
        try:
            from config import get_ollama_config
            ollama_config = get_ollama_config()
            
            # Get model name and base URL from config
            model = ollama_config['model'].replace('ollama/', '')
            api_base = ollama_config['api_base']
            
            # Configure DSPy to use Ollama directly via dspy.LM
            _dspy_lm = dspy.LM(f'ollama/{model}', api_base=api_base)
            dspy.configure(lm=_dspy_lm)
            _dspy_configured = True
            logger.debug("DSPy configured successfully")
        except Exception as e:
            logger.warning(f"Failed to configure DSPy: {e}")


# Configure DSPy once in main thread (DSPy is not thread-safe)
_dspy_configured = False

def _configure_dspy():
    """Configure DSPy once in the main thread."""
    global _dspy_configured
    if _dspy_configured:
        return
    
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available")
        return
    
    try:
        from config import get_ollama_config
        ollama_config = get_ollama_config()
        
        # Get model name and base URL from config
        model = ollama_config['model'].replace('ollama/', '')
        api_base = ollama_config['api_base']
        
        # Configure DSPy to use Ollama directly via dspy.LM
        lm = dspy.LM(f'ollama/{model}', api_base=api_base)
        dspy.configure(lm=lm)
        _dspy_configured = True
        logger.debug("DSPy configured successfully")
    except Exception as e:
        logger.warning(f"Failed to configure DSPy: {e}")


def _extract_fields_with_dspy(
    page_content: str,
    medical_name: str,
    missing_fields: List[str],
    age: Optional[int] = None,
    icd_code_names: Optional[str] = None,
    resistance_gene: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Use DSPy to extract missing fields from drugs.com page content.
    Handles large pages by chunking the content and processing chunks.
    
    Args:
        page_content: Extracted content from drugs.com page
        medical_name: Name of the antibiotic
        missing_fields: List of field names that are missing (e.g., ['dose_duration', 'route_of_administration'])
        age: Patient age (optional)
        icd_code_names: Transformed ICD code names (comma-separated, optional)
        resistance_gene: Resistance gene name (optional)
        
    Returns:
        Dictionary with extracted field values
    """
    if not DSPY_AVAILABLE:
        logger.warning("DSPy not available, cannot extract fields")
        return {}
    
    try:
        # Ensure DSPy is configured (only in main thread)
        _configure_dspy()
        
        if not _dspy_configured:
            logger.warning("DSPy configuration failed, cannot extract fields")
            return {}
        
        # Define extraction signature
        class ExtractDosageInfo(dspy.Signature):
            """Extract dosage information from drugs.com page content for an antibiotic."""
            
            page_content: str = dspy.InputField(desc="Content from drugs.com dosage page")
            antibiotic_name: str = dspy.InputField(desc="Name of the antibiotic")
            patient_age: str = dspy.InputField(desc="Patient age or 'adult'")
            icd_code_names: str = dspy.InputField(desc="ICD code names (disease conditions) or 'none'")
            resistance_gene: str = dspy.InputField(desc="Resistance gene name or 'none'")
            missing_fields: str = dspy.InputField(desc="Comma-separated list of fields to extract")
            
            dose_duration: str = dspy.OutputField(desc="Dosing information in format 'dose,route,frequency,duration' or null if not found")
            route_of_administration: str = dspy.OutputField(desc="Route of administration: 'IV', 'PO', 'IM', 'IV/PO', or null")
            general_considerations: str = dspy.OutputField(desc="Concise clinical notes/considerations in plain text (max 200 chars) or null if not found. Extract only key points like monitoring requirements, contraindications, special precautions - be brief and factual. Use plain text, not bullet points.")
            coverage_for: str = dspy.OutputField(desc="What conditions/infections this antibiotic covers or null if not found")
            renal_adjustment: str = dspy.OutputField(desc="Renal adjustment/dosing guidelines or null if not found")
        
        # Create DSPy module
        extractor = dspy.ChainOfThought(ExtractDosageInfo)
        
        # Prepare input
        patient_age_str = f"{age} years" if age else "adult"
        missing_fields_str = ", ".join(missing_fields)
        icd_code_names_str = icd_code_names if icd_code_names else "none"
        resistance_gene_str = resistance_gene if resistance_gene else "none"
        
        # Chunk the content if it's too large
        chunk_size = 6000  # Characters per chunk
        chunks = _chunk_text(page_content, chunk_size=chunk_size, overlap=500)
        
        logger.info(f"[DSPy] Processing {len(chunks)} chunks for {medical_name} (total length: {len(page_content)} chars)")
        
        # Process each chunk and accumulate results
        all_results = {
            'dose_duration': [],
            'route_of_administration': [],
            'general_considerations': [],
            'coverage_for': [],
            'renal_adjustment': []
        }
        
        for i, chunk in enumerate(chunks):
            try:
                logger.debug(f"[DSPy] Processing chunk {i+1}/{len(chunks)} for {medical_name}")
                
                # Extract fields from this chunk
                result = extractor(
                    page_content=chunk,
                    antibiotic_name=medical_name,
                    patient_age=patient_age_str,
                    icd_code_names=icd_code_names_str,
                    resistance_gene=resistance_gene_str,
                    missing_fields=missing_fields_str
                )
                
                # Collect non-null results
                if hasattr(result, 'dose_duration') and result.dose_duration and result.dose_duration.lower() not in ['null', 'none', 'not found', '']:
                    all_results['dose_duration'].append(result.dose_duration)
                if hasattr(result, 'route_of_administration') and result.route_of_administration and result.route_of_administration.lower() not in ['null', 'none', 'not found', '']:
                    all_results['route_of_administration'].append(result.route_of_administration)
                if hasattr(result, 'general_considerations') and result.general_considerations and result.general_considerations.lower() not in ['null', 'none', 'not found', '']:
                    all_results['general_considerations'].append(result.general_considerations)
                if hasattr(result, 'coverage_for') and result.coverage_for and result.coverage_for.lower() not in ['null', 'none', 'not found', '']:
                    all_results['coverage_for'].append(result.coverage_for)
                if hasattr(result, 'renal_adjustment') and result.renal_adjustment and result.renal_adjustment.lower() not in ['null', 'none', 'not found', '']:
                    all_results['renal_adjustment'].append(result.renal_adjustment)
                    
            except Exception as e:
                logger.warning(f"[DSPy] Error processing chunk {i+1} for {medical_name}: {e}")
                continue
        
        # Merge results: use first non-null value found, or combine if multiple found
        extracted = {}
        
        if 'dose_duration' in missing_fields:
            if all_results['dose_duration']:
                extracted['dose_duration'] = all_results['dose_duration'][0]
            else:
                extracted['dose_duration'] = None
        
        if 'route_of_administration' in missing_fields:
            if all_results['route_of_administration']:
                extracted['route_of_administration'] = all_results['route_of_administration'][0]
            else:
                extracted['route_of_administration'] = None
        
        if 'general_considerations' in missing_fields:
            if all_results['general_considerations']:
                considerations = all_results['general_considerations']
                considerations.sort(key=len)
                if len(considerations) == 1:
                    extracted['general_considerations'] = considerations[0]
                else:
                    combined = " ".join(considerations[:2])
                    if len(combined) > 300:
                        extracted['general_considerations'] = considerations[0]
                    else:
                        extracted['general_considerations'] = combined
            else:
                extracted['general_considerations'] = None
        
        if 'coverage_for' in missing_fields:
            if all_results['coverage_for']:
                extracted['coverage_for'] = all_results['coverage_for'][0]
            else:
                extracted['coverage_for'] = None
        
        if 'renal_adjustment' in missing_fields:
            if all_results['renal_adjustment']:
                extracted['renal_adjustment'] = all_results['renal_adjustment'][0]
            else:
                extracted['renal_adjustment'] = None
        
        logger.info(f"[DSPy] Extracted fields for {medical_name}: {[k for k, v in extracted.items() if v]}")
        return extracted
        
    except Exception as e:
        logger.error(f"Error extracting fields with DSPy for {medical_name}: {e}")
        # Fallback to LangChain if DSPy fails
        try:
            from config import get_ollama_llm
            llm = get_ollama_llm()
            return _extract_fields_with_langchain(page_content, medical_name, missing_fields, age, llm, icd_code_names, resistance_gene)
        except:
            return {}


def _extract_fields_with_langchain(
    page_content: str,
    medical_name: str,
    missing_fields: List[str],
    age: Optional[int],
    llm: Any,
    icd_code_names: Optional[str] = None,
    resistance_gene: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Fallback: Use LangChain structured output to extract fields.
    Handles large pages by chunking the content.
    
    Args:
        page_content: Extracted content from drugs.com page
        medical_name: Name of the antibiotic
        missing_fields: List of field names that are missing
        age: Patient age (optional)
        llm: LangChain BaseChatModel
        icd_code_names: Transformed ICD code names (comma-separated, optional)
        resistance_gene: Resistance gene name (optional)
        
    Returns:
        Dictionary with extracted field values
    """
    try:
        from pydantic import BaseModel, Field
        
        class DosageExtractionResult(BaseModel):
            """Schema for extracted dosage information."""
            dose_duration: Optional[str] = Field(None, description="Dosing information in format 'dose,route,frequency,duration'")
            route_of_administration: Optional[str] = Field(None, description="Route: 'IV', 'PO', 'IM', 'IV/PO', or null")
            general_considerations: Optional[str] = Field(None, description="Concise clinical notes/considerations in plain text (max 200 chars). Extract only key points like monitoring requirements, contraindications, special precautions - be brief and factual. Use plain text, not bullet points.")
            coverage_for: Optional[str] = Field(None, description="What conditions/infections this antibiotic covers")
            renal_adjustment: Optional[str] = Field(None, description="Renal adjustment/dosing guidelines")
        
        patient_age_str = f"{age} years" if age else "adult"
        missing_fields_str = ", ".join(missing_fields)
        icd_code_names_str = icd_code_names if icd_code_names else "none"
        resistance_gene_str = resistance_gene if resistance_gene else "none"
        
        # Chunk the content if it's too large
        chunk_size = 6000  # Characters per chunk
        chunks = _chunk_text(page_content, chunk_size=chunk_size, overlap=500)
        
        logger.info(f"[LangChain] Processing {len(chunks)} chunks for {medical_name} (total length: {len(page_content)} chars)")
        
        # Process each chunk and accumulate results
        all_results = {
            'dose_duration': [],
            'route_of_administration': [],
            'general_considerations': [],
            'coverage_for': [],
            'renal_adjustment': []
        }
        
        structured_llm = llm.with_structured_output(DosageExtractionResult)
        
        for i, chunk in enumerate(chunks):
            try:
                logger.debug(f"[LangChain] Processing chunk {i+1}/{len(chunks)} for {medical_name}")
                
                prompt = f"""Extract dosage information for {medical_name} from the following drugs.com page content.

Antibiotic Name: {medical_name}
Patient Age: {patient_age_str}
ICD Code Names (Disease Conditions): {icd_code_names_str}
Resistance Gene: {resistance_gene_str}
Missing Fields to Extract: {missing_fields_str}

PAGE CONTENT (chunk {i+1} of {len(chunks)}):
{chunk}

Extract the following fields:
- dose_duration: Dosing information in format 'dose,route,frequency,duration' or null if not found
- route_of_administration: 'IV', 'PO', 'IM', 'IV/PO', or null if not found
- general_considerations: Concise clinical notes/considerations in plain text (max 200 chars) or null if not found. Extract only key points like monitoring requirements, contraindications, special precautions - be brief and factual. Use plain text, not bullet points.
- coverage_for: What conditions/infections this antibiotic covers or null if not found
- renal_adjustment: Renal adjustment/dosing guidelines or null if not found

Only extract fields that are in the missing_fields list. If a field is not missing, return null for it."""
                
                result = structured_llm.invoke(prompt)
                
                # Collect non-null results
                if result.dose_duration:
                    all_results['dose_duration'].append(result.dose_duration)
                if result.route_of_administration:
                    all_results['route_of_administration'].append(result.route_of_administration)
                if result.general_considerations:
                    all_results['general_considerations'].append(result.general_considerations)
                if result.coverage_for:
                    all_results['coverage_for'].append(result.coverage_for)
                if result.renal_adjustment:
                    all_results['renal_adjustment'].append(result.renal_adjustment)
                    
            except Exception as e:
                logger.warning(f"[LangChain] Error processing chunk {i+1} for {medical_name}: {e}")
                continue
        
        # Merge results: use first non-null value found, or combine if multiple found
        extracted = {}
        
        if 'dose_duration' in missing_fields:
            if all_results['dose_duration']:
                extracted['dose_duration'] = all_results['dose_duration'][0]
            else:
                extracted['dose_duration'] = None
        
        if 'route_of_administration' in missing_fields:
            if all_results['route_of_administration']:
                extracted['route_of_administration'] = all_results['route_of_administration'][0]
            else:
                extracted['route_of_administration'] = None
        
        if 'general_considerations' in missing_fields:
            if all_results['general_considerations']:
                # For general_considerations, prefer concise entries
                considerations = all_results['general_considerations']
                # Sort by length (shorter = more concise)
                considerations.sort(key=len)
                # Take the shortest one, or combine first 2 if they're both short
                if len(considerations) == 1:
                    extracted['general_considerations'] = considerations[0]
                else:
                    # Combine first 2 shortest, but limit total length
                    combined = " ".join(considerations[:2])
                    if len(combined) > 300:
                        # If too long, just take the shortest
                        extracted['general_considerations'] = considerations[0]
                    else:
                        extracted['general_considerations'] = combined
            else:
                extracted['general_considerations'] = None
        
        if 'coverage_for' in missing_fields:
            if all_results['coverage_for']:
                extracted['coverage_for'] = all_results['coverage_for'][0]
            else:
                extracted['coverage_for'] = None
        
        if 'renal_adjustment' in missing_fields:
            if all_results['renal_adjustment']:
                extracted['renal_adjustment'] = all_results['renal_adjustment'][0]
            else:
                extracted['renal_adjustment'] = None
        
        logger.info(f"[LangChain] Extracted fields for {medical_name}: {[k for k, v in extracted.items() if v]}")
        return extracted
        
    except Exception as e:
        logger.error(f"Error extracting fields with LangChain for {medical_name}: {e}")
        return {}


def _scrape_antibiotic_page(
    antibiotic: Dict[str, Any],
    category: str,
    idx: int
) -> Tuple[str, int, Optional[str], List[str]]:
    """
    Scrape page content for a single antibiotic in a separate thread with its own browser.
    Only does Selenium scraping - no DSPy extraction (that happens in main thread).
    
    Args:
        antibiotic: Antibiotic dictionary
        category: Category name (first_choice, second_choice, alternative_antibiotic)
        idx: Index of the antibiotic in the list
        
    Returns:
        Tuple of (category, idx, page_content, missing_fields) where page_content is None if scraping failed
    """
    medical_name = antibiotic.get('medical_name', '')
    if not medical_name:
        return (category, idx, None, [])
    
    # Determine which fields are missing
    # Check if ANY of these fields are null: coverage_for, route_of_administration, renal_adjustment, general_considerations
    # If any are null, we need to extract ALL fields for data interoperability
    coverage_for = antibiotic.get('coverage_for')
    route_of_administration = antibiotic.get('route_of_administration')
    renal_adjustment = antibiotic.get('renal_adjustment')
    general_considerations = antibiotic.get('general_considerations')
    dose_duration = antibiotic.get('dose_duration')
    
    # Check if any of the 4 key fields are null
    needs_full_extraction = (
        coverage_for is None or 
        route_of_administration is None or 
        renal_adjustment is None or 
        general_considerations is None
    )
    
    missing_fields = []
    if dose_duration is None:
        missing_fields.append('dose_duration')
    if route_of_administration is None:
        missing_fields.append('route_of_administration')
    if general_considerations is None:
        missing_fields.append('general_considerations')
    if coverage_for is None:
        missing_fields.append('coverage_for')
    if renal_adjustment is None:
        missing_fields.append('renal_adjustment')
    
    # If any of the 4 key fields are null, extract all fields
    if needs_full_extraction:
        missing_fields = ['dose_duration', 'route_of_administration', 'general_considerations', 'coverage_for', 'renal_adjustment']
    
    if not missing_fields:
        return (category, idx, None, [])
    
    driver = None
    try:
        # Create a new driver for this thread
        driver = _get_selenium_driver()
        if not driver:
            logger.error(f"[Thread] Could not create Selenium driver for {medical_name}")
            return (category, idx, None, missing_fields)
        
        logger.info(f"[Thread] Scraping {medical_name} from drugs.com (missing: {', '.join(missing_fields)})...")
        
        # Search Google using Selenium for drugs.com dosage URL
        drugs_com_url = _google_search_drugs_com_selenium(medical_name, driver)
        
        if drugs_com_url:
            # Navigate directly to the URL
            logger.info(f"[Thread] Navigating directly to {drugs_com_url}")
            
            # Scrape the page content from #content element
            page_content = _scrape_drugs_com_page(drugs_com_url, driver)
            
            if page_content:
                logger.info(f"[Thread] Successfully scraped page for {medical_name}")
                return (category, idx, page_content, missing_fields)
            else:
                logger.warning(f"[Thread] Could not scrape page for {medical_name}")
                return (category, idx, None, missing_fields)
        else:
            logger.warning(f"[Thread] Could not find drugs.com dosage URL for {medical_name}")
            return (category, idx, None, missing_fields)
            
    except Exception as e:
        logger.error(f"[Thread] Error scraping {medical_name}: {e}", exc_info=True)
        return (category, idx, None, missing_fields)
    finally:
        # Always close the driver
        if driver:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"[Thread] Error closing Selenium driver for {medical_name}: {e}")


def enrichment_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrichment node that extracts content from drugs.com for antibiotics with missing fields.
    Takes input from synthesize_node output (result field).
    Finds entries where dose_duration OR route_of_administration OR general_considerations is null.
    Uses Selenium to search Google and find drugs.com dosage pages.
    Processes 3 antibiotics in parallel using multiple browsers.
    Drops records if no valid drugs.com dosage URL is found.
    
    Args:
        state: Pipeline state dictionary (should have 'result' from synthesize_node)
        
    Returns:
        Updated state with enriched result (records without valid drugs.com URLs are removed)
    """
    try:
        result = state.get('result', {})
        if not result:
            logger.warning("No result to enrich")
            return {'result': result}
        
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium/undetected-chromedriver is not available. Cannot perform enrichment.")
            logger.error("Please install undetected-chromedriver: pip install undetected-chromedriver")
            logger.error("The enrichment node requires undetected-chromedriver to scrape drugs.com pages.")
            return {'result': result}
        
        therapy_plan = result.get('antibiotic_therapy_plan', {})
        
        # Collect all antibiotics that need processing
        antibiotics_to_process = []
        
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            antibiotics = therapy_plan.get(category, [])
            if not isinstance(antibiotics, list):
                continue
            
            for idx, antibiotic in enumerate(antibiotics):
                medical_name = antibiotic.get('medical_name', '')
                if not medical_name:
                    continue
                
                # Check if ANY of these fields are null: coverage_for, route_of_administration, renal_adjustment, general_considerations
                # If any are null, we need to extract ALL fields for data interoperability
                coverage_for = antibiotic.get('coverage_for')
                route_of_administration = antibiotic.get('route_of_administration')
                renal_adjustment = antibiotic.get('renal_adjustment')
                general_considerations = antibiotic.get('general_considerations')
                dose_duration = antibiotic.get('dose_duration')
                
                # Check if any of the 4 key fields are null OR if dose_duration is null
                if (coverage_for is None or 
                    route_of_administration is None or 
                    renal_adjustment is None or 
                    general_considerations is None or
                    dose_duration is None):
                    antibiotics_to_process.append((antibiotic, category, idx))
        
        if not antibiotics_to_process:
            logger.info("No antibiotics need enrichment")
            return {'result': result}
        
        logger.info(f"Processing {len(antibiotics_to_process)} antibiotics in parallel (3 at a time)...")
        
        # Process antibiotics in parallel (3 at a time)
        antibiotics_to_remove = {
            'first_choice': [],
            'second_choice': [],
            'alternative_antibiotic': []
        }
        
        input_params = state.get('input_parameters', {})
        
        # Get ICD code names from transformed state
        icd_transformation = state.get('icd_transformation', {})
        icd_code_names = None
        if icd_transformation and icd_transformation.get('severity_codes_transformed'):
            icd_code_names = icd_transformation['severity_codes_transformed']
        
        # Get resistance gene from input parameters and format (handle comma-separated)
        resistance_gene_raw = input_params.get('resistant_gene', '')
        resistance_gene = format_resistance_genes(resistance_gene_raw) if resistance_gene_raw else None
        
        # Configure DSPy once before processing
        _configure_dspy()
        
        age = input_params.get('age')
        
        # Separate antibiotics by category
        first_choice_ab = [(ab, cat, idx) for ab, cat, idx in antibiotics_to_process if cat == 'first_choice']
        second_choice_ab = [(ab, cat, idx) for ab, cat, idx in antibiotics_to_process if cat == 'second_choice']
        alternative_ab = [(ab, cat, idx) for ab, cat, idx in antibiotics_to_process if cat == 'alternative_antibiotic']
        
        # Process first_choice and second_choice sequentially
        for antibiotic, category, idx in first_choice_ab + second_choice_ab:
            medical_name = antibiotic.get('medical_name', 'unknown')
            try:
                logger.info(f"Processing {medical_name}...")
                
                # Step 1: Scrape page content
                logger.info(f"  [1/2] Scraping {medical_name} from drugs.com...")
                category_result, idx_result, page_content, missing_fields = _scrape_antibiotic_page(antibiotic, category, idx)
                
                if not page_content:
                    logger.warning(f"  Could not scrape {medical_name}, marking for removal")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                # Step 2: Extract fields using DSPy
                logger.info(f"  [2/2] Extracting fields for {medical_name}...")
                extracted_fields = _extract_fields_with_dspy(
                    page_content=page_content,
                    medical_name=medical_name,
                    missing_fields=missing_fields,
                    age=age,
                    icd_code_names=icd_code_names,
                    resistance_gene=resistance_gene
                )
                
                # Update antibiotic with extracted fields
                if 'dose_duration' in extracted_fields and extracted_fields['dose_duration']:
                    antibiotic['dose_duration'] = extracted_fields['dose_duration']
                    logger.info(f"  ✓ Updated dose_duration for {medical_name}")
                
                if 'route_of_administration' in extracted_fields and extracted_fields['route_of_administration']:
                    antibiotic['route_of_administration'] = extracted_fields['route_of_administration']
                    logger.info(f"  ✓ Updated route_of_administration for {medical_name}")
                
                if 'general_considerations' in extracted_fields and extracted_fields['general_considerations']:
                    antibiotic['general_considerations'] = extracted_fields['general_considerations']
                    logger.info(f"  ✓ Updated general_considerations for {medical_name}")
                
                if 'coverage_for' in extracted_fields and extracted_fields['coverage_for']:
                    antibiotic['coverage_for'] = extracted_fields['coverage_for']
                    logger.info(f"  ✓ Updated coverage_for for {medical_name}")
                
                if 'renal_adjustment' in extracted_fields and extracted_fields['renal_adjustment']:
                    antibiotic['renal_adjustment'] = extracted_fields['renal_adjustment']
                    logger.info(f"  ✓ Updated renal_adjustment for {medical_name}")
                
                logger.info(f"  ✓ Completed {medical_name}")
                
            except Exception as e:
                logger.error(f"Error processing {medical_name}: {e}", exc_info=True)
                antibiotics_to_remove[category].append(idx)
        
        # Process alternative_antibiotic: scrape all concurrently, then select top 5, then process sequentially
        if alternative_ab:
            logger.info(f"Processing {len(alternative_ab)} alternative_antibiotic: scraping all concurrently, then selecting top 5...")
            
            # Step 1: Scrape all alternative_antibiotic concurrently
            scraped_alternative = {}  # {(category, idx): (page_content, missing_fields, antibiotic, num_chunks)}
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_antibiotic = {
                    executor.submit(_scrape_antibiotic_page, ab, cat, idx): (ab, cat, idx)
                    for ab, cat, idx in alternative_ab
                }
                
                for future in as_completed(future_to_antibiotic):
                    try:
                        category, idx, page_content, missing_fields = future.result()
                        ab, cat, idx_orig = future_to_antibiotic[future]
                        
                        if page_content:
                            # Calculate number of chunks (fewer chunks = better)
                            chunks = _chunk_text(page_content, chunk_size=6000, overlap=500)
                            num_chunks = len(chunks)
                            scraped_alternative[(category, idx)] = (page_content, missing_fields, ab, num_chunks)
                        else:
                            antibiotics_to_remove[category].append(idx)
                    except Exception as e:
                        logger.error(f"Error getting scraping result: {e}")
                        if future in future_to_antibiotic:
                            ab, cat, idx = future_to_antibiotic[future]
                            antibiotics_to_remove[cat].append(idx)
                        else:
                            logger.warning(f"Future not found in future_to_antibiotic mapping")
            
            # Step 2: Select top 5 with fewer chunks
            if scraped_alternative:
                # Sort by number of chunks (ascending - fewer chunks first)
                sorted_alternatives = sorted(
                    scraped_alternative.items(),
                    key=lambda x: x[1][3] if len(x[1]) > 3 else float('inf')  # Sort by num_chunks (4th element in tuple)
                )
                
                # Take top 5
                top_5 = sorted_alternatives[:5]
                logger.info(f"Selected top 5 alternative_antibiotic (with fewer chunks): {[ab.get('medical_name', 'unknown') for _, (_, _, ab, _) in top_5]}")
                
                # Step 3: Process top 5 sequentially
                for (category, idx), (page_content, missing_fields, antibiotic, num_chunks) in top_5:
                    medical_name = antibiotic.get('medical_name', 'unknown')
                    try:
                        logger.info(f"Processing {medical_name} (alternative_antibiotic, {num_chunks} chunks)...")
                        
                        # Extract fields using DSPy
                        extracted_fields = _extract_fields_with_dspy(
                            page_content=page_content,
                            medical_name=medical_name,
                            missing_fields=missing_fields,
                            age=age,
                            icd_code_names=icd_code_names,
                            resistance_gene=resistance_gene
                        )
                        
                        # Update antibiotic with extracted fields
                        if 'dose_duration' in extracted_fields and extracted_fields['dose_duration']:
                            antibiotic['dose_duration'] = extracted_fields['dose_duration']
                            logger.info(f"  ✓ Updated dose_duration for {medical_name}")
                        
                        if 'route_of_administration' in extracted_fields and extracted_fields['route_of_administration']:
                            antibiotic['route_of_administration'] = extracted_fields['route_of_administration']
                            logger.info(f"  ✓ Updated route_of_administration for {medical_name}")
                        
                        if 'general_considerations' in extracted_fields and extracted_fields['general_considerations']:
                            antibiotic['general_considerations'] = extracted_fields['general_considerations']
                            logger.info(f"  ✓ Updated general_considerations for {medical_name}")
                        
                        if 'coverage_for' in extracted_fields and extracted_fields['coverage_for']:
                            antibiotic['coverage_for'] = extracted_fields['coverage_for']
                            logger.info(f"  ✓ Updated coverage_for for {medical_name}")
                        
                        if 'renal_adjustment' in extracted_fields and extracted_fields['renal_adjustment']:
                            antibiotic['renal_adjustment'] = extracted_fields['renal_adjustment']
                            logger.info(f"  ✓ Updated renal_adjustment for {medical_name}")
                        
                        logger.info(f"  ✓ Completed {medical_name}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {medical_name}: {e}", exc_info=True)
                        antibiotics_to_remove[category].append(idx)
                
                # Remove the alternative_antibiotic that weren't in top 5
                top_5_indices = {idx for (_, idx), _ in top_5}
                for (category, idx), _ in scraped_alternative.items():
                    if idx not in top_5_indices:
                        antibiotics_to_remove[category].append(idx)
                        # Safely get antibiotic name
                        antibiotic_data = scraped_alternative.get((category, idx))
                        if antibiotic_data and len(antibiotic_data) >= 3:
                            ab_name = antibiotic_data[2].get('medical_name', 'unknown') if isinstance(antibiotic_data[2], dict) else 'unknown'
                        else:
                            ab_name = 'unknown'
                        logger.info(f"Removed {ab_name} (not in top 5)")
        
        # Remove antibiotics that didn't have valid drugs.com URLs
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            antibiotics = therapy_plan.get(category, [])
            if isinstance(antibiotics, list):
                # Remove in reverse order to maintain indices
                for idx in sorted(antibiotics_to_remove[category], reverse=True):
                    if 0 <= idx < len(antibiotics):
                        removed_ab = antibiotics.pop(idx)
                        ab_name = removed_ab.get('medical_name', 'unknown') if isinstance(removed_ab, dict) else 'unknown'
                        logger.info(f"Removed {ab_name} from {category} (no valid drugs.com URL)")
                    else:
                        logger.warning(f"Invalid index {idx} for {category} (list length: {len(antibiotics)})")
        
        logger.info("Enrichment complete")
        return {'result': result}
        
    except Exception as e:
        logger.error(f"Error in enrichment_node: {e}", exc_info=True)
        raise

