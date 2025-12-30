"""
Enrichment node for LangGraph - Enriches missing dosage information from drugs.com using Selenium.
Uses LlamaIndex for structured extraction.
Only processes entries where is_complete is False.
"""
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import quote_plus
from pathlib import Path
import time
import random
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from utils import format_resistance_genes, get_icd_names_from_state, create_llm
from schemas import AntibioticMatchResult, DosageExtractionResult
from prompts import ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE, DOSAGE_EXTRACTION_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    from llama_index.core.node_parser import SentenceSplitter
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None
    SentenceSplitter = None

# NLTK imports with fallback
try:
    import nltk
    NLTK_AVAILABLE = True
    # Download punkt tokenizer if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
except ImportError:
    logger.error("NLTK not available. Install: pip install nltk")
    NLTK_AVAILABLE = False
except Exception as e:
    logger.warning(f"NLTK setup issue: {e}")
    NLTK_AVAILABLE = False

logger = logging.getLogger(__name__)

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
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        driver = webdriver.Chrome(options=chrome_options)
        
        # Set reasonable timeouts (will retry on failure)
        driver.set_page_load_timeout(120)
        driver.implicitly_wait(10)
        driver.set_script_timeout(120)
        
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
        return None
    
    try:
        search_query = f"{antibiotic_name} dosage drug.com"
        logger.info(f"Searching DuckDuckGo for {antibiotic_name}...")
        
        # Retry page load for DuckDuckGo
        max_retries = 3
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                driver.get("https://duckduckgo.com/")
                time.sleep(random.uniform(1.5, 3.0))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"DuckDuckGo page load failed (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"DuckDuckGo page load failed after {max_retries} attempts: {e}")
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
            driver.get(duckduckgo_url)
            time.sleep(2)
        
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
        
        # Retry page load with exponential backoff
        max_retries = 3
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                driver.get(url)
                time.sleep(random.uniform(1.5, 2.5))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Page load failed for {url} (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"Page load failed after {max_retries} attempts for {url}: {e}, continuing with title check")
        
        # Get page title
        page_title = driver.title if driver.title else ""
        
        if not page_title:
            logger.warning(f"No page title found for {url}, skipping validation")
            return True  # Fail open if no title
        
        # Format prompt
        prompt = ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE.format(
            antibiotic_name=antibiotic_name,
            page_title=page_title
        )
        
        # Use LlamaIndex for structured extraction
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=AntibioticMatchResult,
            llm=llm,
            prompt_template_str="{input_str}",
            verbose=False
        )
        
        result = program(input_str=prompt)
        
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


def _scrape_drugs_com_page(url: str, driver: Any) -> Optional[str]:
    """
    Scrape content from drugs.com page using Selenium.
    
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
        
        # Retry page load with exponential backoff
        max_retries = 3
        retry_delay = 2.0
        for attempt in range(max_retries):
            try:
                driver.get(url)
                time.sleep(random.uniform(2.0, 3.5))
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Page load failed for {url} (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    logger.warning(f"Page load failed after {max_retries} attempts for {url}: {e}")
                    return None
        
        # Retry waiting for content element
        max_wait_retries = 3
        content_element = None
        for attempt in range(max_wait_retries):
            try:
                content_element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, "content"))
                )
                break
            except Exception as e:
                if attempt < max_wait_retries - 1:
                    logger.warning(f"Content element not found (attempt {attempt + 1}/{max_wait_retries}): {e}, retrying...")
                    time.sleep(retry_delay)
                else:
                    logger.warning(f"Content element not found after {max_wait_retries} attempts: {e}")
        
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
                return text
            except Exception as e:
                logger.error(f"Fallback extraction also failed: {e}")
                return None
        
        text = content_element.text
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        logger.info(f"Extracted {len(text)} characters from #content element")
        return text
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None
        
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return None


def _chunk_text_with_llamaindex(text: str, chunk_size: int = 6000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into chunks using LlamaIndex's SentenceSplitter with NLTK.
    Maintains sentence boundaries and overlap for cross-chunk context.
    
    Args:
        text: Text to chunk
        chunk_size: Maximum size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not LLAMAINDEX_AVAILABLE or not NLTK_AVAILABLE:
        logger.warning("LlamaIndex or NLTK not available, using fallback chunking")
        return _chunk_text_fallback(text, chunk_size, chunk_overlap)
    
    try:
        # Create SentenceSplitter with specified chunk size and overlap
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n"
        )
        
        # Split text into nodes (chunks)
        from llama_index.core.schema import Document
        doc = Document(text=text)
        nodes = splitter.get_nodes_from_documents([doc])
        
        # Extract text from nodes
        chunks = [node.text for node in nodes]
        
        logger.debug(f"Split text into {len(chunks)} chunks using LlamaIndex SentenceSplitter")
        return chunks
        
    except Exception as e:
        logger.warning(f"Error using LlamaIndex SentenceSplitter: {e}, using fallback")
        return _chunk_text_fallback(text, chunk_size, chunk_overlap)


def _chunk_text_fallback(text: str, chunk_size: int = 6000, overlap: int = 200) -> List[str]:
    """
    Fallback chunking method if LlamaIndex/NLTK is not available.
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
        
        if end < len(text):
            # Try to break at sentence boundaries
            for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_punct = chunk.rfind(punct)
                if last_punct > chunk_size * 0.7:
                    chunk = chunk[:last_punct + 1]
                    end = start + len(chunk)
                    break
        
        chunks.append(chunk)
        start = end - overlap
        
        if start >= len(text):
            break
    
    return chunks


def _extract_fields_with_llamaindex(
    page_content: str,
    medical_name: str,
    missing_fields: List[str],
    existing_data: Dict[str, Any],
    age: Optional[int],
    icd_code_names: Optional[str] = None,
    resistance_gene: Optional[str] = None,
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
        
        # Build existing data context
        existing_data_context = ""
        if existing_data:
            existing_fields = []
            if existing_data.get('dose_duration'):
                existing_fields.append(f"dose_duration={existing_data['dose_duration']}")
            if existing_data.get('route_of_administration'):
                existing_fields.append(f"route_of_administration={existing_data['route_of_administration']}")
            if existing_data.get('coverage_for'):
                existing_fields.append(f"coverage_for={existing_data['coverage_for']}")
            if existing_data.get('renal_adjustment'):
                existing_fields.append(f"renal_adjustment={existing_data['renal_adjustment']}")
            if existing_data.get('general_considerations'):
                existing_fields.append(f"general_considerations={existing_data['general_considerations'][:100]}")
            
            if existing_fields:
                existing_data_context = "\n".join(existing_fields)
        
        # Chunk the content using LlamaIndex SentenceSplitter
        chunk_size = 6000
        chunk_overlap = 200  # Overlap for cross-chunk context
        chunks = _chunk_text_with_llamaindex(page_content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        logger.info(f"Processing {len(chunks)} chunks for {medical_name} (total length: {len(page_content)} chars)")
        
        # Process each chunk and accumulate results
        all_results = {
            'dose_duration': [],
            'route_of_administration': [],
            'general_considerations': [],
            'coverage_for': [],
            'renal_adjustment': []
        }
        
        # Track previous chunks' extracted fields for cross-chunk context
        previous_chunks_context = []
        
        attempt = 0
        for i, chunk in enumerate(chunks):
            while True:
                attempt += 1
                try:
                    logger.debug(f"Processing chunk {i+1}/{len(chunks)} for {medical_name} (attempt {attempt})")
                    
                    # Build cross-chunk context from previous chunks
                    cross_chunk_context = ""
                    if previous_chunks_context:
                        cross_chunk_context = "\n\nPREVIOUS CHUNKS CONTEXT (for consistency):\n"
                        for prev_idx, prev_fields in enumerate(previous_chunks_context[-3:], start=1):  # Last 3 chunks
                            prev_summary = ", ".join([f"{k}={v[:50]}" for k, v in prev_fields.items() if v])
                            if prev_summary:
                                cross_chunk_context += f"Chunk {prev_idx}: {prev_summary}\n"
                    
                    # Format prompt with cross-chunk context
                    prompt = DOSAGE_EXTRACTION_PROMPT_TEMPLATE.format(
                        medical_name=medical_name,
                        patient_age=patient_age_str,
                        icd_codes=icd_code_names_str,
                        gene_context=gene_context,
                        gene_matching=gene_matching,
                        missing_fields=missing_fields_str,
                        existing_data=existing_data_context,
                        cross_chunk_context=cross_chunk_context,
                        chunk_num=i+1,
                        total_chunks=len(chunks),
                        chunk_content=chunk
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
                        logger.warning(f"Empty result from chunk {i+1} for {medical_name}, retrying...")
                        time.sleep(retry_delay)
                        continue
                    
                    # Store extracted fields from this chunk for cross-chunk context
                    extracted_from_chunk = {}
                    if 'dose_duration' in missing_fields and result.dose_duration:
                        all_results['dose_duration'].append(result.dose_duration)
                        extracted_from_chunk['dose_duration'] = result.dose_duration
                    if 'route_of_administration' in missing_fields and result.route_of_administration:
                        all_results['route_of_administration'].append(result.route_of_administration)
                        extracted_from_chunk['route_of_administration'] = result.route_of_administration
                    if 'general_considerations' in missing_fields and result.general_considerations:
                        all_results['general_considerations'].append(result.general_considerations)
                        extracted_from_chunk['general_considerations'] = result.general_considerations
                    if 'coverage_for' in missing_fields and result.coverage_for:
                        all_results['coverage_for'].append(result.coverage_for)
                        extracted_from_chunk['coverage_for'] = result.coverage_for
                    if 'renal_adjustment' in missing_fields and result.renal_adjustment:
                        all_results['renal_adjustment'].append(result.renal_adjustment)
                        extracted_from_chunk['renal_adjustment'] = result.renal_adjustment
                    
                    # Store context for next chunks
                    if extracted_from_chunk:
                        previous_chunks_context.append(extracted_from_chunk)
                    
                    break  # Success, move to next chunk
                    
                except Exception as e:
                    logger.warning(f"Error processing chunk {i+1} for {medical_name} (attempt {attempt}): {e}")
                    time.sleep(retry_delay)
                    continue
        
        # Merge results: blend with existing data
        extracted = {}
        
        # For each field, use existing if present, otherwise use first extracted value
        for field in ['dose_duration', 'route_of_administration', 'coverage_for', 'renal_adjustment']:
            if field in missing_fields:
                if all_results[field]:
                    extracted[field] = all_results[field][0]
                elif existing_data.get(field):
                    # Keep existing if extraction failed
                    extracted[field] = existing_data[field]
                else:
                    extracted[field] = None
            else:
                # Not missing, preserve existing
                extracted[field] = existing_data.get(field)
        
        # Special handling for general_considerations - blend if both exist
        if 'general_considerations' in missing_fields:
            if all_results['general_considerations']:
                considerations = all_results['general_considerations']
                considerations.sort(key=len)
                
                if existing_data.get('general_considerations'):
                    # Blend with existing
                    existing_cons = existing_data['general_considerations']
                    new_cons = considerations[0]
                    # Combine if total length is reasonable
                    combined = f"{existing_cons}. {new_cons}"
                    if len(combined) <= 300:
                        extracted['general_considerations'] = combined
                    else:
                        extracted['general_considerations'] = existing_cons  # Keep existing if too long
                else:
                    # No existing, use extracted
                    if len(considerations) == 1:
                        extracted['general_considerations'] = considerations[0]
                    else:
                        combined = " ".join(considerations[:2])
                        if len(combined) > 300:
                            extracted['general_considerations'] = considerations[0]
                        else:
                            extracted['general_considerations'] = combined
            elif existing_data.get('general_considerations'):
                extracted['general_considerations'] = existing_data['general_considerations']
            else:
                extracted['general_considerations'] = None
        else:
            # Not missing, preserve existing
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
) -> Tuple[str, int, Optional[str], List[str], bool, int]:
    """
    Scrape page content for a single antibiotic in a separate thread with its own browser.
    
    Args:
        antibiotic: Antibiotic dictionary
        category: Category name (first_choice, second_choice, alternative_antibiotic)
        idx: Index of the antibiotic in the list
        
    Returns:
        Tuple of (category, idx, page_content, missing_fields, validation_failed, num_chunks) 
        where validation_failed=True means the drug name didn't match and should be removed
        num_chunks is the number of chunks the page content would be split into
    """
    medical_name = antibiotic.get('medical_name', '')
    if not medical_name:
        return (category, idx, None, [], False, 0)
    
    # Determine which fields are missing - check ALL fields that are null
    missing_fields = []
    if antibiotic.get('dose_duration') is None:
        missing_fields.append('dose_duration')
    if antibiotic.get('route_of_administration') is None:
        missing_fields.append('route_of_administration')
    if antibiotic.get('coverage_for') is None:
        missing_fields.append('coverage_for')
    if antibiotic.get('renal_adjustment') is None:
        missing_fields.append('renal_adjustment')
    if antibiotic.get('general_considerations') is None:
        missing_fields.append('general_considerations')
    
    if not missing_fields:
        return (category, idx, None, [], False, 0)
    
    driver = None
    try:
        driver = _get_selenium_driver()
        if not driver:
            logger.error(f"[Thread] Could not create Selenium driver for {medical_name}")
            return (category, idx, None, missing_fields, False, 0)
        
        logger.info(f"[Thread] Scraping {medical_name} from drugs.com (missing: {', '.join(missing_fields)})...")
        
        drugs_com_url = _google_search_drugs_com_selenium(medical_name, driver)
        
        if drugs_com_url:
            # Validate that the page is about the same antibiotic before scraping
            is_valid = _validate_antibiotic_match(drugs_com_url, medical_name, driver)
            if not is_valid:
                logger.warning(f"[Thread] Page validation failed for {medical_name} - drug name doesn't match, will remove from result")
                return (category, idx, None, missing_fields, True, 0)  # validation_failed=True
            
            logger.info(f"[Thread] Navigating directly to {drugs_com_url}")
            page_content = _scrape_drugs_com_page(drugs_com_url, driver)
            
            if page_content:
                # Calculate number of chunks
                chunks = _chunk_text_with_llamaindex(page_content, chunk_size=6000, chunk_overlap=200)
                num_chunks = len(chunks)
                logger.info(f"[Thread] Successfully scraped page for {medical_name} ({num_chunks} chunks)")
                return (category, idx, page_content, missing_fields, False, num_chunks)
            else:
                logger.warning(f"[Thread] Could not scrape page for {medical_name}")
                return (category, idx, None, missing_fields, False, 0)
        else:
            logger.warning(f"[Thread] Could not find drugs.com dosage URL for {medical_name}")
            return (category, idx, None, missing_fields, False, 0)
            
    except Exception as e:
        logger.error(f"[Thread] Error scraping {medical_name}: {e}", exc_info=True)
        return (category, idx, None, missing_fields, False, 0)
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
                missing_fields = []
                if antibiotic.get('dose_duration') is None:
                    missing_fields.append('dose_duration')
                if antibiotic.get('route_of_administration') is None:
                    missing_fields.append('route_of_administration')
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
        from utils import get_resistance_genes_from_input, format_resistance_genes
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistance_gene = format_resistance_genes(resistant_genes)  # Returns None if empty
        
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
                category_result, idx_result, page_content, scraped_missing_fields, validation_failed, num_chunks = _scrape_antibiotic_page(antibiotic, category, idx)
                
                # If validation failed (name doesn't match), mark for removal
                if validation_failed:
                    logger.warning(f"  Validation failed for {medical_name} - removing from result")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                if not page_content:
                    logger.warning(f"  Could not scrape {medical_name}, removing from result (first_choice/second_choice require extraction)")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                # Step 2: Extract fields using LlamaIndex
                logger.info(f"  [2/2] Extracting fields for {medical_name}...")
                extracted_fields = _extract_fields_with_llamaindex(
                    page_content=page_content,
                    medical_name=medical_name,
                    missing_fields=missing_fields,
                    existing_data=antibiotic,  # Pass existing data to blend with
                    age=age,
                    icd_code_names=icd_code_names,
                    resistance_gene=resistance_gene
                )
                
                # Update antibiotic with extracted fields (only update missing fields)
                updated = False
                required_fields_updated = False
                for field in missing_fields:
                    if field in extracted_fields and extracted_fields[field]:
                        antibiotic[field] = extracted_fields[field]
                        updated = True
                        # Check if this is a required field (for completeness)
                        if field in ['coverage_for', 'dose_duration', 'route_of_administration']:
                            required_fields_updated = True
                        logger.info(f"  ✓ Updated {field} for {medical_name}")
                
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
                if not updated or (missing_fields and not required_fields_updated):
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
                scraped_alternative = {}  # {(category, idx): (page_content, missing_fields, antibiotic, num_chunks, validation_failed)}
                
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future_to_antibiotic = {
                        executor.submit(_scrape_antibiotic_page, ab, cat, idx): (ab, cat, idx, mf)
                        for ab, cat, idx, mf in alternative_ab
                    }
                    
                    for future in as_completed(future_to_antibiotic):
                        try:
                            category, idx, page_content, missing_fields, validation_failed, num_chunks = future.result()
                            ab, cat, idx_orig, mf = future_to_antibiotic[future]
                            
                            if validation_failed:
                                # Mark for removal
                                antibiotics_to_remove[category].append(idx)
                                logger.warning(f"Validation failed for {ab.get('medical_name', 'unknown')} - will remove")
                            elif page_content:
                                # Store successful scrapes with chunk count
                                scraped_alternative[(category, idx)] = (page_content, missing_fields, ab, num_chunks, False)
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
                    sorted_alternatives = sorted(
                        scraped_alternative.items(),
                        key=lambda x: x[1][3] if len(x[1]) > 3 else float('inf')  # Sort by num_chunks (4th element)
                    )
                    
                    # Step 3: Process sequentially, trying to get enough complete ones
                    # For alternative_antibiotic: if extraction fails for one, we can try others
                    # Note: processed_count continues from first_choice/second_choice processing above
                    successfully_completed = []
                    processed_indices = set()
                    alternative_completed = 0
                    
                    # Process enough to reach 5 total complete entries
                    for (category, idx), (page_content, missing_fields, antibiotic, num_chunks, _) in sorted_alternatives:
                        if alternative_completed >= needed:
                            break
                        
                        medical_name = antibiotic.get('medical_name', 'unknown')
                        try:
                            logger.info(f"Processing {medical_name} (alternative_antibiotic, {num_chunks} chunks)...")
                            
                            # Extract fields using LlamaIndex
                            extracted_fields = _extract_fields_with_llamaindex(
                                page_content=page_content,
                                medical_name=medical_name,
                                missing_fields=missing_fields,
                                existing_data=antibiotic,
                                age=age,
                                icd_code_names=icd_code_names,
                                resistance_gene=resistance_gene
                            )
                            
                            # Update antibiotic with extracted fields
                            updated = False
                            for field in missing_fields:
                                if field in extracted_fields and extracted_fields[field]:
                                    antibiotic[field] = extracted_fields[field]
                                    updated = True
                                    logger.info(f"  ✓ Updated {field} for {medical_name}")
                            
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
                                successfully_completed.append((category, idx))
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
                            ab_name = scraped_alternative[(category, idx)][2].get('medical_name', 'unknown') if (category, idx) in scraped_alternative else 'unknown'
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
        
        # Final cleanup: Remove ALL incomplete records from ALL categories
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            antibiotics = therapy_plan.get(category, [])
            if isinstance(antibiotics, list):
                complete_antibiotics = [ab for ab in antibiotics if ab.get('is_complete', False)]
                incomplete_antibiotics = [ab for ab in antibiotics if not ab.get('is_complete', False)]
                
                if incomplete_antibiotics:
                    logger.info(f"Removing {len(incomplete_antibiotics)} incomplete entries from {category} (keeping {len(complete_antibiotics)} complete)")
                    for ab in incomplete_antibiotics:
                        ab_name = ab.get('medical_name', 'unknown')
                        logger.info(f"  Removed incomplete {ab_name} from {category}")
                    
                    # Keep only complete ones
                    therapy_plan[category] = complete_antibiotics
        
        logger.info("Enrichment complete")
        
        # Save results
        input_params = state.get('input_parameters', {})
        _save_enrichment_results(input_params, result)
        
        return {'result': result}
        
    except Exception as e:
        logger.error(f"Error in enrichment_node: {e}", exc_info=True)
        raise


def _save_enrichment_results(input_params: Dict, result: Dict) -> None:
    """Save enrichment results to file."""
    try:
        from config import get_output_config
        output_config = get_output_config()
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "enrichment_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'input_parameters': input_params,
                'result': result
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enrichment results saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save enrichment results: {e}")
