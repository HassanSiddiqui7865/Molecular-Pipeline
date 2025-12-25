"""
Enrichment node for LangGraph - Enriches missing dosage information from drugs.com using Selenium.
Uses LangGraph memory store to store previous chunk context to avoid randomness.
Only processes entries where is_complete is False.
"""
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import quote_plus
import time
import random
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.store.memory import InMemoryStore
from langgraph.store.base import BaseStore

from utils import format_resistance_genes, get_icd_names_from_state

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
        
        driver.get("https://duckduckgo.com/")
        time.sleep(random.uniform(1.5, 3.0))
        
        try:
            search_box = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.ID, "searchbox_input"))
            )
            
            if not search_box:
                search_box = driver.find_element(By.CSS_SELECTOR, "input[name='q']")
            
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
    driver: Any,
    llm: BaseChatModel
) -> bool:
    """
    Validate that the drugs.com page is for the same antibiotic we're searching for.
    Uses LLM to verify medical match based on page title.
    
    Args:
        url: URL of the drugs.com page
        antibiotic_name: Name of the antibiotic we're searching for
        driver: Selenium WebDriver instance
        llm: LangChain BaseChatModel for validation
        
    Returns:
        True if the page matches the antibiotic, False otherwise
    """
    if not driver or not llm:
        return False
    
    try:
        logger.info(f"Validating antibiotic match for {antibiotic_name} at {url}...")
        
        # Navigate to page and get title
        driver.get(url)
        time.sleep(random.uniform(1.5, 2.5))
        
        # Get page title
        page_title = driver.title if driver.title else ""
        
        if not page_title:
            logger.warning(f"No page title found for {url}, skipping validation")
            return True  # Fail open if no title
        
        # Use LLM to validate match based on title only
        from pydantic import BaseModel, Field
        
        class AntibioticMatchResult(BaseModel):
            """Schema for antibiotic match validation."""
            is_match: bool = Field(..., description="True if the page title indicates this is about the same antibiotic we're searching for, False otherwise")
            reason: str = Field(..., description="Brief explanation of why it matches or doesn't match")
        
        structured_llm = llm.with_structured_output(AntibioticMatchResult)
        
        prompt = f"""Validate if drugs.com page matches the antibiotic we're searching for.

SEARCHING FOR: {antibiotic_name}
PAGE TITLE: {page_title}

TASK: Determine if page title indicates the same drug (same active ingredient/medically equivalent).

Return is_match=True if same drug, is_match=False if different drug."""
        
        result = structured_llm.invoke(prompt)
        
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
        driver.get(url)
        time.sleep(random.uniform(2.0, 3.5))
        
        try:
            content_element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "content"))
            )
            
            text = content_element.text
            
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Extracted {len(text)} characters from #content element")
            return text
            
        except Exception as e:
            logger.warning(f"Could not find #content element, trying fallback: {e}")
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
        
        if end < len(text):
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


def _get_or_create_store(state: Dict[str, Any]) -> BaseStore:
    """
    Get or create LangGraph memory store from state.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        BaseStore instance
    """
    # Ensure metadata exists
    if 'metadata' not in state:
        state['metadata'] = {}
    
    metadata = state['metadata']
    
    # Check if store already exists in metadata
    if 'enrichment_store' in metadata:
        store = metadata['enrichment_store']
        if store is not None:
            return store
    
    # Create new store if it doesn't exist
    # For production, use a persistent store; for now, use InMemoryStore
    try:
        # Simple embedding function for semantic search (optional)
        def embed(texts: list[str]) -> list[list[float]]:
            # Return dummy embeddings for now (can be enhanced with actual embeddings)
            return [[0.0] * 128 for _ in texts]
        
        store = InMemoryStore(index={"embed": embed, "dims": 128})
        metadata['enrichment_store'] = store
        logger.info("Created new enrichment memory store")
        return store
    except Exception as e:
        logger.warning(f"Error creating store with index, using basic store: {e}")
        store = InMemoryStore()
        metadata['enrichment_store'] = store
        return store


def _store_chunk_context(
    store: BaseStore,
    medical_name: str,
    chunk_index: int,
    chunk_content: str,
    extracted_fields: Dict[str, Any]
):
    """
    Store chunk context in memory store.
    
    Args:
        store: LangGraph store instance
        medical_name: Name of the antibiotic
        chunk_index: Index of the chunk
        chunk_content: Content of the chunk
        extracted_fields: Fields extracted from this chunk
    """
    try:
        namespace = ("enrichment", "chunks", medical_name)
        key = f"chunk_{chunk_index}"
        
        value = {
            "chunk_content": chunk_content[:1000],  # Store first 1000 chars for context
            "extracted_fields": extracted_fields,
            "chunk_index": chunk_index
        }
        
        store.put(namespace, key, value)
        logger.debug(f"Stored chunk context for {medical_name}, chunk {chunk_index}")
    except Exception as e:
        logger.warning(f"Error storing chunk context: {e}")


def _get_previous_chunk_contexts(
    store: BaseStore,
    medical_name: str
) -> List[Dict[str, Any]]:
    """
    Retrieve previous chunk contexts from memory store.
    
    Args:
        store: LangGraph store instance
        medical_name: Name of the antibiotic
        
    Returns:
        List of previous chunk contexts
    """
    try:
        namespace = ("enrichment", "chunks", medical_name)
        items = store.search(namespace)
        
        contexts = []
        for item in items:
            if item.value:
                contexts.append(item.value)
        
        # Sort by chunk_index
        contexts.sort(key=lambda x: x.get("chunk_index", 0))
        return contexts
    except Exception as e:
        logger.warning(f"Error retrieving chunk contexts: {e}")
        return []


def _extract_fields_with_langchain_memory(
    page_content: str,
    medical_name: str,
    missing_fields: List[str],
    existing_data: Dict[str, Any],
    age: Optional[int],
    llm: BaseChatModel,
    store: BaseStore,
    icd_code_names: Optional[str] = None,
    resistance_gene: Optional[str] = None
) -> Dict[str, Optional[str]]:
    """
    Use LangChain structured output to extract fields, using memory to store/retrieve chunk contexts.
    Blends extracted fields with existing data to avoid randomness.
    
    Args:
        page_content: Extracted content from drugs.com page
        medical_name: Name of the antibiotic
        missing_fields: List of field names that are missing
        existing_data: Existing data for this antibiotic (to blend with)
        age: Patient age (optional)
        llm: LangChain BaseChatModel
        store: LangGraph store for memory
        icd_code_names: Transformed ICD code names (comma-separated, optional)
        resistance_gene: Resistance gene name (optional)
        
    Returns:
        Dictionary with extracted field values (blended with existing data)
    """
    try:
        from pydantic import BaseModel, Field
        
        class DosageExtractionResult(BaseModel):
            """Schema for extracted dosage information."""
            dose_duration: Optional[str] = Field(None, description="Dosing information in natural text format including ALL dosages (loading and maintenance) in concise way. MUST match the specific ICD code conditions and consider resistance gene and patient age. Examples: '600 mg IV q12h for 14 days', 'Loading: 1g IV, then 500 mg q12h for 7-14 days', '450 mg q24h on Days 1 and 2, then 300 mg q24h for 7-14 days', 'Trimethoprim 160 mg plus Sulfamethoxazole 800 mg PO q12h for 7 days'. EXCLUDE monitoring details, target levels, infusion rates, administration notes (place in general_considerations). Include loading doses if present - keep concise and natural. DO NOT create duplicates - choose ONE most appropriate dosage. Use existing value if already present, otherwise extract from content.")
            route_of_administration: Optional[str] = Field(None, description="Route of administration. Must be one of: 'IV', 'PO', 'IM', 'IV/PO'. Examples: 'IV', 'PO', 'IV/PO'. Use existing value if already present, otherwise extract from content.")
            general_considerations: Optional[str] = Field(None, description="Clinical notes and considerations. If dose_duration contains multiple dosages (separated by |), mention which condition from ICD codes each dosage is for. Examples: 'Monitor renal function, risk of nephrotoxicity' or 'For bacteremia: 600 mg IV q12h. For pneumonia: 500 mg PO q8h. Monitor renal function.'. Use existing value if already present, otherwise extract from content.")
            coverage_for: Optional[str] = Field(None, description="Conditions or infections this antibiotic covers using clinical terminology only (e.g., 'MRSA bacteremia', 'VRE bacteremia', 'Staphylococcus aureus bacteremia'). Do NOT include ICD codes (e.g., A41.2) or ICD code names (e.g., 'Sepsis due to...'). Use clinical terms like 'bacteremia', 'sepsis', 'endocarditis'. Use existing value if already present, otherwise extract from content.")
            renal_adjustment: Optional[str] = Field(None, description="Renal adjustment or dosing guidelines for patients with renal impairment. Be concise and factual - NO repetition, NO references, NO citations. Extract only essential adjustment information. Examples: 'Adjust dose in CrCl < 30 mL/min' or 'No adjustment needed' or 'Reduce dose by 50% in CrCl < 30 mL/min'. Use existing value if already present, otherwise extract from content.")
        
        patient_age_str = f"{age} years" if age else "adult"
        missing_fields_str = ", ".join(missing_fields)
        icd_code_names_str = icd_code_names if icd_code_names else "none"
        resistance_gene_str = resistance_gene if resistance_gene else "none"
        
        # Get previous chunk contexts from memory
        previous_contexts = _get_previous_chunk_contexts(store, medical_name)
        
        # Build context summary from previous chunks
        previous_context_summary = ""
        if previous_contexts:
            previous_context_summary = "\n\nPREVIOUS CHUNK CONTEXTS (for consistency):\n"
            for ctx in previous_contexts[-3:]:  # Last 3 chunks
                prev_fields = ctx.get("extracted_fields", {})
                if prev_fields:
                    prev_summary = ", ".join([f"{k}={v}" for k, v in prev_fields.items() if v])
                    if prev_summary:
                        previous_context_summary += f"- {prev_summary}\n"
        
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
                existing_data_context = f"\n\nEXISTING DATA (preserve and blend with):\n" + "\n".join(existing_fields)
        
        # Chunk the content if it's too large
        chunk_size = 6000
        chunks = _chunk_text(page_content, chunk_size=chunk_size, overlap=500)
        
        logger.info(f"[LangChain+Memory] Processing {len(chunks)} chunks for {medical_name} (total length: {len(page_content)} chars)")
        
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
                logger.debug(f"[LangChain+Memory] Processing chunk {i+1}/{len(chunks)} for {medical_name}")
                
                prompt = f"""Extract ONLY missing fields for {medical_name} from drugs.com content.
Be ACCURATE - extract dosages matching patient conditions.

PATIENT: Name={medical_name} | Age={patient_age_str} | ICD={icd_code_names_str} | Gene={resistance_gene_str}

MISSING FIELDS (extract ONLY these): {missing_fields_str}

EXISTING DATA (context only, do NOT extract):
{existing_data_context}

PREVIOUS CHUNK CONTEXTS:
{previous_context_summary}

PAGE CONTENT (chunk {i+1} of {len(chunks)}):
{chunk}

DOSAGE EXTRACTION RULES:
1. Extract dosages SPECIFICALLY appropriate for ICD: {icd_code_names_str}
2. Consider Resistance Gene: {resistance_gene_str}
3. Consider Patient Age: {patient_age_str} (pediatric vs adult)
4. DO NOT extract duplicates - choose ONE most appropriate
5. Include loading doses if present - keep concise and natural (e.g., "Loading: 1g IV, then 500 mg q12h for 7-14 days")
6. Extract ONLY dosage(s) relevant to ICD: {icd_code_names_str}
7. Frequency MUST include "q" prefix: q8h, q12h, q24h (NEVER just 8h, 12h, 24h)
8. Be precise - no variations or duplicates

CONSISTENCY:
- Use existing fields as context
- Extracted fields must be medically consistent
- All fields must work together logically

INSTRUCTIONS:
1. Extract ONLY fields in missing_fields: {missing_fields_str}
2. Return existing values for fields NOT in missing_fields
3. Match dose_duration to ICD: {icd_code_names_str}, Gene: {resistance_gene_str}, Age: {patient_age_str}
4. Maintain consistency with previous chunk contexts

FIELD DESCRIPTIONS (extract ONLY if in missing_fields):
- dose_duration: Natural text format including ALL dosages (loading and maintenance) in concise way (e.g., "600 mg IV q12h for 14 days", "Loading: 1g IV, then 500 mg q12h for 7-14 days", "450 mg q24h on Days 1 and 2, then 300 mg q24h for 7-14 days")
  * Match ICD: {icd_code_names_str}, Gene: {resistance_gene_str}, Age: {patient_age_str}
  * Frequency MUST include "q" prefix (q8h, q12h, q24h)
  * Include loading doses if present - keep concise and natural (e.g., "Loading: 1g IV, then 500 mg q12h for 7-14 days")
  * NO duplicates - choose ONE most appropriate
- route_of_administration: 'IV', 'PO', 'IM', 'IV/PO'
- general_considerations: Clinical notes. If multiple dosages, mention which ICD condition each is for
- coverage_for: Conditions matching patient's clinical condition using clinical terminology only (e.g., "MRSA bacteremia", "VRE bacteremia")
- renal_adjustment: Concise adjustment info (NO repetition/references/citations)
  Examples: "Adjust dose in CrCl < 30 mL/min" or "No adjustment needed"

CRITICAL:
- Extract ONLY missing fields - return existing for others
- Use ICD, Gene, Age for ACCURATE dosages
- NO duplicates or conflicting dosages
- Include loading doses if present - keep concise and natural
- Accuracy > completeness"""
                
                result = structured_llm.invoke(prompt)
                
                # Store chunk context in memory
                extracted_from_chunk = {}
                if result.dose_duration:
                    extracted_from_chunk['dose_duration'] = result.dose_duration
                if result.route_of_administration:
                    extracted_from_chunk['route_of_administration'] = result.route_of_administration
                if result.general_considerations:
                    extracted_from_chunk['general_considerations'] = result.general_considerations
                if result.coverage_for:
                    extracted_from_chunk['coverage_for'] = result.coverage_for
                if result.renal_adjustment:
                    extracted_from_chunk['renal_adjustment'] = result.renal_adjustment
                
                _store_chunk_context(store, medical_name, i, chunk, extracted_from_chunk)
                
                # Collect non-null results (only for missing fields)
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
                    
            except Exception as e:
                logger.warning(f"[LangChain+Memory] Error processing chunk {i+1} for {medical_name}: {e}")
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
        
        logger.info(f"[LangChain+Memory] Extracted fields for {medical_name}: {[k for k, v in extracted.items() if v and k in missing_fields]}")
        return extracted
        
    except Exception as e:
        logger.error(f"Error extracting fields with LangChain+Memory for {medical_name}: {e}")
        return {}


def _scrape_antibiotic_page(
    antibiotic: Dict[str, Any],
    category: str,
    idx: int,
    llm: Optional[BaseChatModel] = None
) -> Tuple[str, int, Optional[str], List[str], bool, int]:
    """
    Scrape page content for a single antibiotic in a separate thread with its own browser.
    
    Args:
        antibiotic: Antibiotic dictionary
        category: Category name (first_choice, second_choice, alternative_antibiotic)
        idx: Index of the antibiotic in the list
        llm: Optional LLM for validation
        
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
            if llm:
                is_valid = _validate_antibiotic_match(drugs_com_url, medical_name, driver, llm)
                if not is_valid:
                    logger.warning(f"[Thread] Page validation failed for {medical_name} - drug name doesn't match, will remove from result")
                    return (category, idx, None, missing_fields, True, 0)  # validation_failed=True
            
            logger.info(f"[Thread] Navigating directly to {drugs_com_url}")
            page_content = _scrape_drugs_com_page(drugs_com_url, driver)
            
            if page_content:
                # Calculate number of chunks
                chunks = _chunk_text(page_content, chunk_size=6000, overlap=500)
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
        
        # Get or create memory store
        store = _get_or_create_store(state)
        
        # Get LLM
        from config import get_ollama_llm
        llm = get_ollama_llm()
        
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
        resistance_gene = format_resistance_genes(resistant_genes) if resistant_genes else None
        
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
        
        # Process first_choice and second_choice sequentially
        # For first_choice and second_choice: if extraction fails (no required fields), drop the antibiotic
        for antibiotic, category, idx, missing_fields in first_choice_ab + second_choice_ab:
            medical_name = antibiotic.get('medical_name', 'unknown')
            try:
                logger.info(f"Processing {medical_name}...")
                
                # Step 1: Scrape page content (with validation)
                logger.info(f"  [1/2] Scraping {medical_name} from drugs.com...")
                category_result, idx_result, page_content, scraped_missing_fields, validation_failed, num_chunks = _scrape_antibiotic_page(antibiotic, category, idx, llm)
                
                # If validation failed (name doesn't match), mark for removal
                if validation_failed:
                    logger.warning(f"  Validation failed for {medical_name} - removing from result")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                if not page_content:
                    logger.warning(f"  Could not scrape {medical_name}, removing from result (first_choice/second_choice require extraction)")
                    antibiotics_to_remove[category].append(idx)
                    continue
                
                # Step 2: Extract fields using LangChain with memory
                logger.info(f"  [2/2] Extracting fields for {medical_name} with memory context...")
                extracted_fields = _extract_fields_with_langchain_memory(
                    page_content=page_content,
                    medical_name=medical_name,
                    missing_fields=missing_fields,
                    existing_data=antibiotic,  # Pass existing data to blend with
                    age=age,
                    llm=llm,
                    store=store,
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
                    continue
                
                if updated:
                    logger.info(f"  ✓ Completed enrichment for {medical_name} (is_complete={antibiotic['is_complete']})")
                else:
                    logger.warning(f"  No fields updated for {medical_name}")
                
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
                        executor.submit(_scrape_antibiotic_page, ab, cat, idx, llm): (ab, cat, idx, mf)
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
                    processed_count = 0
                    successfully_completed = []
                    processed_indices = set()
                    
                    # Process enough to reach 5 total complete entries
                    for (category, idx), (page_content, missing_fields, antibiotic, num_chunks, _) in sorted_alternatives:
                        if processed_count >= needed:
                            break
                        
                        medical_name = antibiotic.get('medical_name', 'unknown')
                        try:
                            logger.info(f"Processing {medical_name} (alternative_antibiotic, {num_chunks} chunks)...")
                            
                            # Extract fields using LangChain with memory
                            extracted_fields = _extract_fields_with_langchain_memory(
                                page_content=page_content,
                                medical_name=medical_name,
                                missing_fields=missing_fields,
                                existing_data=antibiotic,
                                age=age,
                                llm=llm,
                                store=store,
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
                                processed_count += 1
                                successfully_completed.append((category, idx))
                                logger.info(f"  ✓ Completed enrichment for {medical_name} (is_complete=True, {processed_count}/{needed} needed)")
                            elif updated:
                                # Got some fields but not complete - keep it for alternative_antibiotic
                                logger.info(f"  ✓ Updated fields for {medical_name} but not yet complete (is_complete={antibiotic['is_complete']})")
                            else:
                                # Extraction failed completely - for alternative_antibiotic, we can try others, but remove this one
                                logger.warning(f"  No fields extracted for {medical_name}, will try others if needed")
                                antibiotics_to_remove[category].append(idx)
                            
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
        
        # Final cleanup: Remove incomplete alternative_antibiotic if we already have 5+ complete ones
        alternative_antibiotics = therapy_plan.get('alternative_antibiotic', [])
        if isinstance(alternative_antibiotics, list):
            complete_alternatives = [ab for ab in alternative_antibiotics if ab.get('is_complete', False)]
            incomplete_alternatives = [ab for ab in alternative_antibiotics if not ab.get('is_complete', False)]
            
            if len(complete_alternatives) >= 5:
                logger.info(f"Found {len(complete_alternatives)} complete alternative_antibiotic entries (>= 5), removing {len(incomplete_alternatives)} incomplete ones")
                # Keep only complete ones
                therapy_plan['alternative_antibiotic'] = complete_alternatives
                for ab in incomplete_alternatives:
                    ab_name = ab.get('medical_name', 'unknown')
                    logger.info(f"Removed incomplete {ab_name} from alternative_antibiotic (already have 5+ complete)")
            else:
                logger.info(f"Found {len(complete_alternatives)} complete alternative_antibiotic entries (< 5), keeping {len(incomplete_alternatives)} incomplete ones")
        
        logger.info("Enrichment complete")
        return {'result': result}
        
    except Exception as e:
        logger.error(f"Error in enrichment_node: {e}", exc_info=True)
        raise
