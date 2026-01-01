"""
ICD Transform node for LangGraph - Transforms ICD codes to human-readable names.
First tries database lookup via SSH, then falls back to scraping icd10data.com.
"""
import logging
import time
from typing import Dict, Any, Optional, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Import SSH tunnel functions from db_session
try:
    from db_session import get_ssh_tunnel, PARAMIKO_AVAILABLE
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("db_session not available, SSH tunnel disabled")
    def get_ssh_tunnel(db_config):
        raise ImportError("SSH tunnel not available")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    logger.warning("psycopg2 not available. Install with: pip install psycopg2-binary")

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    logger.warning("Selenium not available. Install with: pip install selenium")


def _normalize_icd_code(code: str) -> str:
    """
    Normalize ICD code by removing whitespace and converting to uppercase.
    
    Args:
        code: ICD code string
        
    Returns:
        Normalized ICD code
    """
    return code.strip().upper()


def _query_icd_code_from_db(code: str, cursor) -> Optional[str]:
    """
    Query a single ICD code from database using an existing cursor.
    
    Args:
        code: Normalized ICD code
        cursor: Database cursor (RealDictCursor)
        
    Returns:
        ICD code name/description, or None if not found or error
    """
    try:
        # Step 1: Query disease_identifiers to find disease_id
        # First try exact match, then try pattern match if needed
        code_lower = code.lower()
        
        # Try exact match first
        query1 = """
            SELECT disease_id 
            FROM dev.disease_identifiers 
            WHERE identifier_type = 'ICD10' 
            AND LOWER(identifier_value) = %s 
            LIMIT 1
        """
        cursor.execute(query1, (code_lower,))
        result1 = cursor.fetchone()
        
        # If not found, try pattern match
        if not result1:
            query1_pattern = """
                SELECT disease_id 
                FROM dev.disease_identifiers 
                WHERE identifier_type = 'ICD10' 
                AND LOWER(identifier_value) LIKE %s 
                LIMIT 1
            """
            pattern = f"{code_lower}%"
            cursor.execute(query1_pattern, (pattern,))
            result1 = cursor.fetchone()
        
        if not result1:
            logger.debug(f"ICD code {code} not found in disease_identifiers table")
            return None
        
        # Extract disease_id
        if isinstance(result1, dict):
            disease_id = result1.get('disease_id')
        else:
            disease_id = result1[0] if result1 else None
        
        if not disease_id:
            logger.debug(f"No disease_id found for ICD code {code}")
            return None
        
        # Step 2: Query dev.diseases table to get disease name
        query2 = 'SELECT * FROM dev.diseases WHERE id = %s LIMIT 1'
        
        cursor.execute(query2, (disease_id,))
        result2 = cursor.fetchone()
        
        if not result2:
            logger.debug(f"Disease with id {disease_id} not found in diseases table")
            return None
        
        # Extract name from result
        # Try common column names: 'name', 'disease_name', 'title'
        if isinstance(result2, dict):
            name = result2.get('name') or result2.get('disease_name') or result2.get('title')
        else:
            # For tuple results, try to find name column by index
            # This is less reliable, so we'll try to get column names
            if hasattr(cursor, 'description') and cursor.description:
                column_names = [desc[0] for desc in cursor.description]
                try:
                    name_idx = column_names.index('name')
                    name = result2[name_idx] if len(result2) > name_idx else None
                except ValueError:
                    # Try other common names
                    for col_name in ['disease_name', 'title']:
                        try:
                            name_idx = column_names.index(col_name)
                            name = result2[name_idx] if len(result2) > name_idx else None
                            if name:
                                break
                        except ValueError:
                            continue
                    else:
                        name = None
            else:
                name = None
        
        if name:
            logger.info(f"Found ICD code {code} in database: {name}")
            return name
        else:
            logger.debug(f"Disease found but no name column available for disease_id {disease_id}")
            return None
    except Exception as e:
        logger.warning(f"Error querying database for ICD code {code}: {e}")
        return None


@contextmanager
def _get_db_connection_with_tunnel(db_config: Dict[str, Any]):
    """
    Context manager that opens SSH tunnel and database connection.
    Yields (connection, cursor) tuple.
    
    Args:
        db_config: Database configuration dictionary
        
    Yields:
        (connection, cursor) tuple
    """
    if not PARAMIKO_AVAILABLE:
        raise ImportError("paramiko not available")
    
    if not POSTGRESQL_AVAILABLE:
        raise ImportError("psycopg2 not available")
    
    connection = None
    cursor = None
    
    # Use nested context manager for SSH tunnel (from db_session)
    with get_ssh_tunnel(db_config) as local_port:
        try:
            # Connect to PostgreSQL database through SSH tunnel
            connection = psycopg2.connect(
                host='127.0.0.1',
                port=local_port,
                user=db_config['db_username'],
                password=db_config['db_password'],
                database=db_config['db_name']
            )
            
            # Create cursor with RealDictCursor for dictionary-like results
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            yield (connection, cursor)
            
        finally:
            # Clean up cursor
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            
            # Clean up connection
            if connection:
                try:
                    connection.close()
                except Exception:
                    pass


def _get_icd_code_name_scraping(code: str) -> str:
    """
    Get ICD code name by searching directly on icd10data.com using Selenium.
    
    Args:
        code: Normalized ICD code
        
    Returns:
        ICD code name/description, or original code if not found
    """
    driver = None
    try:
        # Initialize Chrome driver
        options = webdriver.ChromeOptions()
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=options)
        driver.maximize_window()
        
        # Navigate to icd10data.com
        logger.info(f"Opening icd10data.com for ICD code {code}...")
        driver.get("https://www.icd10data.com/")
        
        # Wait only for search box to be available (don't wait for entire page)
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "searchText"))
        )
        
        # Enter the code and search immediately
        search_box.clear()
        search_box.send_keys(code)
        
        # Click the search button
        search_button = driver.find_element(By.ID, "search")
        search_button.click()
        
        # Wait for search results to appear
        logger.info("Waiting for search results...")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.searchLine"))
        )
        
        # Find the first result that contains the exact code and extract description
        description = None
        
        try:
            # Find all search result lines
            search_lines = driver.find_elements(By.CSS_SELECTOR, "div.searchLine")
            
            logger.info(f"Found {len(search_lines)} search results")
            
            # Look for the result that matches the exact code
            for search_line in search_lines:
                try:
                    # Check if this search line contains the exact code in identifier span
                    identifier_spans = search_line.find_elements(By.CSS_SELECTOR, "span.identifier")
                    code_found = False
                    
                    for span in identifier_spans:
                        span_text = span.text.strip()
                        # Check for exact match (code should match exactly, not just be contained)
                        if span_text == code or span_text.startswith(code + ' ') or span_text.startswith(code + '-'):
                            code_found = True
                            break
                    
                    if code_found:
                        # Extract description from searchPadded div
                        try:
                            search_padded = search_line.find_element(By.CSS_SELECTOR, "div.searchPadded")
                            # Get the first div inside searchPadded which contains the description
                            description_div = search_padded.find_element(By.CSS_SELECTOR, "div")
                            description = description_div.text.strip()
                            logger.info(f"Found matching result for {code}: {description}")
                            break
                        except:
                            continue
                except:
                    continue
        except Exception as e:
            logger.warning(f"Error finding search results: {e}")
        
        if not description:
            logger.warning(f"No matching result found for code {code}")
            return code
        
        return description
        
    except Exception as e:
        logger.error(f"Error getting ICD code name for {code}: {e}", exc_info=True)
        return code
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def _transform_icd_codes(severity_codes: List[str], db_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Transform ICD codes to human-readable names.
    First tries database lookup via SSH, then falls back to scraping icd10data.com.
    Opens database connection once and reuses it for all codes.
    
    Args:
        severity_codes: List of ICD codes (e.g., ["A41.9", "B95.3"])
        db_config: Database configuration dictionary (optional)
        
    Returns:
        Dictionary with original codes, transformed names, and combined string
    """
    if not isinstance(severity_codes, list):
        return {
            'original_codes': [],
            'code_names': [],
            'severity_codes_transformed': ''
        }
    
    codes = [str(code).strip().upper() for code in severity_codes if code and str(code).strip()]
    
    if not codes:
        return {
            'original_codes': [],
            'code_names': [],
            'severity_codes_transformed': ''
        }
    
    # Normalize codes
    normalized_codes = [_normalize_icd_code(code) for code in codes]
    
    # Transform to names (database first, then scraping)
    code_names = []
    
    # Try to use database if config is available
    db_available = False
    if db_config and db_config.get('ssh_host'):
        try:
            # Open connection once and reuse for all codes
            with _get_db_connection_with_tunnel(db_config) as (connection, cursor):
                db_available = True
                logger.info(f"Opened database connection for {len(normalized_codes)} ICD codes")
                
                for code in normalized_codes:
                    try:
                        logger.info(f"Fetching ICD description for {code} from database...")
                        name = _query_icd_code_from_db(code, cursor)
                        
                        # If not found in database, will fall back to scraping
                        if name:
                            code_names.append({
                                'code': code,
                                'name': name
                            })
                        else:
                            # Mark for scraping fallback
                            code_names.append({
                                'code': code,
                                'name': None  # Will be filled by scraping
                            })
                    except Exception as e:
                        logger.warning(f"Error querying database for {code}: {e}. Will try scraping.")
                        code_names.append({
                            'code': code,
                            'name': None  # Will be filled by scraping
                        })
        except Exception as e:
            logger.warning(f"Failed to open database connection: {e}. Falling back to scraping for all codes.")
            db_available = False
    
    # If database wasn't used or some codes weren't found, use scraping
    if not db_available:
        # Initialize all codes for scraping
        code_names = [{'code': code, 'name': None} for code in normalized_codes]
    
    # Fill in missing names using scraping
    for item in code_names:
        if item['name'] is None:
            code = item['code']
            try:
                logger.info(f"Fetching ICD description for {code} via scraping...")
                name = _get_icd_code_name_scraping(code)
                item['name'] = name
                # Add small delay to avoid rate limiting
                time.sleep(1)
            except Exception as e:
                # If scraping fails, use original code as name
                logger.warning(f"Failed to get description for {code}: {e}. Using original code.")
                item['name'] = code
                time.sleep(0.5)  # Shorter delay on error
    
    # Create combined string
    # If name equals code (not found), just show the code; otherwise show "code (name)"
    combined_parts = []
    for item in code_names:
        if item['name'] == item['code']:
            combined_parts.append(item['code'])
        else:
            combined_parts.append(f"{item['code']} ({item['name']})")
    combined = ', '.join(combined_parts)
    
    return {
        'original_codes': normalized_codes,
        'code_names': code_names,
        'severity_codes_transformed': combined
    }


def icd_transform_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that transforms ICD codes to human-readable names.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with transformed ICD codes
    """
    try:
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        
        # Emit progress for ICD transform start
        if progress_callback:
            progress_callback('icd_transform', 0, 'Transforming ICD codes...')
        
        input_params = state.get('input_parameters', {})
        severity_codes = input_params.get('severity_codes', [])
        
        if not severity_codes or not isinstance(severity_codes, list) or len(severity_codes) == 0:
            logger.warning("No severity codes found in input parameters")
            if progress_callback:
                progress_callback('icd_transform', 100, 'No ICD codes to transform')
            return {
                'icd_transformation': {
                    'original_codes': [],
                    'code_names': [],
                    'severity_codes_transformed': ''
                }
            }
        
        # Emit progress for database lookup
        if progress_callback:
            progress_callback('icd_transform', 30, f'Looking up {len(severity_codes)} ICD codes...')
        
        # Get database config if available
        from config import get_database_config
        db_config = None
        try:
            db_config = get_database_config()
            # Only use if SSH host is configured
            if not db_config.get('ssh_host'):
                db_config = None
        except Exception as e:
            logger.debug(f"Database config not available: {e}")
        
        # Transform ICD codes
        transformation_result = _transform_icd_codes(severity_codes, db_config)
        
        logger.info(f"Transformed {len(transformation_result['original_codes'])} ICD codes: {transformation_result['severity_codes_transformed']}")
        
        # Emit progress for ICD transform complete
        if progress_callback:
            progress_callback('icd_transform', 100, f'Transformed {len(transformation_result["original_codes"])} ICD codes')
        
        return {
            'icd_transformation': transformation_result
        }
        
    except Exception as e:
        logger.error(f"Error in icd_transform_node: {e}", exc_info=True)
        # Don't halt pipeline - return original codes if transformation fails
        input_params = state.get('input_parameters', {})
        severity_codes = input_params.get('severity_codes', [])
        
        if severity_codes and isinstance(severity_codes, list):
            codes = [str(code).strip().upper() for code in severity_codes if code and str(code).strip()]
            if codes:
                return {
                    'icd_transformation': {
                        'original_codes': codes,
                        'code_names': [{'code': code, 'name': code} for code in codes],
                        'severity_codes_transformed': ', '.join(codes)
                    }
                }
        return {
            'icd_transformation': {
                'original_codes': [],
                'code_names': [],
                'severity_codes_transformed': ''
            }
        }