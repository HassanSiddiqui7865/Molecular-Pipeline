"""
ICD Transform node for LangGraph - Transforms ICD codes to human-readable names.
First tries database lookup via SSH, then falls back to scraping icd10data.com.
"""
import logging
import time
import socket
import threading
from typing import Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Try to import SSH library
try:
    import paramiko
    PARAMIKO_AVAILABLE = True
except ImportError:
    PARAMIKO_AVAILABLE = False
    logger.warning("paramiko not available. Install with: pip install paramiko")

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


def _handler(src, dst):
    """Forward data from src to dst."""
    try:
        while True:
            data = src.recv(1024)
            if not data:
                break
            dst.send(data)
    except Exception:
        pass
    finally:
        try:
            src.close()
            dst.close()
        except Exception:
            pass


def _forward_tunnel(local_port, remote_host, remote_port, transport):
    """Forward connections from local_port to remote_host:remote_port via transport."""
    class ForwardServer:
        def __init__(self, remote_host, remote_port, transport):
            self.server = None
            self.remote_host = remote_host
            self.remote_port = remote_port
            self.transport = transport
            self.running = False
            
        def handle(self, client, addr):
            try:
                chan = self.transport.open_channel('direct-tcpip', (self.remote_host, self.remote_port), addr)
            except Exception as e:
                logger.warning(f"Error opening channel: {e}")
                try:
                    client.close()
                except Exception:
                    pass
                return
            
            if chan is None:
                logger.warning(f"Channel not opened for {self.remote_host}:{self.remote_port}")
                try:
                    client.close()
                except Exception:
                    pass
                return
            
            # Start forwarding in both directions
            threading.Thread(target=_handler, args=(client, chan), daemon=True).start()
            threading.Thread(target=_handler, args=(chan, client), daemon=True).start()
        
        def start(self, local_port):
            self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.bind(('127.0.0.1', local_port))
            self.server.listen(100)
            self.running = True
            
            while self.running:
                try:
                    client, addr = self.server.accept()
                    threading.Thread(target=self.handle, args=(client, addr), daemon=True).start()
                except Exception:
                    break
        
        def stop(self):
            self.running = False
            if self.server:
                try:
                    self.server.close()
                except Exception:
                    pass
    
    return ForwardServer(remote_host, remote_port, transport)


@contextmanager
def _get_ssh_tunnel(db_config: Dict[str, Any]):
    """
    Create and manage SSH tunnel to database server using paramiko.
    
    Args:
        db_config: Database configuration dictionary
        
    Yields:
        Local port number for the tunnel
    """
    if not PARAMIKO_AVAILABLE:
        raise ImportError("paramiko not available")
    
    ssh_client = None
    forward_server = None
    local_port = None
    
    try:
        # Create SSH client
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        # Load SSH key if provided
        pkey = None
        if db_config.get('ssh_key_path'):
            from pathlib import Path
            ssh_key_path_str = db_config['ssh_key_path']
            ssh_key_path = Path(ssh_key_path_str)
            ssh_key_passphrase = db_config.get('ssh_key_passphrase', '')
            
            # If no explicit passphrase is set, use SSH password as passphrase
            if not ssh_key_passphrase and db_config.get('ssh_password'):
                ssh_key_passphrase = db_config.get('ssh_password')
            
            # If path doesn't exist, try looking for "key" in project root
            if not ssh_key_path.exists():
                project_root = Path(__file__).parent.parent.parent
                potential_key = project_root / 'key'
                if potential_key.exists() and potential_key.is_file():
                    ssh_key_path = potential_key
            
            # Try different key types
            if ssh_key_path.exists() and ssh_key_path.is_file():
                key_errors = []
                for key_class in [paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey]:
                    try:
                        try:
                            pkey = key_class.from_private_key_file(str(ssh_key_path))
                            break
                        except paramiko.ssh_exception.PasswordRequiredException:
                            if ssh_key_passphrase:
                                pkey = key_class.from_private_key_file(str(ssh_key_path), password=ssh_key_passphrase)
                                break
                            else:
                                raise
                    except paramiko.ssh_exception.PasswordRequiredException:
                        key_errors.append(f"{key_class.__name__}: requires passphrase")
                    except Exception as e:
                        key_errors.append(f"{key_class.__name__}: {e}")
                
                if not pkey and not db_config.get('ssh_password'):
                    raise ValueError(f"Could not load SSH key. Tried: {', '.join(key_errors)}")
        
        # Connect to SSH server
        logger.info(f"Connecting to SSH server {db_config['ssh_host']}:{db_config['ssh_port']}...")
        ssh_client.connect(
            hostname=db_config['ssh_host'],
            port=db_config['ssh_port'],
            username=db_config['ssh_username'],
            pkey=pkey,
            password=db_config.get('ssh_password'),
            allow_agent=False,
            look_for_keys=False
        )
        
        # Get transport for port forwarding
        transport = ssh_client.get_transport()
        
        # Find an available local port
        temp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        temp_socket.bind(('127.0.0.1', 0))
        local_port = temp_socket.getsockname()[1]
        temp_socket.close()
        
        # Start port forwarding
        forward_server = _forward_tunnel(
            local_port,
            db_config['db_host'],
            db_config['db_port'],
            transport
        )
        
        # Start forwarding server in a thread
        forward_thread = threading.Thread(target=forward_server.start, args=(local_port,))
        forward_thread.daemon = True
        forward_thread.start()
        
        logger.info(f"SSH tunnel established. Local port: {local_port}")
        
        yield local_port
        
    finally:
        # Clean up
        if forward_server:
            try:
                forward_server.stop()
            except Exception as e:
                logger.warning(f"Error stopping forward server: {e}")
        
        if ssh_client:
            try:
                ssh_client.close()
                logger.info("SSH tunnel closed")
            except Exception as e:
                logger.warning(f"Error closing SSH connection: {e}")


def _get_icd_code_from_database(code: str, db_config: Dict[str, Any]) -> Optional[str]:
    """
    Get ICD code name from database via SSH tunnel using two-step query:
    1. Query dev.disease_identifiers to find disease_id by ICD10 code
    2. Query diseases table to get disease name by disease_id
    
    Args:
        code: Normalized ICD code
        db_config: Database configuration dictionary
        
    Returns:
        ICD code name/description, or None if not found or error
    """
    if not PARAMIKO_AVAILABLE:
        logger.warning("paramiko not available, cannot query database")
        return None
    
    try:
        with _get_ssh_tunnel(db_config) as local_port:
            # Connect to PostgreSQL database through SSH tunnel
            if not POSTGRESQL_AVAILABLE:
                logger.warning("psycopg2 not available, cannot query PostgreSQL database")
                return None
            
            connection = psycopg2.connect(
                host='127.0.0.1',
                port=local_port,
                user=db_config['db_username'],
                password=db_config['db_password'],
                database=db_config['db_name']
            )
            
            try:
                # Create cursor with RealDictCursor for dictionary-like results
                cursor = connection.cursor(cursor_factory=RealDictCursor)
                
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
                finally:
                    cursor.close()
                        
            finally:
                connection.close()
                
    except Exception as e:
        logger.warning(f"Error querying database for ICD code {code}: {e}")
        return None


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


def _get_icd_code_name(code: str, db_config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get ICD code name, first trying database, then falling back to scraping.
    
    Args:
        code: Normalized ICD code
        db_config: Database configuration dictionary (optional)
        
    Returns:
        ICD code name/description, or original code if not found
    """
    # Try database first if config is provided
    if db_config and db_config.get('ssh_host'):
        try:
            name = _get_icd_code_from_database(code, db_config)
            if name:
                return name
            logger.info(f"ICD code {code} not found in database, falling back to scraping")
        except Exception as e:
            logger.warning(f"Database lookup failed for {code}: {e}, falling back to scraping")
    
    # Fallback to scraping
    if SELENIUM_AVAILABLE:
        return _get_icd_code_name_scraping(code)
    else:
        logger.warning("Selenium not available, cannot scrape. Returning original code.")
        return code


def _transform_icd_codes(severity_codes: str, db_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Transform ICD codes to human-readable names.
    First tries database lookup via SSH, then falls back to scraping icd10data.com.
    
    Args:
        severity_codes: Comma-separated string of ICD codes (e.g., "A41.9, B95.3")
        db_config: Database configuration dictionary (optional)
        
    Returns:
        Dictionary with original codes, transformed names, and combined string
    """
    if not severity_codes or not severity_codes.strip():
        return {
            'original_codes': [],
            'code_names': [],
            'severity_codes_transformed': ''
        }
    
    # Split by comma and clean up
    codes = [code.strip() for code in severity_codes.split(',')]
    codes = [code for code in codes if code]  # Remove empty strings
    
    # Normalize codes
    normalized_codes = [_normalize_icd_code(code) for code in codes]
    
    # Transform to names (database first, then scraping)
    code_names = []
    for code in normalized_codes:
        try:
            logger.info(f"Fetching ICD description for {code}...")
            name = _get_icd_code_name(code, db_config)
            code_names.append({
                'code': code,
                'name': name
            })
            # Add small delay only if we used scraping (database is fast)
            if not db_config or not db_config.get('ssh_host') or name == code:
                time.sleep(1)  # Small delay to avoid rate limiting
        except Exception as e:
            # If individual code fails, use original code as name
            logger.warning(f"Failed to get description for {code}: {e}. Using original code.")
            code_names.append({
                'code': code,
                'name': code  # Fallback to original code
            })
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
        input_params = state.get('input_parameters', {})
        severity_codes = input_params.get('severity_codes', '')
        
        if not severity_codes:
            logger.warning("No severity codes found in input parameters")
            return {
                'icd_transformation': {
                    'original_codes': [],
                    'code_names': [],
                    'severity_codes_transformed': ''
                }
            }
        
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
        
        return {
            'icd_transformation': transformation_result
        }
        
    except Exception as e:
        logger.error(f"Error in icd_transform_node: {e}", exc_info=True)
        # Don't halt pipeline - return original codes if transformation fails
        input_params = state.get('input_parameters', {})
        severity_codes = input_params.get('severity_codes', '')
        if severity_codes:
            # Return original codes as fallback
            codes = [code.strip().upper() for code in severity_codes.split(',') if code.strip()]
            return {
                'icd_transformation': {
                    'original_codes': codes,
                    'code_names': [{'code': code, 'name': code} for code in codes],
                    'severity_codes_transformed': severity_codes
                }
            }
        return {
            'icd_transformation': {
                'original_codes': [],
                'code_names': [],
                'severity_codes_transformed': ''
            }
        }