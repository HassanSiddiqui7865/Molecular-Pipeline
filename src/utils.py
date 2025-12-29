"""
Utility functions for the Molecular Pipeline.
"""
import logging
import time
from typing import Optional, List, Dict, Any, TypeVar, Callable, Type
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    from llama_index.llms.ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None
    Ollama = None

T = TypeVar('T', bound=BaseModel)


def fix_text_encoding(text: Optional[str]) -> str:
    """
    Fix encoding issues in text fields - replaces problematic Unicode characters
    that appear as squares in PDFs with standard ASCII equivalents.
    
    Args:
        text: Input text that may contain problematic Unicode characters
        
    Returns:
        Cleaned text with all problematic characters replaced
    """
    if not text:
        return ""
    
    # Comprehensive replacement of ALL problematic Unicode characters
    import re
    
    # Replace ALL Unicode dash/hyphen variants with standard ASCII hyphen
    # This catches en dash, em dash, horizontal bar, minus sign, and ALL variants
    dash_pattern = re.compile(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D\u2500-\u2501\u2E3A\u2E3B]')
    text = dash_pattern.sub('-', text)
    
    # Also replace common dash characters explicitly
    text = text.replace('–', '-')  # En dash
    text = text.replace('—', '-')  # Em dash
    text = text.replace('−', '-')  # Minus sign
    text = text.replace('‐', '-')  # Hyphen
    
    # Replace bullet points and special characters that might appear as squares
    bullet_pattern = re.compile(r'[\u2022\u25CF\u25E6\u2023\u2043\u2219\u25AA\u25AB]')
    text = bullet_pattern.sub('-', text)
    text = text.replace('•', '-')  # Bullet
    text = text.replace('·', '-')  # Middle dot
    
    # Replace problematic spaces
    space_pattern = re.compile(r'[\u00A0\u2000-\u200F\u2028-\u2029\uFEFF]')
    text = space_pattern.sub(' ', text)
    
    # Replace ellipsis
    text = text.replace('\u2026', '...')  # Horizontal ellipsis
    text = text.replace('…', '...')       # Ellipsis
    
    # Replace any box-drawing or geometric characters that might appear as squares
    # This includes box drawing characters (2500-257F) and geometric shapes (25A0-25FF)
    # Also catch the specific black square character (U+25A0)
    box_pattern = re.compile(r'[\u2500-\u257F\u25A0-\u25FF\u2580-\u259F]')
    text = box_pattern.sub('-', text)
    
    # Explicitly replace common square characters
    text = text.replace('\u25A0', '-')  # Black square
    text = text.replace('\u25A1', '-')  # White square
    text = text.replace('\u25AA', '-')  # Black small square
    text = text.replace('\u25AB', '-')  # White small square
    text = text.replace('■', '-')       # Black square (alternative)
    text = text.replace('□', '-')       # White square (alternative)
    
    # Replace any remaining non-printable or problematic characters
    # But preserve common Unicode characters like Greek letters (β, etc.)
    # Only remove control characters and truly problematic ones
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    
    return text.strip()


def format_resistance_genes(resistant_genes: List[str]) -> Optional[str]:
    """
    Format resistance genes list to readable format.
    
    Args:
        resistant_genes: List of resistance genes (e.g., ["vanA", "mecA"])
        
    Returns:
        Formatted string for use in prompts (e.g., "vanA and mecA" or "vanA"), or None if empty
    """
    if not resistant_genes:
        return None
    
    genes = [gene.strip() for gene in resistant_genes if gene and str(gene).strip()]
    
    if not genes:
        return None
    
    if len(genes) == 1:
        return genes[0]
    elif len(genes) == 2:
        return f"{genes[0]} and {genes[1]}"
    else:
        # Format as "gene1, gene2, and gene3"
        return ", ".join(genes[:-1]) + f", and {genes[-1]}"


def format_icd_codes(severity_codes: List[str]) -> str:
    """
    Format ICD codes list to readable format.
    
    Args:
        severity_codes: List of ICD codes (e.g., ["A41.9", "B95.3"])
        
    Returns:
        Formatted string for use in prompts (e.g., "A41.9 and B95.3" or "A41.9")
    """
    if not severity_codes:
        return "not specified"
    
    codes = [code.strip().upper() for code in severity_codes if code and str(code).strip()]
    
    if not codes:
        return "not specified"
    
    if len(codes) == 1:
        return codes[0]
    elif len(codes) == 2:
        return f"{codes[0]} and {codes[1]}"
    else:
        # Format as "code1, code2, and code3"
        return ", ".join(codes[:-1]) + f", and {codes[-1]}"


def get_icd_names_from_state(state: dict) -> str:
    """
    Get transformed ICD codes formatted as "ICD-10 Code (icd name)" from state.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Formatted ICD codes string (e.g., "A41.2 (Sepsis, other specified), A41.81 (Sepsis due to other specified organisms)")
    """
    icd_transformation = state.get('icd_transformation', {})
    code_names_list = icd_transformation.get('code_names', [])
    
    # Format as "ICD-10 Code (icd name)"
    if code_names_list:
        formatted_codes = []
        for item in code_names_list:
            if isinstance(item, dict):
                code = item.get('code', '')
                name = item.get('name', '')
                if code and name and name != code:
                    formatted_codes.append(f"{code} ({name})")
                elif code:
                    formatted_codes.append(code)
        
        if formatted_codes:
            return ', '.join(formatted_codes)
    
    # Fallback: try severity_codes_transformed
    severity_codes_transformed = icd_transformation.get('severity_codes_transformed', '')
    if severity_codes_transformed:
        return severity_codes_transformed
    
    # Fallback to original codes
    input_params = state.get('input_parameters', {})
    severity_codes = input_params.get('severity_codes', [])
    if severity_codes:
        return format_icd_codes(severity_codes)
    
    return "not specified"


def format_pathogens(pathogens: List[Dict[str, str]]) -> str:
    """
    Format pathogens list to readable format.
    
    Args:
        pathogens: List of dicts with 'pathogen_name' and 'pathogen_count'
        
    Returns:
        Formatted string for use in prompts (e.g., "Staphylococcus aureus (10^6 CFU/ML) and E. coli (10^5 CFU/ML)")
    """
    if not pathogens:
        return "unknown"
    
    formatted = []
    for pathogen in pathogens:
        if isinstance(pathogen, dict):
            name = pathogen.get('pathogen_name', '').strip()
            count = pathogen.get('pathogen_count', '').strip()
            if name:
                if count:
                    formatted.append(f"{name} ({count})")
                else:
                    formatted.append(name)
    
    if not formatted:
        return "unknown"
    
    if len(formatted) == 1:
        return formatted[0]
    elif len(formatted) == 2:
        return f"{formatted[0]} and {formatted[1]}"
    else:
        return ", ".join(formatted[:-1]) + f", and {formatted[-1]}"


def get_pathogens_from_input(input_params: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Get pathogens list from input parameters.
    
    Args:
        input_params: Input parameters dictionary
        
    Returns:
        List of pathogen dicts with 'pathogen_name' and 'pathogen_count'
    """
    pathogens = input_params.get('pathogens', [])
    if not isinstance(pathogens, list):
        return []
    
    result = []
    for p in pathogens:
        if isinstance(p, dict):
            name = p.get('pathogen_name', '').strip()
            count = p.get('pathogen_count', '').strip()
            if name:
                result.append({
                    'pathogen_name': name,
                    'pathogen_count': count
                })
    return result


def get_resistance_genes_from_input(input_params: Dict[str, Any]) -> List[str]:
    """
    Get resistance genes list from input parameters.
    
    Args:
        input_params: Input parameters dictionary
        
    Returns:
        List of resistance gene strings
    """
    genes = input_params.get('resistant_genes', [])
    if not isinstance(genes, list):
        return []
    
    return [str(g).strip() for g in genes if g and str(g).strip()]


def get_severity_codes_from_input(input_params: Dict[str, Any]) -> List[str]:
    """
    Get severity codes list from input parameters.
    
    Args:
        input_params: Input parameters dictionary
        
    Returns:
        List of ICD code strings
    """
    codes = input_params.get('severity_codes', [])
    if not isinstance(codes, list):
        return []
    
    return [str(c).strip().upper() for c in codes if c and str(c).strip()]


def create_llm() -> Optional[Ollama]:
    """
    Create LlamaIndex Ollama LLM instance.
    Shared utility function used across all nodes.
    
    Returns:
        Ollama LLM instance or None if unavailable
    """
    if not LLAMAINDEX_AVAILABLE:
        return None
    
    try:
        from config import get_ollama_config
        config = get_ollama_config()
        return Ollama(
            model=config['model'].replace('ollama/', ''),
            base_url=config['api_base'],
            temperature=config['temperature'],
            request_timeout=600.0
        )
    except Exception as e:
        logger.error(f"Failed to create Ollama LLM: {e}")
        return None


def clean_null_strings(data: Any) -> Any:
    """
    Recursively convert string 'null' values to actual None/null.
    Shared utility function for cleaning LLM output.
    
    Args:
        data: Data structure (dict, list, or primitive) to clean
        
    Returns:
        Cleaned data structure with 'null' strings converted to None
    """
    if isinstance(data, dict):
        return {k: clean_null_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_null_strings(item) for item in data]
    elif isinstance(data, str) and data.lower() == 'null':
        return None
    else:
        return data


def call_llm_with_retry(
    output_cls: Type[T],
    prompt: str,
    llm: Optional[Ollama],
    retry_delay: float = 2.0,
    max_attempts: Optional[int] = None,
    operation_name: str = "LLM operation",
    empty_result_handler: Optional[Callable[[], Optional[T]]] = None
) -> Optional[T]:
    """
    Call LLM with structured output and infinite retry logic.
    Shared utility function for all LLM calls across nodes.
    
    Args:
        output_cls: Pydantic model class for structured output
        prompt: Prompt string to send to LLM
        llm: Ollama LLM instance (can be None)
        retry_delay: Delay between retries in seconds
        max_attempts: Maximum number of attempts (None = infinite)
        operation_name: Name of operation for logging
        empty_result_handler: Optional function to call if result is empty
        
    Returns:
        Parsed result as output_cls instance, or None if max_attempts reached
    """
    if not LLAMAINDEX_AVAILABLE or not llm:
        logger.warning(f"{operation_name}: LLM not available")
        return None
    
    attempt = 0
    while True:
        attempt += 1
        if max_attempts and attempt > max_attempts:
            logger.error(f"{operation_name}: Max attempts ({max_attempts}) reached, giving up")
            return None
        
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=output_cls,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            
            result = program(input_str=prompt)
            
            if not result:
                logger.warning(f"{operation_name}: Empty result (attempt {attempt}), retrying...")
                if empty_result_handler:
                    fallback = empty_result_handler()
                    if fallback is not None:
                        return fallback
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            return result
            
        except Exception as e:
            logger.warning(f"{operation_name}: Error (attempt {attempt}): {e}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
