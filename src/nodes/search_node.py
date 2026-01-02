"""
Search node for LangGraph - Performs Perplexity search.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

try:
    from perplexity import Perplexity
except ImportError:
    print("Error: perplexityai package not installed. Install it with: pip install perplexityai")
    print("Note: The package name is 'perplexityai' but import is 'from perplexity import Perplexity'")
    Perplexity = None

from schemas import SearchResult

from prompts import SEARCH_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class PerplexitySearch:
    """Wrapper for Perplexity Search API using the official SDK."""
    
    def __init__(self, api_key: str, max_tokens: int = 25000, max_tokens_per_page: int = 2048):
        """
        Initialize Perplexity Search client.
        
        Args:
            api_key: Perplexity API key
            max_tokens: Maximum tokens for the search response
            max_tokens_per_page: Maximum tokens per page
        """
        if Perplexity is None:
            raise ImportError("perplexityai package is required. Install it with: pip install perplexityai")
        
        self.client = Perplexity(api_key=api_key)
        self.max_tokens = max_tokens
        self.max_tokens_per_page = max_tokens_per_page
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Perform a search query using Perplexity API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of search results with 'title', 'url', and 'snippet' fields
        """
        try:
            # Perform search using the SDK
            search = self.client.search.create(
                query=query,
                max_results=max_results,
                max_tokens=self.max_tokens,
                max_tokens_per_page=self.max_tokens_per_page
            )
            
            # Convert SDK results to our standardized format
            results = []
            if hasattr(search, 'results') and search.results:
                for result in search.results:
                    # Extract title, url, and snippet/text from result object
                    title = getattr(result, 'title', '') or getattr(result, 'name', 'No Title')
                    url = getattr(result, 'url', '') or getattr(result, 'link', '')
                    
                    # Try to get snippet/text from various possible attributes
                    snippet = (
                        getattr(result, 'snippet', None) or
                        getattr(result, 'text', None) or
                        getattr(result, 'description', None) or
                        getattr(result, 'content', None) or
                        ''
                    )
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet
                    })
            
            # If no results from search.results, try alternative attributes
            if not results:
                # Check if there's a response or answer attribute
                if hasattr(search, 'response') or hasattr(search, 'answer'):
                    content = getattr(search, 'response', None) or getattr(search, 'answer', '')
                    if content:
                        results.append({
                            'title': 'Perplexity Search Result',
                            'url': '',
                            'snippet': str(content)
                        })
            
            return results
            
        except Exception as e:
            logger.error(f"Error calling Perplexity API: {e}")
            import traceback
            traceback.print_exc()
            return []


def format_search_query(**kwargs) -> str:
    """
    Format the search template with dynamic inputs.
    
    Args:
        **kwargs: Values to fill in the template (pathogen_name, resistant_gene, panel, severity_codes_text)
        
    Returns:
        Formatted query string
    """
    try:
        # Build conditional resistance phrase
        resistant_gene = kwargs.get('resistant_gene')
        if resistant_gene:
            resistance_phrase = f" with {resistant_gene} resistance"
        else:
            resistance_phrase = ""
        
        # Build condition text based on panel
        panel = kwargs.get('panel')
        condition_text = ""
        if panel and panel != 'N/A' and panel != 'Not specified':
            panel_condition_map = {
                'Blood': 'bacteremia or sepsis',
                'Urine': 'UTI or urinary tract infection',
                'Sputum': 'pneumonia or respiratory infection',
                'Respiratory': 'pneumonia or respiratory infection',
                'CSF': 'meningitis',
                'Wound': 'wound infection',
                'Nail': 'onychomycosis or tinea unguium',
                'Skin': 'skin infection or dermatitis',
                'Vaginal': 'vaginitis or vaginal infection',
                'RT-PCR Monkeypox Virus (F3L gene) RT-PCR': 'monkeypox or mpox',
                'SARS - CoV2 + RSV + INFLUENZA A & B': 'COVID-19 or SARS-CoV-2 or RSV or influenza',
                'SARS CoV2 ONLY': 'COVID-19 or SARS-CoV-2',
                'RESPIRATORY TRACT PANEL (RPP)': 'respiratory tract infection or pneumonia',
                'UTI': 'UTI or urinary tract infection',
                'FUNGAL, SEPSIS & WOUND PANEL': 'fungal infection or sepsis or wound infection',
                'SEXUALLY TRANSMITTED INFECTION PANEL (STI)': 'sexually transmitted infection or STI',
                'HELICOBACTER PYLORI': 'Helicobacter pylori or H. pylori or peptic ulcer',
                'GASTROENTERITIS PANEL': 'gastroenteritis or gastrointestinal infection',
                'Womens Health Panel (Vaginosis)': 'bacterial vaginosis or vaginitis or vaginal infection'
            }
            condition = panel_condition_map.get(panel, panel.lower())
            condition_text = f" for {condition}"
        
        kwargs['resistance_phrase'] = resistance_phrase
        kwargs['condition_text'] = condition_text
        return SEARCH_PROMPT_TEMPLATE.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing template parameter: {e}")
        logger.error(f"Required parameter: pathogen_name")
        return SEARCH_PROMPT_TEMPLATE


def _save_search_cache(cache_path: str, search_query: str, search_results: List[Dict]) -> None:
    """
    Save search results to cache file.
    
    Args:
        cache_path: Path to cache file
        search_query: The search query used
        search_results: List of search results to cache
    """
    try:
        from config import get_output_config
        output_config = get_output_config()
        
        # Check if saving is enabled
        if not output_config.get('save_enabled', True):
            logger.debug("Saving search cache disabled (production mode)")
            return
    except Exception:
        # If config check fails, allow saving (fallback)
        pass
    
    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'search_query': search_query,
        'search_results': search_results,
        'cached_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Search results cached to {cache_path}")


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that performs Perplexity search based on input parameters.
    Uses cached results if available, otherwise performs new search.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with search_query and search_results
    """
    try:
        input_params = state.get('input_parameters', {})
        
        # Get pathogens
        from utils import get_pathogens_from_input, format_pathogens
        pathogens = get_pathogens_from_input(input_params)
        pathogen_name = format_pathogens(pathogens)
        
        # Get resistance genes (returns None if empty)
        from utils import get_resistance_genes_from_input, format_resistance_genes
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistant_gene = format_resistance_genes(resistant_genes)  # Returns None if empty
        
        # Get transformed ICD codes if available, otherwise use original
        icd_transformation = state.get('icd_transformation', {})
        code_names_list = icd_transformation.get('code_names', [])
        
        # Format ICD codes as "ICD-10 Code (icd name)"
        severity_codes_text = ''
        if code_names_list:
            # Format as "Code (Name)" for each ICD code
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
                icd_formatted = ', '.join(formatted_codes)
                severity_codes_text = f" for patient with ICD-10 codes {icd_formatted}"
        
        # Fallback: if no names extracted, try using severity_codes_transformed
        if not severity_codes_text:
            severity_codes_transformed = icd_transformation.get('severity_codes_transformed', '')
            if not severity_codes_transformed:
                from utils import get_severity_codes_from_input, format_icd_codes
                severity_codes_list = get_severity_codes_from_input(input_params)
                severity_codes_transformed = format_icd_codes(severity_codes_list)
            
            if severity_codes_transformed:
                # Extract codes and names from format like "A41.2 (Sepsis)"
                # Or extract from parentheses as fallback
                if '(' in severity_codes_transformed:
                    # Already has format with parentheses
                    severity_codes_text = f" for patient with ICD-10 codes {severity_codes_transformed}"
                else:
                    # Just codes, try to get names from code_names_list
                    from utils import get_severity_codes_from_input
                    severity_codes_list = get_severity_codes_from_input(input_params)
                    if severity_codes_list and code_names_list:
                        # Match codes with names
                        code_to_name = {item.get('code'): item.get('name') for item in code_names_list if isinstance(item, dict)}
                        formatted_codes = []
                        for code in severity_codes_list:
                            name = code_to_name.get(code)
                            if name and name != code:
                                formatted_codes.append(f"{code} ({name})")
                            else:
                                formatted_codes.append(code)
                        if formatted_codes:
                            icd_formatted = ', '.join(formatted_codes)
                            severity_codes_text = f" for patient with ICD-10 codes {icd_formatted}"
        
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        
        # Emit progress for search start
        if progress_callback:
            progress_callback('search', 0, 'Starting search...')
        
        # Get panel from input parameters
        panel = input_params.get('panel', '')
        
        # Format search query
        search_query = format_search_query(
            pathogen_name=pathogen_name,
            resistant_gene=resistant_gene,
            panel=panel,
            severity_codes_text=severity_codes_text
        )
        
        # Check for cached results first - always use if available
        cached_data = state.get('metadata', {}).get('cached_search_results')
        used_cache = False
        
        if cached_data and cached_data.get('search_results'):
            logger.info(f"Using cached search results (from cache)")
            raw_results = cached_data['search_results']
            # Limit to max_results if needed
            max_results = state.get('metadata', {}).get('max_search_results', 5)
            if len(raw_results) > max_results:
                raw_results = raw_results[:max_results]
            used_cache = True
            # Emit progress for cache hit
            if progress_callback:
                progress_callback('search', 50, f'Using cached results ({len(raw_results)} results)')
        else:
            # Perform new search if no cache
            logger.info(f"Performing Perplexity search: {search_query[:100]}...")
            
            # Emit progress for search in progress
            if progress_callback:
                progress_callback('search', 30, 'Performing Perplexity search...')
            
            # Get Perplexity client from state metadata
            perplexity_client: PerplexitySearch = state.get('metadata', {}).get('perplexity_client')
            if not perplexity_client:
                raise ValueError("Perplexity client not found in state metadata")
            
            # Perform search
            max_results = state.get('metadata', {}).get('max_search_results', 5)
            raw_results = perplexity_client.search(search_query, max_results=max_results)
            
            # Emit progress for search complete
            if progress_callback:
                progress_callback('search', 80, f'Search completed ({len(raw_results)} results)')
        
        # Convert to SearchResult schema
        search_results = [
            SearchResult(
                title=result.get('title', 'No Title'),
                url=result.get('url', ''),
                snippet=result.get('snippet', '')
            )
            for result in raw_results
        ]
        
        logger.info(f"Found {len(search_results)} search results")
        
        # Emit progress for search complete
        if progress_callback:
            progress_callback('search', 100, f'Search complete ({len(search_results)} results)')
        
        # Save cache if we performed a new search (not from cache)
        if not used_cache and raw_results:
            cache_path = state.get('metadata', {}).get('cache_path')
            if cache_path:
                try:
                    _save_search_cache(cache_path, search_query, raw_results)
                except Exception as e:
                    logger.warning(f"Failed to save cache: {e}")
        
        return {
            'search_query': search_query,
            'search_results': [result.model_dump() for result in search_results]
        }
        
    except Exception as e:
        logger.error(f"Error in search_node: {e}", exc_info=True)
        return {
            'errors': state.get('errors', []) + [f"Search error: {str(e)}"]
        }

