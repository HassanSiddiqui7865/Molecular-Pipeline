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

from utils import format_resistance_genes

logger = logging.getLogger(__name__)


# Search prompt template for pathogen information extraction
SEARCH_PROMPT_TEMPLATE = """Evidence-based first-line and second-line antibiotic therapy for {pathogen_name} with {resistant_gene} resistance, covering drug selection, dosage, treatment duration, and stewardship considerations{severity_codes_text}"""


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
        **kwargs: Values to fill in the template (pathogen_name, resistant_gene)
        
    Returns:
        Formatted query string
    """
    try:
        return SEARCH_PROMPT_TEMPLATE.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing template parameter: {e}")
        logger.error(f"Required parameters: pathogen_name, resistant_gene")
        return SEARCH_PROMPT_TEMPLATE


def _save_search_cache(cache_path: str, search_query: str, search_results: List[Dict]) -> None:
    """
    Save search results to cache file.
    
    Args:
        cache_path: Path to cache file
        search_query: The search query used
        search_results: List of search results to cache
    """
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
        
        # Get resistance genes
        from utils import get_resistance_genes_from_input, format_resistance_genes
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistant_gene = format_resistance_genes(resistant_genes)
        
        # Get transformed ICD codes if available, otherwise use original
        icd_transformation = state.get('icd_transformation', {})
        severity_codes_transformed = icd_transformation.get('severity_codes_transformed', '')
        
        # If transformation failed or not available, use original codes
        if not severity_codes_transformed:
            from utils import get_severity_codes_from_input, format_icd_codes
            severity_codes_list = get_severity_codes_from_input(input_params)
            severity_codes_transformed = format_icd_codes(severity_codes_list)
        
        # Add ICD codes to search query if available
        severity_codes_text = ''
        if severity_codes_transformed:
            severity_codes_text = f" for patient with ICD-10 codes: {severity_codes_transformed}"
        
        # Format search query
        search_query = format_search_query(
            pathogen_name=pathogen_name,
            resistant_gene=resistant_gene,
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
        else:
            # Perform new search if no cache
            logger.info(f"Performing Perplexity search: {search_query[:100]}...")
            
            # Get Perplexity client from state metadata
            perplexity_client: PerplexitySearch = state.get('metadata', {}).get('perplexity_client')
            if not perplexity_client:
                raise ValueError("Perplexity client not found in state metadata")
            
            # Perform search
            max_results = state.get('metadata', {}).get('max_search_results', 5)
            raw_results = perplexity_client.search(search_query, max_results=max_results)
        
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

