"""
Test script for extract_node.py
Tests the extraction node by loading search results from perplexity cache files.
"""
import json
import sys
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nodes.extract_node import extract_node
from nodes.icd_transform_node import icd_transform_node
from nodes.search_node import search_node, PerplexitySearch
from config import get_perplexity_config
from utils import get_pathogens_from_input, get_resistance_genes_from_input, get_severity_codes_from_input

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_search_results(file_path: str) -> List[Dict[str, Any]]:
    """
    Load search results from a JSON file (perplexity cache or other format).
    
    Args:
        file_path: Path to JSON file containing search results
        
    Returns:
        List of search result dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            search_results = data
        elif isinstance(data, dict):
            # Check for perplexity cache format
            if 'search_results' in data:
                search_results = data.get('search_results', [])
            elif 'results' in data:
                search_results = data.get('results', [])
            else:
                logger.error(f"Unexpected JSON structure in {file_path}")
                return []
        else:
            logger.error(f"Unexpected JSON structure in {file_path}")
            return []
        
        logger.info(f"Loaded {len(search_results)} results from {Path(file_path).name}")
        return search_results
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise


def save_extraction_result(data: Dict[str, Any], output_path: str):
    """Save extract_node output to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved: {Path(output_path).name}")


def print_extraction_summary(source_results: List[Dict[str, Any]]):
    """Print a simple summary of extraction results."""
    total_first = 0
    total_second = 0
    total_alt = 0
    total_not_known = 0
    total_resistance_genes = 0
    
    for source in source_results:
        therapy_plan = source.get('antibiotic_therapy_plan', {})
        total_first += len(therapy_plan.get('first_choice', []))
        total_second += len(therapy_plan.get('second_choice', []))
        total_alt += len(therapy_plan.get('alternative_antibiotic', []))
        total_not_known += len(therapy_plan.get('not_known', []))
        total_resistance_genes += len(source.get('pharmacist_analysis_on_resistant_gene', []))
    
    logger.info(f"Summary: {len(source_results)} sources | "
                f"First: {total_first} | Second: {total_second} | Alt: {total_alt} | "
                f"Unknown: {total_not_known} | Genes: {total_resistance_genes}")


def test_extract_node(
    input_parameters: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    output_file: str = None
) -> Dict[str, Any]:
    """Test extract_node with provided input parameters and search results."""
    if not search_results:
        logger.error("No search results provided")
        raise ValueError("search_results is required")
    
    logger.info(f"Processing {len(search_results)} sources...")
    
    state = {
        'search_results': search_results,
        'input_parameters': input_parameters
    }
    
    try:
        result = extract_node(state)
        source_results = result.get('source_results', [])
        
        if not source_results:
            logger.warning("No results returned")
            return result
        
        print_extraction_summary(source_results)
        
        # Save output
        if not output_file:
            output_dir = project_root / "output"
            output_file = output_dir / "extraction_result.json"
        
        save_extraction_result({
            'input_parameters': input_parameters,
            'source_results': source_results
        }, str(output_file))
        
        logger.info("Test completed")
        return result
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


def get_inputs_from_file() -> Dict[str, Any]:
    """Load input parameters from input.json."""
    input_file = project_root / "input.json"
    
    if not input_file.exists():
        logger.error(f"input.json not found at {input_file}")
        raise FileNotFoundError(f"input.json not found at {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """Main function."""
    # Load inputs from input.json
    input_parameters = get_inputs_from_file()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # First argument: path to search results JSON file or perplexity cache
        search_results_file = sys.argv[1]
        search_results = load_search_results(search_results_file)
    else:
        # No arguments: try to find most recent perplexity cache
        output_dir = project_root / "output"
        cache_files = list(output_dir.glob("perplexity_cache_*.json"))
        
        if cache_files:
            latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using cache: {latest_cache.name}")
            search_results = load_search_results(str(latest_cache))
        else:
            # No cache found - perform ICD transformation and search
            logger.info("No cache files found. Performing ICD transformation and search...")
            
            # Initialize state
            state = {
                'input_parameters': input_parameters
            }
            
            # Step 1: Transform ICD codes to names
            logger.info("Transforming ICD codes to names...")
            icd_result = icd_transform_node(state)
            state.update(icd_result)
            
            # Step 2: Perform search
            logger.info("Performing Perplexity search...")
            
            # Get Perplexity configuration
            perplexity_config = get_perplexity_config()
            if not perplexity_config.get('api_key') or perplexity_config.get('api_key') == 'YOUR_PERPLEXITY_API_KEY':
                logger.error("Please set your PERPLEXITY_API_KEY in .env file")
                sys.exit(1)
            
            # Create Perplexity client
            perplexity_client = PerplexitySearch(
                api_key=perplexity_config['api_key'],
                max_tokens=perplexity_config.get('max_tokens', 50000),
                max_tokens_per_page=perplexity_config.get('max_tokens_per_page', 4096)
            )
            
            # Generate cache path
            pathogens = get_pathogens_from_input(input_parameters)
            resistant_genes = get_resistance_genes_from_input(input_parameters)
            severity_codes = get_severity_codes_from_input(input_parameters)
            
            pathogens_str = json.dumps(pathogens, sort_keys=True) if pathogens else ''
            genes_str = json.dumps(resistant_genes, sort_keys=True) if resistant_genes else ''
            codes_str = json.dumps(severity_codes, sort_keys=True) if severity_codes else ''
            
            cache_key = f"{pathogens_str}_{genes_str}_{codes_str}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_path = output_dir / f"perplexity_cache_{cache_hash}.json"
            
            # Add Perplexity client and metadata to state
            state['metadata'] = {
                'perplexity_client': perplexity_client,
                'max_search_results': perplexity_config.get('max_search_results', 10),
                'cache_path': str(cache_path)
            }
            
            # Perform search
            search_result = search_node(state)
            state.update(search_result)
            
            # Extract search results (already in dict format from search_node)
            search_results = search_result.get('search_results', [])
            
            logger.info(f"Search completed. Found {len(search_results)} results.")
    
    # Second argument: output file path (optional)
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run test
    test_extract_node(
        input_parameters=input_parameters,
        search_results=search_results,
        output_file=output_file
    )


if __name__ == "__main__":
    main()

