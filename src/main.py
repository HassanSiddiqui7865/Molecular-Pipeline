"""
Main Orchestration Script
Coordinates Perplexity search and LangChain-based extraction using LangGraph.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Dict

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nodes.search_node import PerplexitySearch
from graph import create_pipeline_graph, run_pipeline
from schemas import OutputData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def save_output(data: Dict, output_path: str):
    """Save extracted data to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Output saved to: {output_path}")


def get_cache_filename(inputs: Dict) -> Path:
    """Generate cache filename based on input parameters."""
    import hashlib
    import json
    
    # Get pathogens - support both formats
    from utils import get_pathogens_from_input, get_resistance_genes_from_input, get_severity_codes_from_input
    
    pathogens = get_pathogens_from_input(inputs)
    resistant_genes = get_resistance_genes_from_input(inputs)
    severity_codes = get_severity_codes_from_input(inputs)
    
    # Create cache key from arrays
    pathogens_str = json.dumps(pathogens, sort_keys=True) if pathogens else ''
    genes_str = json.dumps(resistant_genes, sort_keys=True) if resistant_genes else ''
    codes_str = json.dumps(severity_codes, sort_keys=True) if severity_codes else ''
    
    cache_key = f"{pathogens_str}_{genes_str}_{codes_str}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return project_root / "output" / f"perplexity_cache_{cache_hash}.json"


def load_cached_search_results(cache_path: Path) -> Dict:
    """Load cached search results if they exist."""
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'search_results' in data:
                    logger.info(f"Loading cached search results from {cache_path}")
                    return data
        except Exception as e:
            logger.warning(f"Error loading cache: {e}")
    return None


def main():
    """Main execution function."""
    # Load configuration from environment variables
    from config import get_perplexity_config
    perplexity_config = get_perplexity_config()
    
    if not perplexity_config.get('api_key') or perplexity_config.get('api_key') == 'YOUR_PERPLEXITY_API_KEY':
        logger.error("Please set your PERPLEXITY_API_KEY in .env file")
        sys.exit(1)
    
    perplexity = PerplexitySearch(
        api_key=perplexity_config['api_key'],
        max_tokens=perplexity_config.get('max_tokens', 50000),
        max_tokens_per_page=perplexity_config.get('max_tokens_per_page', 4096)
    )
    
    # Ollama configuration is now handled in each node that needs it
    
    # Load inputs from command-line argument (JSON file path) or stdin
    inputs = None
    
    if len(sys.argv) > 1:
        # First argument: path to JSON file with input parameters
        input_file = Path(sys.argv[1])
        if input_file.exists():
            with open(input_file, 'r', encoding='utf-8') as f:
                inputs = json.load(f)
        else:
            logger.error(f"Input file not found at {input_file}")
            sys.exit(1)
    else:
        # Try to read from stdin
        try:
            if not sys.stdin.isatty():
                inputs = json.load(sys.stdin)
            else:
                logger.error("No input provided. Usage: python main.py <input.json>")
                logger.error("Or pipe JSON input: echo '{\"pathogens\":[...]}' | python main.py")
                sys.exit(1)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON input: {e}")
            sys.exit(1)
    
    if not inputs:
        logger.error("No input parameters provided")
        sys.exit(1)
    
    logger.info(f"Input parameters: {inputs}")
    
    # Check for cached search results - always use if available
    cache_path = get_cache_filename(inputs)
    cached_data = load_cached_search_results(cache_path)
    
    # Create LangGraph pipeline
    graph = create_pipeline_graph()
    
    # Run pipeline
    logger.info("Running LangGraph pipeline...")
    max_search_results = perplexity_config.get('max_search_results', 10)
    final_state = run_pipeline(
        graph=graph,
        input_parameters=inputs,
        perplexity_client=perplexity,
        max_search_results=max_search_results,
        cached_search_results=cached_data,
        cache_path=str(cache_path)
    )
    
    # Prepare final output - include all state data needed for PDF export
    output_data = {
        'input_parameters': final_state.get('input_parameters', inputs),
        'extraction_date': final_state.get('extraction_date'),
        'result': final_state.get('result', {}),
        'icd_transformation': final_state.get('icd_transformation', {}),
        # Include negative organisms and genes if available in metadata or result
        'negative_organisms': (
            final_state.get('metadata', {}).get('negative_organisms') or
            final_state.get('result', {}).get('negative_organisms') or
            []
        ),
        'negative_resistance_genes': (
            final_state.get('metadata', {}).get('negative_resistance_genes') or
            final_state.get('result', {}).get('negative_resistance_genes') or
            []
        )
    }
    
    # Validate output with Pydantic (but preserve result even if validation fails)
    result_backup = output_data.get('result', {})
    try:
        validated_output = OutputData(**output_data)
        output_data = validated_output.model_dump()
        # Ensure result is preserved
        if result_backup and not output_data.get('result'):
            output_data['result'] = result_backup
    except Exception as e:
        logger.warning(f"Output validation warning: {e}")
        # Continue with output_data even if validation fails
        # Ensure result is preserved
        if result_backup:
            output_data['result'] = result_backup
    
    # Save output with timestamp (only if saving is enabled)
    from config import get_output_config
    output_config = get_output_config()
    
    if output_config.get('save_enabled', True):
        output_dir = project_root / output_config.get('directory', 'output')
        
        # Generate timestamp for filename
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get base filename and add timestamp before .json extension
        base_filename = output_config.get('filename', 'pathogen_info_output.json')
        if base_filename.endswith('.json'):
            filename_with_timestamp = base_filename.replace('.json', f'_{timestamp}.json')
        else:
            filename_with_timestamp = f"{base_filename}_{timestamp}.json"
        
        output_path = output_dir / filename_with_timestamp
        save_output(output_data, str(output_path))
    else:
        logger.info("Saving output disabled (production mode)")
    
    # Log result summary
    result = output_data.get('result', {})
    if result:
        therapy_plan = result.get('antibiotic_therapy_plan', {})
        first_count = len(therapy_plan.get('first_choice', []))
        second_count = len(therapy_plan.get('second_choice', []))
        alt_count = len(therapy_plan.get('alternative_antibiotic', []))
        logger.info(f"Result: {first_count} first_choice, {second_count} second_choice, {alt_count} alternative antibiotics")
    
    logger.info(f"Extraction complete")


if __name__ == "__main__":
    main()

