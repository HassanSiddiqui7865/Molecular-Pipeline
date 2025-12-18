"""
Test script for extract_node.py
Tests the extraction node by loading search results from perplexity cache files.
"""
import json
import sys
import logging
import re
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from nodes.extract_node import extract_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
        
        logger.info(f"Loaded {len(search_results)} search results from {file_path}")
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


def extract_input_from_cache(cache_file: str) -> Dict[str, Any]:
    """
    Extract input parameters from perplexity cache file by parsing the search query.
    
    Args:
        cache_file: Path to perplexity cache JSON file
        
    Returns:
        Dictionary with input parameters
    """
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        search_query = data.get('search_query', '')
        
        # Parse search query to extract input parameters
        # Example: "Evidence-based first-line and second-line antibiotic therapy for Enterococcus faecium with vanA resistance..."
        input_params = {}
        
        # Extract pathogen name (look for "for [Pathogen] with")
        pathogen_match = re.search(r'for\s+([^w]+?)\s+with\s+([^\s,]+)', search_query, re.IGNORECASE)
        if pathogen_match:
            input_params['pathogen_name'] = pathogen_match.group(1).strip()
            input_params['resistant_gene'] = pathogen_match.group(2).strip()
        else:
            # Fallback: try to find common patterns
            if 'Enterococcus faecium' in search_query:
                input_params['pathogen_name'] = 'Enterococcus faecium'
            if 'vanA' in search_query:
                input_params['resistant_gene'] = 'vanA'
        
        # Extract ICD codes (look for "ICD-10 codes:" or "ICD codes:")
        icd_match = re.search(r'ICD[-\s]?10\s+codes?[:\s]+([A-Z0-9.,\s()]+)', search_query, re.IGNORECASE)
        if icd_match:
            icd_text = icd_match.group(1)
            # Extract codes like "A41.9 (Sepsis...), B95.3 (...)"
            codes = re.findall(r'([A-Z]\d+\.\d+)', icd_text)
            if codes:
                input_params['severity_codes'] = ', '.join(codes)
        
        # Set defaults for missing fields
        input_params.setdefault('pathogen_name', 'Enterococcus faecium')
        input_params.setdefault('resistant_gene', 'vanA')
        input_params.setdefault('pathogen_count', '10^5 CFU/ML')
        input_params.setdefault('severity_codes', 'A41.9, B95.3')
        input_params.setdefault('age', 45)
        
        logger.info(f"Extracted input parameters from cache:")
        logger.info(f"  - Pathogen: {input_params['pathogen_name']}")
        logger.info(f"  - Resistance Gene: {input_params['resistant_gene']}")
        logger.info(f"  - Severity Codes: {input_params['severity_codes']}")
        
        return input_params
    except Exception as e:
        logger.error(f"Error extracting input from cache: {e}")
        raise


def save_extraction_result(data: Dict[str, Any], output_path: str):
    """Save extract_node output to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Extraction result saved to: {output_path}")


def print_extraction_summary(source_results: List[Dict[str, Any]]):
    """
    Print a summary of extraction results.
    
    Args:
        source_results: List of source extraction results
    """
    logger.info("\n" + "=" * 60)
    logger.info("Extraction Summary")
    logger.info("=" * 60)
    logger.info(f"Total sources processed: {len(source_results)}")
    
    total_first = 0
    total_second = 0
    total_alt = 0
    total_not_known = 0
    total_resistance_genes = 0
    
    for idx, source in enumerate(source_results, 1):
        therapy_plan = source.get('antibiotic_therapy_plan', {})
        first = len(therapy_plan.get('first_choice', []))
        second = len(therapy_plan.get('second_choice', []))
        alt = len(therapy_plan.get('alternative_antibiotic', []))
        not_known = len(therapy_plan.get('not_known', []))
        resistance_genes = len(source.get('pharmacist_analysis_on_resistant_gene', []))
        
        total_first += first
        total_second += second
        total_alt += alt
        total_not_known += not_known
        total_resistance_genes += resistance_genes
        
        logger.info(f"\nSource {idx}: {source.get('source_title', 'N/A')[:60]}")
        logger.info(f"  - First choice: {first}")
        logger.info(f"  - Second choice: {second}")
        logger.info(f"  - Alternative: {alt}")
        logger.info(f"  - Not known: {not_known}")
        logger.info(f"  - Resistance genes: {resistance_genes}")
        
        # Show sample antibiotics
        if first > 0:
            first_ab = therapy_plan.get('first_choice', [])[0]
            logger.info(f"    Sample first choice: {first_ab.get('medical_name', 'N/A')}")
        if second > 0:
            second_ab = therapy_plan.get('second_choice', [])[0]
            logger.info(f"    Sample second choice: {second_ab.get('medical_name', 'N/A')}")
        if alt > 0:
            alt_ab = therapy_plan.get('alternative_antibiotic', [])[0]
            logger.info(f"    Sample alternative: {alt_ab.get('medical_name', 'N/A')}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Total Summary:")
    logger.info("=" * 60)
    logger.info(f"  - First choice antibiotics: {total_first}")
    logger.info(f"  - Second choice antibiotics: {total_second}")
    logger.info(f"  - Alternative antibiotics: {total_alt}")
    logger.info(f"  - Not known antibiotics: {total_not_known}")
    logger.info(f"  - Resistance gene entries: {total_resistance_genes}")
    logger.info(f"  - Total antibiotics extracted: {total_first + total_second + total_alt + total_not_known}")


def test_extract_node(
    input_parameters: Dict[str, Any],
    search_results: List[Dict[str, Any]],
    output_file: str = None
) -> Dict[str, Any]:
    """
    Test extract_node with provided input parameters and search results.
    
    Args:
        input_parameters: Dictionary with pathogen_name, resistant_gene, pathogen_count, severity_codes, age
        search_results: List of search results (required)
        output_file: Optional path to save extraction result (default: output/extraction_result.json)
        
    Returns:
        Dictionary with extraction results
    """
    logger.info("=" * 60)
    logger.info("Testing extract_node")
    logger.info("=" * 60)
    
    if not search_results:
        logger.error("No search results provided. Please provide a perplexity cache file.")
        raise ValueError("search_results is required")
    
    # Create state dictionary for extract_node
    state = {
        'search_results': search_results,
        'input_parameters': input_parameters
    }
    
    # Log input summary
    logger.info(f"\nInput Parameters:")
    logger.info(f"  - Pathogen: {input_parameters.get('pathogen_name', 'N/A')}")
    logger.info(f"  - Resistance Gene: {input_parameters.get('resistant_gene', 'N/A')}")
    logger.info(f"  - Pathogen Count: {input_parameters.get('pathogen_count', 'N/A')}")
    logger.info(f"  - Severity Codes: {input_parameters.get('severity_codes', 'N/A')}")
    logger.info(f"  - Age: {input_parameters.get('age', 'N/A')}")
    logger.info(f"  - Number of search results: {len(search_results)}")
    
    # Run extract_node
    logger.info("\nRunning extract_node...")
    try:
        result = extract_node(state)
        
        source_results = result.get('source_results', [])
        
        if not source_results:
            logger.warning("No source results returned from extract_node")
            return result
        
        # Print summary
        print_extraction_summary(source_results)
        
        # Save output if output_file is specified
        if output_file:
            output_data = {
                'input_parameters': input_parameters,
                'source_results': source_results
            }
            save_extraction_result(output_data, output_file)
        else:
            # Default output filename
            output_dir = project_root / "output"
            output_file = output_dir / "extraction_result.json"
            output_data = {
                'input_parameters': input_parameters,
                'source_results': source_results
            }
            save_extraction_result(output_data, str(output_file))
        
        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Error running extract_node: {e}", exc_info=True)
        raise


def main():
    """Main function."""
    # Default input parameters for testing
    default_input_parameters = {
        'pathogen_name': 'Enterococcus faecium',
        'resistant_gene': 'vanA',
        'pathogen_count': '10^5 CFU/ML',
        'severity_codes': 'A41.9, B95.3',
        'age': 45
    }
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        # First argument: path to search results JSON file or perplexity cache
        search_results_file = sys.argv[1]
        search_results = load_search_results(search_results_file)
        
        # Try to extract input parameters from perplexity cache
        try:
            input_parameters = extract_input_from_cache(search_results_file)
            logger.info("Extracted input parameters from perplexity cache file")
        except:
            # If extraction fails, use defaults or check for second argument
            if len(sys.argv) > 2:
                input_params_file = sys.argv[2]
                with open(input_params_file, 'r', encoding='utf-8') as f:
                    input_parameters = json.load(f)
            else:
                input_parameters = default_input_parameters
                logger.info("Using default input parameters. Could not extract from cache.")
    else:
        # No arguments: try to find most recent perplexity cache
        output_dir = project_root / "output"
        cache_files = list(output_dir.glob("perplexity_cache_*.json"))
        
        if cache_files:
            # Use most recent cache file
            latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using most recent perplexity cache: {latest_cache.name}")
            search_results = load_search_results(str(latest_cache))
            try:
                input_parameters = extract_input_from_cache(str(latest_cache))
                logger.info("Extracted input parameters from perplexity cache file")
            except:
                input_parameters = default_input_parameters
                logger.info("Using default input parameters. Could not extract from cache.")
        else:
            # No cache files found
            logger.error("No perplexity cache files found in output/ directory.")
            logger.error("Please provide a perplexity cache file as an argument:")
            logger.error("  python test_extract_node.py output/perplexity_cache_*.json")
            sys.exit(1)
    
    # Third argument: output file path (optional)
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Run test
    test_extract_node(
        input_parameters=input_parameters,
        search_results=search_results,
        output_file=output_file
    )


if __name__ == "__main__":
    main()

