"""
Test script for rank_node.py
Loads saved extraction_result.json and tests rank_node without running the whole pipeline.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from nodes.rank_node import rank_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_extraction_result(file_path: str) -> Dict[str, Any]:
    """
    Load extraction_result.json file.
    
    Args:
        file_path: Path to the JSON file containing extraction result
        
    Returns:
        Dictionary with source_results and input_parameters
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract required fields
        source_results = data.get('source_results', [])
        input_parameters = data.get('input_parameters', {})
        
        if not source_results:
            logger.warning(f"No source_results found in {file_path}")
        
        logger.info(f"Loaded {len(source_results)} source results from {file_path}")
        logger.info(f"Input parameters: {input_parameters}")
        
        return {
            'source_results': source_results,
            'input_parameters': input_parameters
        }
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        raise


def create_mock_search_results(source_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create mock search_results from source_results for rank_node context.
    rank_node uses search_results to get source snippets for ranking context.
    
    Args:
        source_results: List of source result dictionaries
        
    Returns:
        List of mock search result dictionaries
    """
    mock_search_results = []
    
    for source_result in source_results:
        source_index = source_result.get('source_index', 0)
        source_title = source_result.get('source_title', '')
        source_url = source_result.get('source_url', '')
        
        # Create a mock snippet from the source title and URL
        # In real pipeline, this would come from Perplexity search
        snippet = f"Information about {source_title}. Source discusses treatment options and antibiotic recommendations."
        
        mock_search_results.append({
            'title': source_title,
            'url': source_url,
            'snippet': snippet
        })
    
    return mock_search_results


def save_rank_output(data: Dict[str, Any], output_path: str):
    """Save rank_node output to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Rank output saved to: {output_path}")


def test_rank_node(extraction_result_file: str, output_file: str = None):
    """
    Test rank_node with saved extraction_result.json.
    
    Args:
        extraction_result_file: Path to JSON file containing extraction result
        output_file: Optional path to save rank_node output (default: test_rank_output.json)
    """
    logger.info("=" * 60)
    logger.info("Testing rank_node")
    logger.info("=" * 60)
    
    # Load extraction result
    extraction_data = load_extraction_result(extraction_result_file)
    
    # Count not_known entries before ranking
    source_results = extraction_data['source_results']
    total_not_known_before = 0
    not_known_by_source = {}
    
    for source_result in source_results:
        source_index = source_result.get('source_index', 0)
        therapy_plan = source_result.get('antibiotic_therapy_plan', {})
        not_known_ab = therapy_plan.get('not_known', [])
        count = len(not_known_ab)
        total_not_known_before += count
        if count > 0:
            not_known_by_source[source_index] = count
            logger.info(f"  Source {source_index}: {count} 'not_known' antibiotics")
    
    logger.info(f"\nTotal 'not_known' antibiotics to rank: {total_not_known_before}")
    
    if total_not_known_before == 0:
        logger.warning("No 'not_known' antibiotics found. Nothing to rank.")
        return
    
    # Create mock search_results for context
    mock_search_results = create_mock_search_results(source_results)
    
    # Create state dictionary for rank_node
    state = {
        'source_results': source_results,
        'input_parameters': extraction_data['input_parameters'],
        'search_results': mock_search_results
    }
    
    # Run rank_node
    logger.info("\nRunning rank_node...")
    try:
        result = rank_node(state)
        
        # Get updated source_results
        updated_source_results = result.get('source_results', source_results)
        
        # Count not_known entries after ranking
        total_not_known_after = 0
        total_first_choice = 0
        total_second_choice = 0
        total_alternative = 0
        ranked_by_source = {}
        
        for source_result in updated_source_results:
            source_index = source_result.get('source_index', 0)
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            first = len(therapy_plan.get('first_choice', []))
            second = len(therapy_plan.get('second_choice', []))
            alternative = len(therapy_plan.get('alternative_antibiotic', []))
            not_known = len(therapy_plan.get('not_known', []))
            
            total_first_choice += first
            total_second_choice += second
            total_alternative += alternative
            total_not_known_after += not_known
            
            if source_index in not_known_by_source:
                ranked_by_source[source_index] = {
                    'before': not_known_by_source[source_index],
                    'after_not_known': not_known,
                    'moved_to_first': first,
                    'moved_to_second': second,
                    'moved_to_alternative': alternative
                }
        
        logger.info("\n" + "=" * 60)
        logger.info("Rank Results:")
        logger.info("=" * 60)
        logger.info(f"  - 'not_known' before ranking: {total_not_known_before}")
        logger.info(f"  - 'not_known' after ranking: {total_not_known_after}")
        logger.info(f"  - Moved to first_choice: {total_first_choice}")
        logger.info(f"  - Moved to second_choice: {total_second_choice}")
        logger.info(f"  - Moved to alternative_antibiotic: {total_alternative}")
        logger.info(f"  - Total ranked: {total_not_known_before - total_not_known_after}")
        
        # Show details by source
        if ranked_by_source:
            logger.info("\n  Ranking by source:")
            for source_index, stats in ranked_by_source.items():
                logger.info(f"    Source {source_index}:")
                logger.info(f"      - Before: {stats['before']} not_known")
                logger.info(f"      - After: {stats['after_not_known']} not_known, "
                          f"{stats['moved_to_first']} first_choice, "
                          f"{stats['moved_to_second']} second_choice, "
                          f"{stats['moved_to_alternative']} alternative")
        
        # Show some examples of ranked antibiotics
        logger.info("\n  Examples of ranked antibiotics:")
        example_count = 0
        for source_result in updated_source_results:
            if example_count >= 5:
                break
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            source_index = source_result.get('source_index', 0)
            
            # Show first_choice examples
            first_choice = therapy_plan.get('first_choice', [])
            for ab in first_choice[:2]:
                if example_count >= 5:
                    break
                logger.info(f"    - {ab.get('medical_name', 'N/A')} → first_choice (source {source_index})")
                example_count += 1
            
            # Show second_choice examples
            second_choice = therapy_plan.get('second_choice', [])
            for ab in second_choice[:2]:
                if example_count >= 5:
                    break
                logger.info(f"    - {ab.get('medical_name', 'N/A')} → second_choice (source {source_index})")
                example_count += 1
            
            # Show alternative examples
            alternative = therapy_plan.get('alternative_antibiotic', [])
            for ab in alternative[:2]:
                if example_count >= 5:
                    break
                logger.info(f"    - {ab.get('medical_name', 'N/A')} → alternative_antibiotic (source {source_index})")
                example_count += 1
        
        # Save output if output_file is specified
        if output_file:
            output_data = {
                'input_parameters': extraction_data['input_parameters'],
                'source_results': updated_source_results,
                'ranking_summary': {
                    'not_known_before': total_not_known_before,
                    'not_known_after': total_not_known_after,
                    'moved_to_first_choice': total_first_choice,
                    'moved_to_second_choice': total_second_choice,
                    'moved_to_alternative_antibiotic': total_alternative
                }
            }
            save_rank_output(output_data, output_file)
        else:
            # Default output filename
            output_dir = project_root / "output"
            output_file = output_dir / "test_rank_output.json"
            output_data = {
                'input_parameters': extraction_data['input_parameters'],
                'source_results': updated_source_results,
                'ranking_summary': {
                    'not_known_before': total_not_known_before,
                    'not_known_after': total_not_known_after,
                    'moved_to_first_choice': total_first_choice,
                    'moved_to_second_choice': total_second_choice,
                    'moved_to_alternative_antibiotic': total_alternative
                }
            }
            save_rank_output(output_data, str(output_file))
        
        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)
        
        return result
        
    except Exception as e:
        logger.error(f"Error running rank_node: {e}", exc_info=True)
        raise


def main():
    """Main function."""
    # Default to extraction_result.json if no argument provided
    if len(sys.argv) > 1:
        extraction_result_file = sys.argv[1]
    else:
        # Use extraction_result.json from output directory
        output_dir = project_root / "output"
        extraction_result_file = output_dir / "extraction_result.json"
        
        if not extraction_result_file.exists():
            logger.error(f"File not found: {extraction_result_file}")
            logger.error("Usage: python test_rank_node.py [path_to_extraction_result.json] [output_file.json]")
            sys.exit(1)
        
        logger.info(f"Using extraction_result.json: {extraction_result_file}")
    
    # Optional output file
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_rank_node(str(extraction_result_file), output_file)


if __name__ == "__main__":
    main()





