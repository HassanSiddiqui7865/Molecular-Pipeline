"""
Test script for synthesize_node.py
Loads saved extract_node output and tests synthesize_node without running the whole pipeline.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nodes.synthesize_node import synthesize_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_extract_output(file_path: str) -> Dict[str, Any]:
    """
    Load extract_node output from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing extract_node output
        
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


def save_synthesize_output(data: Dict[str, Any], output_path: str):
    """Save synthesize_node output to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Synthesize output saved to: {output_path}")


def test_synthesize_node(extract_output_file: str, output_file: str = None):
    """
    Test synthesize_node with saved rank_node output.
    
    Args:
        extract_output_file: Path to JSON file containing rank_node output
        output_file: Optional path to save synthesize_node output (default: synthesize_result.json)
    """
    # Load rank_node output
    extract_data = load_extract_output(extract_output_file)
    
    # Create state dictionary for synthesize_node
    state = {
        'source_results': extract_data['source_results'],
        'input_parameters': extract_data['input_parameters']
    }
    
    # Run synthesize_node
    try:
        result = synthesize_node(state)
        
        # Extract result
        unified_result = result.get('result', {})
        
        if unified_result:
            therapy_plan = unified_result.get('antibiotic_therapy_plan', {})
            first_choice = therapy_plan.get('first_choice', [])
            second_choice = therapy_plan.get('second_choice', [])
            alternative = therapy_plan.get('alternative_antibiotic', [])
            resistance_genes = unified_result.get('pharmacist_analysis_on_resistant_gene', [])
            
            logger.info(
                f"Result: {len(first_choice)} first_choice, "
                f"{len(second_choice)} second_choice, "
                f"{len(alternative)} alternative, "
                f"{len(resistance_genes)} resistance genes"
            )
        else:
            logger.warning("No result returned from synthesize_node")
        
        # Save output (only the synthesized result, not source_results)
        if not output_file:
            output_dir = project_root / "output"
            output_file = output_dir / "synthesize_result.json"
        
        output_data = {
            'input_parameters': state['input_parameters'],
            'result': unified_result
        }
        save_synthesize_output(output_data, str(output_file))
        
        logger.info("Test completed")
        
        return result
        
    except Exception as e:
        logger.error(f"Error running synthesize_node: {e}", exc_info=True)
        raise


def main():
    """Main function."""
    # Default to the most recent output file if no argument provided
    if len(sys.argv) > 1:
        extract_output_file = sys.argv[1]
    else:
        # Try to find rank_result.json (from rank_node)
        output_dir = project_root / "output"
        rank_output_file = output_dir / "rank_result.json"
        
        if rank_output_file.exists():
            extract_output_file = rank_output_file
            logger.info(f"Using rank_result.json: {extract_output_file}")
        else:
            logger.error("rank_result.json not found. Please run rank_node first or provide a path to rank_result.json")
            logger.error("Usage: python tests/synthesize.py [path_to_rank_result.json] [output_file.json]")
            sys.exit(1)
    
    # Optional output file
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_synthesize_node(str(extract_output_file), output_file)


if __name__ == "__main__":
    main()
