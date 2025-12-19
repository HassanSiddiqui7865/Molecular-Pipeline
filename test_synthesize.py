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
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from nodes.synthesize_node import synthesize_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
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
    Test synthesize_node with saved extract_node output.
    
    Args:
        extract_output_file: Path to JSON file containing extract_node output
        output_file: Optional path to save synthesize_node output (default: test_synthesize_output.json)
    """
    logger.info("=" * 60)
    logger.info("Testing synthesize_node")
    logger.info("=" * 60)
    
    # Load rank_node output
    extract_data = load_extract_output(extract_output_file)
    
    # Create state dictionary for synthesize_node
    state = {
        'source_results': extract_data['source_results'],
        'input_parameters': extract_data['input_parameters']
    }
    
    # Log summary of input
    source_results = state['source_results']
    logger.info(f"\nInput Summary:")
    logger.info(f"  - Number of sources: {len(source_results)}")
    
    # Count antibiotics per source
    total_antibiotics = 0
    for idx, source in enumerate(source_results, 1):
        therapy_plan = source.get('antibiotic_therapy_plan', {})
        first = len(therapy_plan.get('first_choice', []))
        second = len(therapy_plan.get('second_choice', []))
        alt = len(therapy_plan.get('alternative_antibiotic', []))
        total = first + second + alt
        total_antibiotics += total
        logger.info(f"  - Source {idx}: {first} first_choice, {second} second_choice, {alt} alternative ({total} total)")
    
    logger.info(f"  - Total antibiotics across all sources: {total_antibiotics}")
    
    # Run synthesize_node
    logger.info("\nRunning synthesize_node...")
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
            overall_confidence = unified_result.get('overall_confidence', 0.0)
            
            logger.info("\n" + "=" * 60)
            logger.info("Synthesize Results:")
            logger.info("=" * 60)
            logger.info(f"  - First choice antibiotics: {len(first_choice)}")
            logger.info(f"  - Second choice antibiotics: {len(second_choice)}")
            logger.info(f"  - Alternative antibiotics: {len(alternative)}")
            logger.info(f"  - Resistance genes: {len(resistance_genes)}")
            # Show first few antibiotics from each category
            if first_choice:
                logger.info("\n  First Choice Antibiotics:")
                for ab in first_choice[:3]:
                    logger.info(f"    - {ab.get('medical_name', 'N/A')}")
            
            if second_choice:
                logger.info("\n  Second Choice Antibiotics:")
                for ab in second_choice[:3]:
                    logger.info(f"    - {ab.get('medical_name', 'N/A')}")
            
            if alternative:
                logger.info("\n  Alternative Antibiotics:")
                for ab in alternative[:3]:
                    logger.info(f"    - {ab.get('medical_name', 'N/A')}")
        else:
            logger.warning("No result returned from synthesize_node")
        
        # Save output if output_file is specified
        if output_file:
            output_data = {
                'input_parameters': state['input_parameters'],
                'source_results': state['source_results'],
                'result': unified_result
            }
            save_synthesize_output(output_data, output_file)
        else:
            # Default output filename
            output_dir = project_root / "output"
            output_file = output_dir / "test_synthesize_output.json"
            output_data = {
                'input_parameters': state['input_parameters'],
                'source_results': state['source_results'],
                'result': unified_result
            }
            save_synthesize_output(output_data, str(output_file))
        
        logger.info("\n" + "=" * 60)
        logger.info("Test completed successfully!")
        logger.info("=" * 60)
        
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
        # First check for test_rank_output.json (saved by rank_node)
        output_dir = project_root / "output"
        rank_output_file = output_dir / "test_rank_output.json"
        
        if rank_output_file.exists():
            extract_output_file = rank_output_file
            logger.info(f"Using test_rank_output.json: {extract_output_file}")
        else:
            # Fall back to finding the most recent output file
            output_files = list(output_dir.glob("pathogen_info_output_*.json"))
            if output_files:
                # Sort by modification time, most recent first
                extract_output_file = max(output_files, key=lambda p: p.stat().st_mtime)
                logger.info(f"Using most recent output file: {extract_output_file}")
            else:
                logger.error("No output files found. Please provide a path to a rank_node output file.")
                logger.error("Usage: python test_synthesize.py [path_to_rank_output.json] [output_file.json]")
                sys.exit(1)
    
    # Optional output file
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_synthesize_node(str(extract_output_file), output_file)


if __name__ == "__main__":
    main()
