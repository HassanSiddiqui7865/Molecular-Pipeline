"""
Test script for rank_node.py
Loads saved extraction_result.json and tests rank_node.
"""
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nodes.rank_node import rank_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_extraction_result(file_path: str) -> Dict[str, Any]:
    """Load extraction_result.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        source_results = data.get('source_results', [])
        input_parameters = data.get('input_parameters', {})
        
        logger.info(f"Loaded {len(source_results)} source results from {file_path}")
        
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


def main():
    """Main function."""
    # Default to extraction_result.json if no argument provided
    if len(sys.argv) > 1:
        extraction_result_file = sys.argv[1]
    else:
        output_dir = project_root / "output"
        extraction_result_file = output_dir / "extraction_result.json"
        
        if not extraction_result_file.exists():
            logger.error(f"File not found: {extraction_result_file}")
            logger.error("Usage: python tests/rank.py [path_to_extraction_result.json]")
            sys.exit(1)
    
    # Load extraction result
    extraction_data = load_extraction_result(str(extraction_result_file))
    
    # Create state dictionary for rank_node
    state = {
        'source_results': extraction_data['source_results'],
        'input_parameters': extraction_data['input_parameters']
    }
    
    # Run rank_node
    try:
        result = rank_node(state)
        
        source_results = result.get('source_results', [])
        if source_results:
            # Count unique antibiotics across all sources
            all_antibiotics = set()
            category_totals = {'first_choice': 0, 'second_choice': 0, 'alternative_antibiotic': 0}
            
            for source_result in source_results:
                therapy_plan = source_result.get('antibiotic_therapy_plan', {})
                for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                    antibiotics = therapy_plan.get(category, [])
                    for ab in antibiotics:
                        if isinstance(ab, dict):
                            name = ab.get('medical_name', '').strip()
                            if name:
                                all_antibiotics.add(name)
                                category_totals[category] += 1
            
            logger.info(
                f"Result: {len(all_antibiotics)} unique antibiotics | "
                f"{category_totals['first_choice']} first_choice, "
                f"{category_totals['second_choice']} second_choice, "
                f"{category_totals['alternative_antibiotic']} alternative"
            )
        
        logger.info("Test completed")
        
    except Exception as e:
        logger.error(f"Error running rank_node: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
