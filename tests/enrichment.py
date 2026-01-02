"""
Test script for enrichment_node.
Loads saved synthesize_node output and tests enrichment_node without running the whole pipeline.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from nodes.enrichment_node import enrichment_node
from nodes.icd_transform_node import icd_transform_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_synthesize_output(file_path: str) -> Dict[str, Any]:
    """
    Load synthesize_node output from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing synthesize_node output
        
    Returns:
        Dictionary with result and input_parameters
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract required fields
        result = data.get('result', {})
        input_parameters = data.get('input_parameters', {})
        
        if not result:
            logger.warning(f"No result found in {file_path}")
        
        logger.info(f"Loaded synthesize output from {file_path}")
        logger.info(f"Input parameters: {input_parameters}")
        
        return {
            'result': result,
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


def create_test_state(json_data: dict) -> dict:
    """
    Create a test state dictionary from JSON data.
    The enrichment_node expects state with 'result', 'input_parameters', and 'icd_transformation' fields.
    
    Args:
        json_data: Loaded JSON data
        
    Returns:
        State dictionary for enrichment_node
    """
    # Extract result from the JSON (which has input_parameters, extraction_date, result)
    result = json_data.get('result', {})
    input_parameters = json_data.get('input_parameters', {})
    
    # Create initial state for icd_transform_node
    initial_state = {
        'input_parameters': input_parameters
    }
    
    # Run icd_transform_node to get transformed ICD codes
    logger.info("Running icd_transform_node to get transformed ICD codes...")
    try:
        icd_state = icd_transform_node(initial_state)
        icd_transformation = icd_state.get('icd_transformation', {})
        logger.info(f"ICD transformation result: {icd_transformation.get('severity_codes_transformed', 'N/A')}")
    except Exception as e:
        logger.warning(f"Error running icd_transform_node: {e}. Using original codes.")
        # Fallback: create a basic transformation structure
        severity_codes = input_parameters.get('severity_codes', '')
        if severity_codes:
            codes = [code.strip().upper() for code in severity_codes.split(',') if code.strip()]
            icd_transformation = {
                'original_codes': codes,
                'code_names': [{'code': code, 'name': code} for code in codes],
                'severity_codes_transformed': severity_codes
            }
        else:
            icd_transformation = {
                'original_codes': [],
                'code_names': [],
                'severity_codes_transformed': ''
            }
    
    state = {
        'result': result,
        'input_parameters': input_parameters,
        'icd_transformation': icd_transformation
    }
    
    logger.info(f"Created test state with {len(result.get('antibiotic_therapy_plan', {}).get('first_choice', []))} first_choice, "
                f"{len(result.get('antibiotic_therapy_plan', {}).get('second_choice', []))} second_choice, "
                f"{len(result.get('antibiotic_therapy_plan', {}).get('alternative_antibiotic', []))} alternative antibiotics")
    logger.info(f"ICD codes: {icd_transformation.get('severity_codes_transformed', 'N/A')}")
    
    return state


def count_antibiotics_with_null_fields(therapy_plan: dict) -> dict:
    """
    Count antibiotics that have null fields.
    
    Args:
        therapy_plan: Antibiotic therapy plan dictionary
        
    Returns:
        Dictionary with counts per category
    """
    counts = {
        'first_choice': {'total': 0, 'with_null': 0},
        'second_choice': {'total': 0, 'with_null': 0},
        'alternative_antibiotic': {'total': 0, 'with_null': 0}
    }
    
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
        antibiotics = therapy_plan.get(category, [])
        if isinstance(antibiotics, list):
            counts[category]['total'] = len(antibiotics)
            for ab in antibiotics:
                dose_duration = ab.get('dose_duration')
                route_of_administration = ab.get('route_of_administration')
                general_considerations = ab.get('general_considerations')
                
                if dose_duration is None or route_of_administration is None or general_considerations is None:
                    counts[category]['with_null'] += 1
    
    return counts


def test_enrichment_node(synthesize_output_file: str):
    """
    Test enrichment_node with saved synthesize_node output.
    
    Args:
        synthesize_output_file: Path to JSON file containing synthesize_node output
    """
    # Load synthesize_node output
    synthesize_data = load_synthesize_output(synthesize_output_file)
    
    # Create test state
    initial_state = create_test_state(synthesize_data)
    
    # Count antibiotics with null fields before enrichment
    therapy_plan_before = initial_state['result'].get('antibiotic_therapy_plan', {})
    counts_before = count_antibiotics_with_null_fields(therapy_plan_before)
    
    logger.info("\n=== BEFORE ENRICHMENT ===")
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
        logger.info(f"{category}: {counts_before[category]['with_null']}/{counts_before[category]['total']} have null fields")
    
    # List antibiotics that will be processed
    logger.info("\n=== ANTIBIOTICS TO PROCESS ===")
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
        antibiotics = therapy_plan_before.get(category, [])
        if isinstance(antibiotics, list):
            for ab in antibiotics:
                medical_name = ab.get('medical_name', 'unknown')
                dose_duration = ab.get('dose_duration')
                route_of_administration = ab.get('route_of_administration')
                general_considerations = ab.get('general_considerations')
                
                if dose_duration is None or route_of_administration is None or general_considerations is None:
                    null_fields = []
                    if dose_duration is None:
                        null_fields.append('dose_duration')
                    if route_of_administration is None:
                        null_fields.append('route_of_administration')
                    if general_considerations is None:
                        null_fields.append('general_considerations')
                    logger.info(f"  {category}: {medical_name} (missing: {', '.join(null_fields)})")
    
    # Run enrichment node
    logger.info("\n=== RUNNING ENRICHMENT NODE ===")
    try:
        updated_state = enrichment_node(initial_state)
        
        # Extract result
        enriched_result = updated_state.get('result', {})
        
        if enriched_result:
            therapy_plan = enriched_result.get('antibiotic_therapy_plan', {})
            first_choice = therapy_plan.get('first_choice', [])
            second_choice = therapy_plan.get('second_choice', [])
            alternative = therapy_plan.get('alternative_antibiotic', [])
            
            logger.info(
                f"Result: {len(first_choice)} first_choice, "
                f"{len(second_choice)} second_choice, "
                f"{len(alternative)} alternative"
            )
        else:
            logger.warning("No result returned from enrichment_node")
        
        logger.info("Test completed")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error running enrichment_node: {e}", exc_info=True)
        raise


def main():
    """Main function."""
    # Default to the most recent output file if no argument provided
    if len(sys.argv) > 1:
        synthesize_output_file = sys.argv[1]
    else:
        # Try to find synthesize_result.json (from synthesize_node)
        output_dir = project_root / "output"
        synthesize_output_file_path = output_dir / "synthesize_result.json"
        
        if synthesize_output_file_path.exists():
            synthesize_output_file = synthesize_output_file_path
            logger.info(f"Using synthesize_result.json: {synthesize_output_file}")
        else:
            logger.error("synthesize_result.json not found. Please run synthesize_node first or provide a path to synthesize_result.json")
            logger.error("Usage: python tests/enrichment.py [path_to_synthesize_result.json]")
            sys.exit(1)
    
    test_enrichment_node(str(synthesize_output_file))


if __name__ == "__main__":
    main()

