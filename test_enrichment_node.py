"""
Test script for enrichment_node.
Tests the enrichment node with a real output JSON file.
"""
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from nodes.enrichment_node import enrichment_node
from nodes.icd_transform_node import icd_transform_node

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_data(json_path: str) -> dict:
    """
    Load test data from JSON file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary with the test data
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded test data from {json_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
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


def main():
    """Main test function."""
    # Path to test JSON file
    test_json_path = Path(__file__).parent / "output" / "pathogen_info_output_20251216_132344.json"
    
    if not test_json_path.exists():
        logger.error(f"Test file not found: {test_json_path}")
        logger.info("Please provide the path to the JSON file as an argument")
        if len(sys.argv) > 1:
            test_json_path = Path(sys.argv[1])
        else:
            sys.exit(1)
    
    logger.info(f"Testing enrichment_node with: {test_json_path}")
    
    # Load test data
    json_data = load_test_data(str(test_json_path))
    
    # Create test state
    initial_state = create_test_state(json_data)
    
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
    except Exception as e:
        logger.error(f"Error running enrichment_node: {e}", exc_info=True)
        sys.exit(1)
    
    # Count antibiotics after enrichment
    therapy_plan_after = updated_state.get('result', {}).get('antibiotic_therapy_plan', {})
    counts_after = count_antibiotics_with_null_fields(therapy_plan_after)
    
    logger.info("\n=== AFTER ENRICHMENT ===")
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
        logger.info(f"{category}: {counts_after[category]['with_null']}/{counts_after[category]['total']} have null fields")
        logger.info(f"  Removed: {counts_before[category]['total'] - counts_after[category]['total']} antibiotics")
    
    # Save results
    output_path = Path(__file__).parent / "output" / "enrichment_test_output.json"
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_state.get('result', {}), f, indent=2, ensure_ascii=False)
        logger.info(f"\n=== RESULTS SAVED ===")
        logger.info(f"Output saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Summary
    logger.info("\n=== SUMMARY ===")
    total_before = sum(counts_before[cat]['total'] for cat in counts_before)
    total_after = sum(counts_after[cat]['total'] for cat in counts_after)
    total_removed = total_before - total_after
    
    logger.info(f"Total antibiotics before: {total_before}")
    logger.info(f"Total antibiotics after: {total_after}")
    logger.info(f"Total removed (no valid drugs.com URL): {total_removed}")
    
    if total_removed > 0:
        logger.warning(f"\n⚠️  {total_removed} antibiotics were removed because no valid drugs.com dosage URL was found")
    else:
        logger.info("\n✓ All antibiotics had valid drugs.com URLs or no enrichment was needed")


if __name__ == "__main__":
    main()

