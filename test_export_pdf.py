"""
Test script for PDF export using Platypus (ReportLab)
"""
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from export_pdf import export_to_pdf

def test_export_pdf(test_json_path: str = None):
    """
    Test the PDF export functionality.
    
    Args:
        test_json_path: Path to test JSON file. If not provided, uses default test file.
    """
    if not test_json_path:
        # Use enrichment test output as default
        test_json_path = project_root / "output" / "enrichment_test_output.json"
    
    test_json_path = Path(test_json_path)
    
    if not test_json_path.exists():
        print(f"Test file not found: {test_json_path}")
        print("Please provide a valid JSON file path or ensure the default test file exists.")
        return
    
    print(f"Loading test data from: {test_json_path}")
    
    # Load test data
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure input_parameters exist (add sample if missing)
    if 'input_parameters' not in data:
        data['input_parameters'] = {}
    
    # Add sample if not present (for testing)
    if 'sample' not in data['input_parameters']:
        data['input_parameters']['sample'] = 'Blood'
    
    # Add medical input parameters from main.py for testing
    input_params = data['input_parameters']
    if 'pathogen_name' not in input_params:
        input_params['pathogen_name'] = 'Staphylococcus aureus'
    if 'resistant_gene' not in input_params:
        input_params['resistant_gene'] = 'mecA'
    if 'pathogen_count' not in input_params:
        input_params['pathogen_count'] = '10^6 CFU/ML'
    if 'severity_codes' not in input_params:
        input_params['severity_codes'] = 'A41.2, B95.6'
    if 'age' not in input_params:
        input_params['age'] = 32
    if 'gender' not in input_params:
        input_params['gender'] = 'Male'
    
    # Add other fields that might be missing for testing (all set to "-" except sample)
    if 'facility_name' not in input_params:
        input_params['facility_name'] = '-'
    if 'provider' not in input_params:
        input_params['provider'] = '-'
    if 'phone' not in input_params:
        input_params['phone'] = '-'
    if 'patient_name' not in input_params:
        input_params['patient_name'] = '-'
    if 'patient_dob' not in input_params:
        input_params['patient_dob'] = '-'
    if 'patient_gender' not in input_params:
        input_params['patient_gender'] = '-'
    if 'drug_allergies' not in input_params:
        input_params['drug_allergies'] = '-'
    if 'lab_accession' not in input_params:
        input_params['lab_accession'] = '-'
    if 'date_collected' not in input_params:
        input_params['date_collected'] = '-'
    if 'date_received' not in input_params:
        input_params['date_received'] = '-'
    if 'date_reported' not in input_params:
        input_params['date_reported'] = '-'
    if 'specimen_type' not in input_params:
        input_params['specimen_type'] = '-'
    if 'specimen_site' not in input_params:
        input_params['specimen_site'] = '-'
    
    # Add ICD transformation data for testing (with code names)
    if 'icd_transformation' not in data:
        data['icd_transformation'] = {
            'original_codes': ['A41.2', 'B95.6'],
            'code_names': [
                {'code': 'A41.2', 'name': 'Sepsis, unspecified organism'},
                {'code': 'B95.6', 'name': 'Staphylococcus aureus as the cause of diseases classified elsewhere'}
            ],
            'severity_codes_transformed': 'A41.2 (Sepsis, unspecified organism), B95.6 (Staphylococcus aureus as the cause of diseases classified elsewhere)'
        }
    
    print("Generating PDF...")
    
    try:
        # Generate PDF
        pdf_path = export_to_pdf(data)
        print(f"✓ PDF generated successfully: {pdf_path}")
        return pdf_path
    except Exception as e:
        print(f"✗ Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PDF export functionality")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file (default: output/enrichment_test_output.json)"
    )
    
    args = parser.parse_args()
    
    test_export_pdf(args.input)
