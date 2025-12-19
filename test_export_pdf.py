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
    
    # Add other fields that might be missing for testing
    input_params = data['input_parameters']
    if 'facility_name' not in input_params:
        input_params['facility_name'] = 'Specialists in Dermatology'
    if 'provider' not in input_params:
        input_params['provider'] = 'Susannah Andrews PA-C'
    if 'phone' not in input_params:
        input_params['phone'] = '(713) 345-1220'
    if 'patient_name' not in input_params:
        input_params['patient_name'] = 'N/A'  # Redacted in image
    if 'patient_dob' not in input_params:
        input_params['patient_dob'] = '08/15/1952'
    if 'patient_gender' not in input_params:
        input_params['patient_gender'] = 'Male'
    if 'drug_allergies' not in input_params:
        input_params['drug_allergies'] = 'NO ALLERGIES PROVIDED'
    if 'lab_accession' not in input_params:
        input_params['lab_accession'] = 'EDI2510170004'
    if 'date_collected' not in input_params:
        input_params['date_collected'] = '10/16/2025'
    if 'date_received' not in input_params:
        input_params['date_received'] = '10/17/2025'
    if 'date_reported' not in input_params:
        input_params['date_reported'] = '10/17/2025'
    if 'specimen_type' not in input_params:
        input_params['specimen_type'] = input_params.get('sample', 'Blood')
    if 'specimen_site' not in input_params:
        input_params['specimen_site'] = 'Left lower leg'
    
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
