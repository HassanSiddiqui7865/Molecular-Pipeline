"""
Test script for PDF export using WeasyPrint (HTML/CSS to PDF)
"""
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Use xhtml2pdf (pisa) - pure Python, no system dependencies
try:
    from export_pdf import export_to_pdf
    PDF_LIBRARY = "xhtml2pdf (pisa)"
except ImportError as e:
    print(f"Error: xhtml2pdf (pisa) not available: {e}")
    print("Please install: pip install xhtml2pdf")
    print("This is a pure Python library with no system dependencies.")
    sys.exit(1)

def test_export_pdf(test_json_path: str = None):
    """
    Test the PDF export functionality.
    
    Args:
        test_json_path: Path to test JSON file. If not provided, uses default test file.
    """
    if not test_json_path:
        # Always use enrichment_result.json as default
        test_json_path = project_root / "output" / "enrichment_result.json"
    
    test_json_path = Path(test_json_path)
    
    if not test_json_path.exists():
        print(f"Test file not found: {test_json_path}")
        print("Please provide a valid JSON file path or ensure the default test file exists.")
        return
    
    print(f"Loading test data from: {test_json_path}")
    
    # Load test data directly from enrichment_result.json
    with open(test_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure icd_transformation exists (should be added statically to the JSON file)
    if 'icd_transformation' not in data:
        # Fallback: create from severity_codes if available
        severity_codes = data.get('input_parameters', {}).get('severity_codes', [])
        if severity_codes:
            data['icd_transformation'] = {
                'original_codes': severity_codes if isinstance(severity_codes, list) else [severity_codes],
                'code_names': [
                    {'code': code, 'name': f'ICD-10 Code {code}'} for code in (severity_codes if isinstance(severity_codes, list) else [severity_codes])
                ],
                'severity_codes_transformed': ', '.join(str(c) for c in (severity_codes if isinstance(severity_codes, list) else [severity_codes]))
            }
        else:
            print("Warning: No icd_transformation found in data and no severity_codes to create fallback")
    
    print(f"Generating PDF using {PDF_LIBRARY}...")
    
    try:
        # Generate PDF
        pdf_path = export_to_pdf(data)
        print(f"✓ PDF generated successfully: {pdf_path}")
        print(f"  Library used: {PDF_LIBRARY}")
        return pdf_path
    except ImportError as e:
        print(f"✗ Import error: {e}")
        if "weasyprint" in str(e).lower():
            print("  Please install WeasyPrint: pip install weasyprint")
        elif "reportlab" in str(e).lower():
            print("  Please install ReportLab: pip install reportlab")
        import traceback
        traceback.print_exc()
        return None
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
        help="Path to input JSON file (default: output/enrichment_result.json)"
    )
    
    args = parser.parse_args()
    
    test_export_pdf(args.input)
