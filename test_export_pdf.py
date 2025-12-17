"""
Test script for PDF export functionality.
"""
import json
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from export_pdf import export_to_pdf

def main():
    # Load the latest output JSON file
    output_dir = project_root / "output"
    
    # Find the most recent output file
    json_files = list(output_dir.glob("pathogen_info_output_*.json"))
    if not json_files:
        print("No output JSON files found in output/ directory")
        return
    
    # Get the most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading: {latest_file}")
    
    # Load JSON data
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Export to PDF
    try:
        pdf_path = export_to_pdf(data)
        print(f"PDF exported to: {pdf_path}")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install xhtml2pdf: pip install xhtml2pdf")
    except Exception as e:
        print(f"Error exporting PDF: {e}")

if __name__ == "__main__":
    main()

