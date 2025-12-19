"""
PDF Export Module using Platypus (ReportLab) - Dynamic report generation
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. Install with: pip install reportlab")


def _format_text_field(text: Optional[str]) -> str:
    """Format any text field to handle None values."""
    if text is None:
        return "N/A"
    return str(text).strip()


def _create_input_section(data: Dict[str, Any], styles) -> list:
    """
    Create the input section of the report.
    Based on the image: blue header with sample type, three columns of information.
    """
    elements = []
    
    input_params = data.get('input_parameters', {})
    
    # Get sample type (default to "WOUND" if not provided, but user wants to use sample from input)
    sample = input_params.get('sample', 'WOUND')
    if not sample or sample.upper() == 'N/A':
        sample = 'WOUND'
    
    # Header with blue background
    header_style = ParagraphStyle(
        'CustomHeader',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=white,
        backColor=HexColor('#4472C4'),  # Blue color
        spaceAfter=12,
        alignment=TA_CENTER,
        leading=20
    )
    
    elements.append(Paragraph(sample.upper(), header_style))
    elements.append(Spacer(1, 0.05 * inch))  # Minimal spacing after header
    
    # Prepare data for three columns
    # Column 1: Facility/Provider (extract from input_params or use defaults)
    facility_name = input_params.get('facility_name', 'N/A')
    provider = input_params.get('provider', 'N/A')
    phone = input_params.get('phone', 'N/A')
    
    # Column 2: Patient Demographics
    patient_name = input_params.get('patient_name', 'N/A')
    patient_dob = input_params.get('patient_dob', 'N/A')
    patient_gender = input_params.get('patient_gender', 'N/A')
    drug_allergies = input_params.get('drug_allergies', 'NO ALLERGIES PROVIDED')
    # Format drug allergies - if it's a list or has special formatting
    if drug_allergies and drug_allergies.upper() not in ['NO ALLERGIES PROVIDED', 'N/A', 'NONE']:
        drug_allergies = drug_allergies
    else:
        drug_allergies = 'NO ALLERGIES PROVIDED'
    
    # Column 3: Lab/Specimen Information
    lab_accession = input_params.get('lab_accession', 'N/A')
    date_collected = input_params.get('date_collected', 'N/A')
    date_received = input_params.get('date_received', 'N/A')
    date_reported = input_params.get('date_reported', 'N/A')
    specimen_type = input_params.get('specimen_type', sample)  # Use sample if specimen_type not provided
    specimen_site = input_params.get('specimen_site', 'N/A')
    
    # Create table with 3 columns
    table_data = []
    
    # Column 1: Facility/Provider
    col1_data = [
        ['<b>Facility Name:</b>', facility_name],
        ['<b>Provider:</b>', provider],
        ['<b>Phone:</b>', phone]
    ]
    
    # Column 2: Patient Demographics
    col2_data = [
        ['<b>Patient Name:</b>', patient_name],
        ['<b>Patient DOB:</b>', patient_dob],
        ['<b>Patient Gender:</b>', patient_gender],
        ['<b>Drug Allergies:</b>', drug_allergies]
    ]
    
    # Column 3: Lab/Specimen
    col3_data = [
        ['<b>Lab Accession #:</b>', lab_accession],
        ['<b>Date Collected:</b>', date_collected],
        ['<b>Date Received:</b>', date_received],
        ['<b>Date Reported:</b>', date_reported],
        ['<b>Specimen Type:</b>', specimen_type],
        ['<b>Specimen Site:</b>', specimen_site]
    ]
    
    # Find max rows to align columns
    max_rows = max(len(col1_data), len(col2_data), len(col3_data))
    
    # Pad columns to same length
    while len(col1_data) < max_rows:
        col1_data.append(['', ''])
    while len(col2_data) < max_rows:
        col2_data.append(['', ''])
    while len(col3_data) < max_rows:
        col3_data.append(['', ''])
    
    # Create compact paragraph style for table cells - very tight spacing
    compact_style = ParagraphStyle(
        'Compact',
        parent=styles['Normal'],
        fontSize=8,
        leading=9,  # Very tight line spacing for compact look
        spaceBefore=0,
        spaceAfter=0
    )
    
    # Build table rows
    for i in range(max_rows):
        row = [
            Paragraph(col1_data[i][0], compact_style),
            Paragraph(_format_text_field(col1_data[i][1]), compact_style),
            Paragraph(col2_data[i][0], compact_style),
            Paragraph(_format_text_field(col2_data[i][1]), compact_style),
            Paragraph(col3_data[i][0], compact_style),
            Paragraph(_format_text_field(col3_data[i][1]), compact_style)
        ]
        table_data.append(row)
    
    # Create table - adjust column widths to fit page (letter size is 8.5 inches, minus margins = ~7.7 inches usable)
    # Total: 6 columns (3 label columns + 3 value columns)
    # Use proportional widths to fit: label columns ~0.9 inch, value columns ~1.3 inch
    # Total: 0.9*3 + 1.3*3 = 2.7 + 3.9 = 6.6 inches (fits comfortably)
    table = Table(table_data, colWidths=[0.9*inch, 1.3*inch, 0.9*inch, 1.3*inch, 0.9*inch, 1.3*inch])
    
    # Style the table - extremely minimal padding to match reference image
    table_style = TableStyle([
        # Header row styling (if needed)
        ('BACKGROUND', (0, 0), (-1, -1), white),
        ('TEXTCOLOR', (0, 0), (-1, -1), black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),  # Minimal bottom padding
        ('TOPPADDING', (0, 0), (-1, -1), 1),  # Minimal top padding
        ('LEFTPADDING', (0, 0), (-1, -1), 2),  # Minimal left padding
        ('RIGHTPADDING', (0, 0), (-1, -1), 2),  # Minimal right padding
        # Highlight label columns (0, 2, 4) with light blue background
        ('BACKGROUND', (0, 0), (0, -1), HexColor('#E7F3FF')),
        ('BACKGROUND', (2, 0), (2, -1), HexColor('#E7F3FF')),
        ('BACKGROUND', (4, 0), (4, -1), HexColor('#E7F3FF')),
    ])
    
    # Add red color for drug allergies if it says "NO ALLERGIES PROVIDED"
    # Drug allergies is in column 2 (Patient Demographics), row 3 (index 3), value column is index 3
    if drug_allergies.upper() == 'NO ALLERGIES PROVIDED':
        # Column 3 (index 3) contains the drug allergies value, row 3 (index 3)
        table_style.add('TEXTCOLOR', (3, 3), (3, 3), HexColor('#FF0000'))
    
    table.setStyle(table_style)
    
    elements.append(table)
    elements.append(Spacer(1, 0.1 * inch))  # Minimal spacing after table
    
    return elements


def export_to_pdf(data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Export data to PDF using Platypus (ReportLab).
    
    Args:
        data: Dictionary containing input_parameters and result
        output_path: Optional path for output file
        
    Returns:
        Path to generated PDF file
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is not installed. Install with: pip install reportlab")
    
    # Generate output path if not provided
    if not output_path:
        from config import get_output_config
        output_config = get_output_config()
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = str(output_dir / f"report_{timestamp}.pdf")
    
    # Create PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.4*inch,
        leftMargin=0.4*inch,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Build story (list of flowables)
    story = []
    
    # Add input section
    input_elements = _create_input_section(data, styles)
    story.extend(input_elements)
    
    # Build PDF
    doc.build(story)
    
    logger.info(f"PDF report exported to: {output_path}")
    return output_path
