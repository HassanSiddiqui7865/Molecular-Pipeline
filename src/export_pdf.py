"""
PDF Export Module using Platypus (ReportLab) - Exact design matching reference image
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch, cm
    from reportlab.lib.colors import HexColor, white, black, lightgrey
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, Flowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available. Install with: pip install reportlab")


def _format_text_field(text: Optional[str]) -> str:
    """Format any text field to handle None values."""
    if text is None:
        return "-"
    text_str = str(text).strip()
    return text_str if text_str else "-"


def _create_medication_section(data: Dict[str, Any], styles) -> list:
    """
    Create medication section exactly matching the reference image.
    Extracts first_choice, second_choice, and alternative_antibiotic from data.
    """
    elements = []
    
    # Add a little space before medication section
    elements.append(Spacer(1, 0.2 * inch))
    
    # Create medication header with blue background - FIXED: Blue header
    med_header_style = ParagraphStyle(
        'MedHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=white,
        alignment=TA_CENTER,
        spaceAfter=6,
        spaceBefore=0
    )
    
    # Create blue header for "MEDICATION RECOMMENDATIONS"
    med_header_table = Table([[Paragraph("MEDICATION RECOMMENDATIONS", med_header_style)]], 
                             colWidths=[7.5 * inch])
    med_header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#4472C4')),  # Same blue as main header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    elements.append(med_header_table)
    elements.append(Spacer(1, 0.1 * inch))
    
    # Import fix_text_encoding for handling Unicode issues
    from utils import fix_text_encoding
    
    # Get antibiotic therapy plan from result (handle both structures)
    result = data.get('result', {})
    therapy_plan = result.get('antibiotic_therapy_plan', {})
    
    # If not found in result, try direct access
    if not therapy_plan:
        therapy_plan = data.get('antibiotic_therapy_plan', {})
    
    # Extract antibiotics from each category
    first_choice_list = therapy_plan.get('first_choice', [])
    second_choice_list = therapy_plan.get('second_choice', [])
    alternative_list = therapy_plan.get('alternative_antibiotic', [])
    
    # Debug logging
    logger.info(f"Found {len(first_choice_list)} first_choice, {len(second_choice_list)} second_choice, {len(alternative_list)} alternative antibiotics")
    
    # Convert antibiotic data to medication format
    medications = []
    
    # Add first choice antibiotics
    for i, ab in enumerate(first_choice_list):
        if ab.get('medical_name'):
            med = {
                'name': ab.get('medical_name', ''),
                'route': ab.get('route_of_administration', ''),
                'dose': ab.get('dose_duration', ''),
                'coverage': ab.get('coverage_for', ''),
                'renal_adjustment': ab.get('renal_adjustment', ''),
                'considerations': ab.get('general_considerations', ''),
                'is_or_option': i > 0,  # "OR" after first item within category
                'line_type': 'First' if i == 0 else '',  # Tag only on first medication
                'show_tag': i == 0,  # Show tag only for first medication in category
                'category': 'first_choice'  # Track category for spacing
            }
            medications.append(med)
    
    # Add second choice antibiotics
    for i, ab in enumerate(second_choice_list):
        if ab.get('medical_name'):
            med = {
                'name': ab.get('medical_name', ''),
                'route': ab.get('route_of_administration', ''),
                'dose': ab.get('dose_duration', ''),
                'coverage': ab.get('coverage_for', ''),
                'renal_adjustment': ab.get('renal_adjustment', ''),
                'considerations': ab.get('general_considerations', ''),
                'is_or_option': i > 0,  # "OR" after first item within category
                'line_type': 'Second' if i == 0 else '',  # Tag only on first medication
                'show_tag': i == 0,  # Show tag only for first medication in category
                'category': 'second_choice'
            }
            medications.append(med)
    
    # Add alternative antibiotics
    for i, ab in enumerate(alternative_list):
        if ab.get('medical_name'):
            med = {
                'name': ab.get('medical_name', ''),
                'route': ab.get('route_of_administration', ''),
                'dose': ab.get('dose_duration', ''),
                'coverage': ab.get('coverage_for', ''),
                'renal_adjustment': ab.get('renal_adjustment', ''),
                'considerations': ab.get('general_considerations', ''),
                'is_or_option': i > 0,  # "OR" after first item within category
                'line_type': 'Alternate' if i == 0 else '',  # Tag only on first medication
                'show_tag': i == 0,  # Show tag only for first medication in category
                'category': 'alternative_antibiotic'
            }
            medications.append(med)
    
    # If no medications found, return empty (don't show section)
    if not medications:
        return elements
    
    # Styles for medication section
    med_table_header_style = ParagraphStyle(
        'MedTableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_CENTER,
        textColor=white,
        spaceBefore=0,
        spaceAfter=0
    )
    
    med_name_style = ParagraphStyle(
        'MedName',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    med_data_style = ParagraphStyle(
        'MedData',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    coverage_style = ParagraphStyle(
        'Coverage',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        textColor=HexColor('#FF0000'),  # Red text
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    considerations_style = ParagraphStyle(
        'Considerations',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    or_header_style = ParagraphStyle(
        'OrHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        spaceBefore=4,
        spaceAfter=4
    )
    
    # Style for the line type tag (First Line, Second Line, Alternate)
    line_tag_style = ParagraphStyle(
        'LineTag',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        textColor=white,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=0
    )
    
    # Create each medication table
    for i, med in enumerate(medications):
        # Add "OR" header if this medication is marked as needing "OR"
        if med.get('is_or_option', False):
            elements.append(Paragraph("OR", or_header_style))
            elements.append(Spacer(1, 0.05 * inch))
        
        # Get line type tag (First Line, Second Line, or Alternate)
        line_type = med.get('line_type', '')
        show_tag = med.get('show_tag', False)
        
        # Create medication header row
        if show_tag:
            # With tag column
            header_data = [
                '',  # Empty cell for tag column (will be styled separately)
                Paragraph("Medication", med_table_header_style),
                Paragraph("Route", med_table_header_style),
                Paragraph("Dose", med_table_header_style)
            ]
        else:
            # Without tag column - regular 3-column header
            header_data = [
                Paragraph("Medication", med_table_header_style),
                Paragraph("Route", med_table_header_style),
                Paragraph("Dose", med_table_header_style)
            ]
        
        # Create medication data row (using fix_text_encoding for Unicode issues)
        med_name = fix_text_encoding(med.get('name', ''))
        med_route = fix_text_encoding(med.get('route', ''))
        med_dose = fix_text_encoding(med.get('dose', ''))
        
        if show_tag:
            # Tag cell with line type
            tag_cell = Paragraph(line_type, line_tag_style)
            
            med_data_row = [
                tag_cell,  # Tag column
                Paragraph(_format_text_field(med_name), med_name_style),
                Paragraph(_format_text_field(med_route), med_data_style),
                Paragraph(_format_text_field(med_dose), med_data_style)
            ]
        else:
            # Regular 3-column row (no tag)
            med_data_row = [
                Paragraph(_format_text_field(med_name), med_name_style),
                Paragraph(_format_text_field(med_route), med_data_style),
                Paragraph(_format_text_field(med_dose), med_data_style)
            ]
        
        # Create coverage row
        coverage_text = med.get('coverage', '')
        if coverage_text:
            coverage_text = fix_text_encoding(coverage_text)
            if not coverage_text.startswith('Coverage For:'):
                coverage_text = 'Coverage For: ' + coverage_text
        else:
            coverage_text = ''
        
        if show_tag:
            # With tag column
            coverage_row = [
                '',  # Empty cell for tag column
                Paragraph(coverage_text, coverage_style) if coverage_text else '',
                '',  # Empty cell for Route column
                ''   # Empty cell for Dose column
            ]
        else:
            # Regular 3-column row (no tag)
            coverage_row = [
                Paragraph(coverage_text, coverage_style) if coverage_text else '',
                '',  # Empty cell for Route column
                ''   # Empty cell for Dose column
            ]
        
        # Build considerations text (separate from renal_adjustment)
        considerations_text = med.get('considerations', '')
        if considerations_text:
            considerations_text = fix_text_encoding(considerations_text)
            if not considerations_text.startswith('Considerations:'):
                considerations_text = 'Considerations: ' + considerations_text
        else:
            considerations_text = ''
        
        # Get renal_adjustment separately
        renal_adjustment = med.get('renal_adjustment', '')
        if renal_adjustment:
            renal_adjustment = fix_text_encoding(renal_adjustment)
            if not renal_adjustment.startswith('Renal Adjustment:'):
                renal_adjustment_text = 'Renal Adjustment: ' + renal_adjustment
            else:
                renal_adjustment_text = renal_adjustment
        else:
            renal_adjustment_text = ''
        
        # Create considerations row
        if show_tag:
            # With tag column
            considerations_row = [
                '',  # Empty cell for tag column
                Paragraph(considerations_text, considerations_style) if considerations_text else '',
                '',  # Empty cell for Route column
                ''   # Empty cell for Dose column
            ]
        else:
            # Regular 3-column row (no tag)
            considerations_row = [
                Paragraph(considerations_text, considerations_style) if considerations_text else '',
                '',  # Empty cell for Route column
                ''   # Empty cell for Dose column
            ]
        
        # Create renal_adjustment row (separate)
        if show_tag:
            # With tag column
            renal_row = [
                '',  # Empty cell for tag column
                Paragraph(renal_adjustment_text, considerations_style) if renal_adjustment_text else '',
                '',  # Empty cell for Route column
                ''   # Empty cell for Dose column
            ]
        else:
            # Regular 3-column row (no tag)
            renal_row = [
                Paragraph(renal_adjustment_text, considerations_style) if renal_adjustment_text else '',
                '',  # Empty cell for Route column
                ''   # Empty cell for Dose column
            ]
        
        # Build table data
        table_data = [header_data, med_data_row, coverage_row, considerations_row, renal_row]
        
        # Calculate column widths based on whether we have tag or not
        if show_tag:
          
            col_widths = [0.6 * inch, 3.3 * inch, 1.1 * inch, 2.5 * inch]
            num_columns = 4
        else:
           
            col_widths = [3.9 * inch, 1.1 * inch, 2.5 * inch]
            num_columns = 3
        
        # Create medication table
        med_table = Table(table_data, colWidths=col_widths, hAlign='LEFT')
        
        # Style the medication table
        med_table_style_list = []
        
        # Blue header for table - FIXED: Blue header for each medication table
        if show_tag:
            # With tag column
            med_table_style_list.extend([
                ('BACKGROUND', (1, 0), (3, 0), HexColor('#4472C4')),  # Blue header for data columns
            ])
        else:
            # Regular 3 columns
            med_table_style_list.extend([
                ('BACKGROUND', (0, 0), (2, 0), HexColor('#4472C4')),  # Blue header for all columns
            ])
        
        # Set alternating row backgrounds
        if show_tag:
            med_table_style_list.extend([
                ('BACKGROUND', (0, 1), (0, 1), HexColor('#367FA9')),  # Blue tag background
                ('BACKGROUND', (1, 1), (1, 1), HexColor('#F0F8FF')),  # Light blue for medication name
                ('BACKGROUND', (2, 1), (3, 1), white),  # White for route and dose
                ('BACKGROUND', (0, 2), (-1, 2), white),  # White for coverage
                ('BACKGROUND', (0, 3), (-1, 3), white),  # White for considerations
                ('BACKGROUND', (0, 4), (-1, 4), white),  # White for renal adjustment
            ])
        else:
            med_table_style_list.extend([
                ('BACKGROUND', (0, 1), (0, 1), HexColor('#F0F8FF')),  # Light blue for medication name
                ('BACKGROUND', (1, 1), (2, 1), white),  # White for route and dose
                ('BACKGROUND', (0, 2), (-1, 2), white),  # White for coverage
                ('BACKGROUND', (0, 3), (-1, 3), white),  # White for considerations
                ('BACKGROUND', (0, 4), (-1, 4), white),  # White for renal adjustment
            ])
        
        # Cell borders
        med_table_style_list.extend([
            ('BOX', (0, 0), (-1, -1), 0.5, black),  # Outer border
            ('INNERGRID', (0, 0), (-1, -1), 0.5, black),  # Inner grid
        ])
        
        # Merge cells for coverage, considerations, and renal_adjustment rows
        if show_tag:
            # With tag column - coverage/considerations/renal span columns 1-3
            med_table_style_list.extend([
                ('SPAN', (1, 2), (3, 2)),  # Coverage row spans medication, route, dose columns
                ('SPAN', (1, 3), (3, 3)),  # Considerations row spans medication, route, dose columns
                ('SPAN', (1, 4), (3, 4)),  # Renal adjustment row spans medication, route, dose columns
            ])
        else:
            # Regular 3 columns - coverage/considerations/renal span all columns
            med_table_style_list.extend([
                ('SPAN', (0, 2), (2, 2)),  # Coverage row spans all columns
                ('SPAN', (0, 3), (2, 3)),  # Considerations row spans all columns
                ('SPAN', (0, 4), (2, 4)),  # Renal adjustment row spans all columns
            ])
        
        # Text alignment
        if show_tag:
            med_table_style_list.extend([
                ('ALIGN', (1, 0), (3, 0), 'CENTER'),  # Header centered (skip tag column)
                ('ALIGN', (0, 1), (0, 1), 'CENTER'),  # Tag column centered
                ('ALIGN', (1, 1), (1, 1), 'LEFT'),    # Medication name left aligned
                ('ALIGN', (2, 1), (2, 1), 'LEFT'),    # Route left aligned
                ('ALIGN', (3, 1), (3, 1), 'LEFT'),    # Dose left aligned
                ('ALIGN', (1, 2), (1, 2), 'LEFT'),    # Coverage left aligned
                ('ALIGN', (1, 3), (1, 3), 'LEFT'),    # Considerations left aligned
                ('ALIGN', (1, 4), (1, 4), 'LEFT'),    # Renal adjustment left aligned
            ])
        else:
            med_table_style_list.extend([
                ('ALIGN', (0, 0), (2, 0), 'CENTER'),  # Header centered
                ('ALIGN', (0, 1), (0, 1), 'LEFT'),    # Medication name left aligned
                ('ALIGN', (1, 1), (1, 1), 'LEFT'),    # Route left aligned
                ('ALIGN', (2, 1), (2, 1), 'LEFT'),    # Dose left aligned
                ('ALIGN', (0, 2), (0, 2), 'LEFT'),    # Coverage left aligned
                ('ALIGN', (0, 3), (0, 3), 'LEFT'),    # Considerations left aligned
                ('ALIGN', (0, 4), (0, 4), 'LEFT'),    # Renal adjustment left aligned
            ])
        
        # Vertical alignment and padding
        med_table_style_list.extend([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('TOPPADDING', (0, 2), (-1, 2), 2),
            ('BOTTOMPADDING', (0, 2), (-1, 2), 2),
            ('TOPPADDING', (0, 3), (-1, 3), 2),
            ('BOTTOMPADDING', (0, 3), (-1, 3), 2),
            ('TOPPADDING', (0, 4), (-1, 4), 2),
            ('BOTTOMPADDING', (0, 4), (-1, 4), 2),
        ])
        
        # If showing tag, make it span the medication and coverage rows
        if show_tag:
            med_table_style_list.extend([
                ('VALIGN', (0, 1), (0, 2), 'MIDDLE'),  # Tag cell vertically centered across rows
            ])
        
        med_table_style = TableStyle(med_table_style_list)
        med_table.setStyle(med_table_style)
        
        # Add the medication table to elements
        elements.append(med_table)
        
        # Add small spacer between medication tables (except after last one)
        if i < len(medications) - 1:
            elements.append(Spacer(1, 0.1 * inch))
    
    return elements


def _create_gene_section(data: Dict[str, Any], styles) -> list:
    """
    Create gene section displaying resistance gene information from pharmacist_analysis_on_resistant_gene.
    """
    elements = []
    
    # Add a little space before gene section
    elements.append(Spacer(1, 0.2 * inch))
    
    # Create gene header with blue background (matching medication section theme)
    gene_header_text = Paragraph(
        "RESISTANCE GENE INFORMATION",
        ParagraphStyle(
            'GeneHeader',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            textColor=white,
            alignment=TA_CENTER,
            spaceAfter=0,
            spaceBefore=0
        )
    )
    
    # Create header table with blue background
    gene_header_table = Table([[gene_header_text]], colWidths=[7.5 * inch])
    gene_header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#4472C4')),  # Same blue as medication header
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    elements.append(gene_header_table)
    elements.append(Spacer(1, 0.1 * inch))
    
    # Get gene analysis data from pharmacist_analysis_on_resistant_gene
    # Check both top-level and result level
    gene_analysis_list = data.get('pharmacist_analysis_on_resistant_gene', [])
    if not gene_analysis_list:
        result = data.get('result', {})
        gene_analysis_list = result.get('pharmacist_analysis_on_resistant_gene', [])
    
    # Debug logging
    logger.info(f"Gene section - found {len(gene_analysis_list)} gene analysis entries")
    
    # If no gene analysis found, return empty section
    if not gene_analysis_list:
        logger.warning("Gene section - no pharmacist_analysis_on_resistant_gene found")
        return elements
    
    # Format gene data using utils
    from utils import fix_text_encoding
    
    # Create gene content styles
    gene_label_style = ParagraphStyle(
        'GeneLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    gene_value_style = ParagraphStyle(
        'GeneValue',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    gene_considerations_style = ParagraphStyle(
        'GeneConsiderations',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    # Process each gene analysis entry
    for gene_analysis in gene_analysis_list:
        gene_name = gene_analysis.get('detected_resistant_gene_name', '')
        medication_classes = gene_analysis.get('potential_medication_class_affected', '')
        considerations = gene_analysis.get('general_considerations', '')
        
        # Fix encoding issues
        gene_name = fix_text_encoding(gene_name) if gene_name else '-'
        medication_classes = fix_text_encoding(medication_classes) if medication_classes else '-'
        considerations = fix_text_encoding(considerations) if considerations else '-'
        
        # Create table data for this gene
        gene_table_data = [
            [
                Paragraph('<b>Detected Resistance Gene:</b>', gene_label_style),
                Paragraph(_format_text_field(gene_name), gene_value_style)
            ],
            [
                Paragraph('<b>Potential Medication Classes Affected:</b>', gene_label_style),
                Paragraph(_format_text_field(medication_classes), gene_value_style)
            ],
            [
                Paragraph('<b>General Considerations:</b>', gene_label_style),
                Paragraph(_format_text_field(considerations), gene_considerations_style)
            ]
        ]
        
        # Calculate column widths
        total_width = 7.5 * inch
        col_widths = [2.5 * inch, 5.0 * inch]
        
        # Create gene table
        gene_table = Table(gene_table_data, colWidths=col_widths, hAlign='LEFT')
        
        # Style the gene table with color theme
        gene_table_style = TableStyle([
            # Add borders for structure
            ('BOX', (0, 0), (-1, -1), 0.5, black),  # Outer border
            ('INNERGRID', (0, 0), (-1, -1), 0.5, black),  # Inner grid
            
            # Label column - light blue background (matching medication section theme)
            ('BACKGROUND', (0, 0), (0, -1), HexColor('#E7F3FF')),  # Light blue for labels
            
            # Value column - white background
            ('BACKGROUND', (1, 0), (1, -1), white),  # White for values
            
            # Text alignment
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            
            # Cell padding
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ])
        
        gene_table.setStyle(gene_table_style)
        
        # Add gene table to elements
        elements.append(gene_table)
        
        # Add spacer between multiple gene entries (if any)
        if gene_analysis != gene_analysis_list[-1]:
            elements.append(Spacer(1, 0.1 * inch))
    
    return elements


def _create_input_section(data: Dict[str, Any], styles) -> list:
    """
    Create the input section of the report exactly matching the reference image.
    """
    elements = []
    
    input_params = data.get('input_parameters', {})
    
    # Get sample type from state (no default - use what's provided)
    sample = input_params.get('sample', '')
    if not sample or sample.upper() == 'N/A':
        sample = input_params.get('specimen_type', '') or input_params.get('specimen_site', '') or ''
    # If still empty, use a generic placeholder
    if not sample:
        sample = 'SAMPLE'
    
    # Create blue header badge as a Table for padding control
    header_text = Paragraph(
        sample.upper(),
        ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            textColor=white,
            alignment=TA_CENTER,
            leading=19,
            spaceAfter=0,
            spaceBefore=0
        )
    )
    
    # Create header table with a little top and bottom padding
    header_table = Table([[header_text]], colWidths=[7.5 * inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#4472C4')),  # Blue color
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),  # A little top padding
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),  # A little bottom padding
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    
    # Add blue header badge
    elements.append(header_table)
    elements.append(Spacer(1, 0.05 * inch))  # Minimal spacing after header
    
    # Prepare data exactly as shown in reference image
    # Column 1: Facility/Provider (left aligned)
    facility_name = input_params.get('facility_name', '-')
    provider = input_params.get('provider', '-')
    phone = input_params.get('phone', '-')
    
    # Column 2: Patient Demographics (center aligned)
    patient_name = input_params.get('patient_name', '-')
    patient_dob = input_params.get('patient_dob', '-')
    patient_gender = input_params.get('patient_gender', '-')
    drug_allergies = input_params.get('drug_allergies', '-')
    
    # Column 3: Lab/Specimen Information (right aligned in reference)
    lab_accession = input_params.get('lab_accession', '-')
    date_collected = input_params.get('date_collected', '-')
    date_received = input_params.get('date_received', '-')
    date_reported = input_params.get('date_reported', '-')
    specimen_type = input_params.get('specimen_type', '-')
    specimen_site = input_params.get('specimen_site', '-')
    
    # Create paragraph style for table cells
    label_style = ParagraphStyle(
        'LabelStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=10,
        spaceBefore=0,
        spaceAfter=0,
        leftIndent=0,
        rightIndent=0,
        alignment=TA_LEFT
    )
    
    value_style = ParagraphStyle(
        'ValueStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=10,
        spaceBefore=0,
        spaceAfter=0,
        leftIndent=0,
        rightIndent=0,
        alignment=TA_LEFT
    )
    
    # Create the table structure exactly as in reference
    table_data = []
    
    # Row 1
    table_data.append([
        Paragraph('<b>Facility Name:</b>', label_style),
        Paragraph(_format_text_field(facility_name), value_style),
        Paragraph('<b>Patient Name:</b>', label_style),
        Paragraph(_format_text_field(patient_name), value_style),
        Paragraph('<b>Lab Accession #:</b>', label_style),
        Paragraph(_format_text_field(lab_accession), value_style)
    ])
    
    # Row 2
    table_data.append([
        Paragraph('<b>Provider:</b>', label_style),
        Paragraph(_format_text_field(provider), value_style),
        Paragraph('<b>Patient DOB:</b>', label_style),
        Paragraph(_format_text_field(patient_dob), value_style),
        Paragraph('<b>Date Collected:</b>', label_style),
        Paragraph(_format_text_field(date_collected), value_style)
    ])
    
    # Row 3
    table_data.append([
        Paragraph('<b>Phone:</b>', label_style),
        Paragraph(_format_text_field(phone), value_style),
        Paragraph('<b>Patient Gender:</b>', label_style),
        Paragraph(_format_text_field(patient_gender), value_style),
        Paragraph('<b>Date Received:</b>', label_style),
        Paragraph(_format_text_field(date_received), value_style)
    ])
    
    # Row 4 - Drug Allergies row
    drug_allergies_style = ParagraphStyle(
        'DrugAllergiesStyle',
        parent=value_style,
        textColor=black  # No special red color since default is now '-'
    )
    
    table_data.append([
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('<b>Drug Allergies:</b>', label_style),
        Paragraph(_format_text_field(drug_allergies), drug_allergies_style),
        Paragraph('<b>Date Reported:</b>', label_style),
        Paragraph(_format_text_field(date_reported), value_style)
    ])
    
    # Row 5
    table_data.append([
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('<b>Specimen Type:</b>', label_style),
        Paragraph(_format_text_field(specimen_type), value_style)
    ])
    
    # Row 6
    table_data.append([
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('<b>Specimen Site:</b>', label_style),
        Paragraph(_format_text_field(specimen_site), value_style)
    ])
    
    # Get medical input parameters
    from utils import get_pathogens_from_input, format_pathogens, get_resistance_genes_from_input, format_resistance_genes
    
    pathogens = get_pathogens_from_input(input_params)
    pathogen_display = format_pathogens(pathogens) if pathogens else '-'
    
    # Get resistance genes
    resistant_genes = get_resistance_genes_from_input(input_params)
    resistant_gene = format_resistance_genes(resistant_genes) if resistant_genes else '-'
    
    age = input_params.get('age', '-')
    if age and age != '-':
        age = str(age)
    
    # Get ICD code names from icd_transformation if available
    from utils import get_icd_names_from_state
    icd_code_names = get_icd_names_from_state(data) if isinstance(data, dict) else '-'
    if not icd_code_names or icd_code_names == 'not specified':
        # Fallback to severity_codes if transformation not available
        severity_codes = input_params.get('severity_codes', '-')
        icd_code_names = severity_codes if severity_codes else '-'
    
    # Get sample and gender for display
    sample_display = input_params.get('sample', '-')
    if not sample_display or sample_display.upper() == 'N/A':
        sample_display = '-'
    
    gender = input_params.get('gender', '-')
    if not gender or gender.upper() == 'N/A':
        gender = '-'
    
    # Row 7 - Medical Input Parameters
    table_data.append([
        Paragraph('<b>Pathogen:</b>', label_style),
        Paragraph(_format_text_field(pathogen_display), value_style),
        Paragraph('<b>Resistant Gene:</b>', label_style),
        Paragraph(_format_text_field(resistant_gene), value_style),
        Paragraph('<b>Sample:</b>', label_style),
        Paragraph(_format_text_field(sample_display), value_style)
    ])
    
    # Row 8 - More Medical Input Parameters (Gender under Sample)
    table_data.append([
        Paragraph('<b>Age:</b>', label_style),
        Paragraph(_format_text_field(age), value_style),
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('<b>Gender:</b>', label_style),
        Paragraph(_format_text_field(gender), value_style)
    ])
    
    # Row 9 - ICD Codes
    table_data.append([
        Paragraph('', label_style),
        Paragraph('', value_style),
        Paragraph('<b>ICD Codes:</b>', label_style),
        Paragraph(_format_text_field(icd_code_names), value_style),
        Paragraph('', label_style),
        Paragraph('', value_style)
    ])
    
    # Calculate column widths
    total_width = 7.5 * inch
    section_width = total_width / 3
    label_width = section_width * 0.48
    value_width = section_width * 0.52
    
    col_widths = [label_width, value_width, label_width, value_width, label_width, value_width]
    
    # Create table with minimal cell padding
    table = Table(table_data, colWidths=col_widths, hAlign='LEFT')
    
    # Style the table to match reference exactly
    table_style = TableStyle([
        ('BOX', (0, 0), (-1, -1), 0, white),
        ('INNERGRID', (0, 0), (-1, -1), 0, white),
        
        ('BACKGROUND', (0, 0), (0, -1), white),
        ('BACKGROUND', (2, 0), (2, -1), white),
        ('BACKGROUND', (4, 0), (4, -1), white),
        
        ('BACKGROUND', (1, 0), (1, -1), white),
        ('BACKGROUND', (3, 0), (3, -1), white),
        ('BACKGROUND', (5, 0), (5, -1), white),
        
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
    ])
    
    table.setStyle(table_style)
    
    # Add table to elements
    elements.append(table)
    elements.append(Spacer(1, 0.1 * inch))
    
    return elements


def _create_negative_sections(data: Dict[str, Any], styles) -> list:
    """
    Create the negative organisms and negative resistance genes sections.
    Uses static lists that always appear in the PDF, regardless of input.
    """
    elements = []
    
    # Static list of negative organisms (always displayed)
    static_negative_organisms = [
        'Acinetobacter baumannii',
        'Bacteroides fragilis',
        'Candida glabrata',
        'Candida albicans',
        'Candida auris',
        'Candida krusei',
        'Candida lusitaniae',
        'Candida parapsilosis',
        'Candida tropicalis',
        'Citrobacter freundii',
        'Clostridium novyi',
        'Clostridium perfringens',
        'Clostridium septicum',
        'Enterobacter cloacae',
        'Enterococcus faecium',
        'Escherichia coli',
        'Group A strep',
        'Group B strep',
        'Group C and G Strep',
        'Herpes Zoster',
        'HSV-1 (Herpes Simplex)',
        'HSV-2 (Herpes Simplex)',
        'kingella kingae',
        'Klebsiella aerogenes',
        'Klebsiella oxytoca',
        'Klebsiella pneumoniae',
        'Morganella morganii',
        'Not an organism',
        'Proteus mirabilis',
        'Proteus vulgaris',
        'Pseudomonas aeruginosa',
        'Trichophyton spp.'
    ]
    
    # Static list of negative resistance genes (always displayed)
    static_negative_resistance_genes = [
        'ampC',
        'Ant-2',
        'Aph 2',
        'aph3',
        'CTX-M1',
        'CTX-M2',
        'dfrA1',
        'dfrA5',
        'Erm B',
        'ErmA',
        'femA',
        'Gyrase A',
        'KPC',
        'mefA',
        'NDM',
        'OXA-48',
        'Par C',
        'QnrA',
        'QnrB',
        'SHV',
        'Sul 2',
        'Sul1',
        'TEM',
        'Tet O',
        'tetB',
        'vanA1',
        'vanA2',
        'VanB'
    ]
    
    # Create styles for negative sections
    negative_header_style = ParagraphStyle(
        'NegativeHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        textColor=white,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    negative_content_style = ParagraphStyle(
        'NegativeContent',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0
    )
    
    # Section 1: NEGATIVE ORGANISMS TESTED (always displayed)
    elements.append(Spacer(1, 0.15 * inch))
    
    # Header
    org_header_text = Paragraph("NEGATIVE ORGANISMS TESTED", negative_header_style)
    org_header_table = Table([[org_header_text]], colWidths=[7.5 * inch])
    org_header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#4472C4')),  # Blue header
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(org_header_table)
    
    # Content
    org_content_text = ', '.join(static_negative_organisms)
    org_content = Paragraph(org_content_text, negative_content_style)
    org_content_table = Table([[org_content]], colWidths=[7.5 * inch])
    org_content_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, black),
        ('BACKGROUND', (0, 0), (-1, -1), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(org_content_table)
    
    # Section 2: NEGATIVE RESISTANCE GENES TESTED (always displayed)
    elements.append(Spacer(1, 0.15 * inch))
    
    # Header
    gene_header_text = Paragraph("NEGATIVE RESISTANCE GENES TESTED", negative_header_style)
    gene_header_table = Table([[gene_header_text]], colWidths=[7.5 * inch])
    gene_header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), HexColor('#4472C4')),  # Blue header
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(gene_header_table)
    
    # Content
    gene_content_text = ', '.join(static_negative_resistance_genes)
    gene_content = Paragraph(gene_content_text, negative_content_style)
    gene_content_table = Table([[gene_content]], colWidths=[7.5 * inch])
    gene_content_table.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, black),
        ('BACKGROUND', (0, 0), (-1, -1), white),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(gene_content_table)
    
    return elements


# Global variable to track total pages and current page
_total_pages = [0]
_current_page = [0]

def _draw_footer(canv):
    """Draw footer at the bottom of the page."""
    footer_text = (
        "The estimated microbial load is the approximate copies of target nucleic acid, present in the original sample (copies per mL), categorized as follows: HIGH (>1 million copies/mL). MODERATE (500,000-1 million copies per mL), and LOW (100,000-500,000 copies per mL). Levels less than LOW are generally not reported, unless deemed potentially significant. Loads less than LOW generally represent normal flora/contaminants. "
        "CLIA# 45D2257672 Processing and Detection Methodology: DNA/RNA extraction from the sample was performed. Reverse transcriptase polymerase chain reaction (TaqMan qPCR) was utilized for detection."
    )
    
    # Save canvas state
    canv.saveState()
    
    # Set footer style
    canv.setFont('Helvetica', 8)
    canv.setFillColor(black)
    
    # Get page dimensions
    page_width = letter[0]
    left_margin = 0.5 * inch
    right_margin = 0.5 * inch
    footer_width = page_width - left_margin - right_margin
    
    # Calculate footer position from bottom of page
    footer_y = 0.6 * inch
    
    # Draw footer text (wrapped to fit page width)
    from reportlab.lib.utils import simpleSplit
    lines = simpleSplit(footer_text, 'Helvetica', 8, footer_width)
    
    # Calculate line height
    line_height = 11  # Line spacing
    
    # Draw each line from top to bottom (first line at top)
    # Start from the topmost position and work down
    y_position = footer_y + (len(lines) - 1) * line_height
    for line in lines:
        canv.drawString(left_margin, y_position, line)
        y_position -= line_height  # Move down for next line
    
    # Restore canvas state
    canv.restoreState()

def _count_pages_first_pass(canv, doc):
    """First pass: just count pages."""
    page_num = canv.getPageNumber()
    _total_pages[0] = max(_total_pages[0], page_num)

def _first_page_footer(canv, doc):
    """Footer callback for first page - track page number."""
    _current_page[0] = canv.getPageNumber()
    _total_pages[0] = max(_total_pages[0], _current_page[0])

def _later_pages_footer(canv, doc):
    """Footer callback for later pages - draw footer on page 3 only."""
    _current_page[0] = canv.getPageNumber()
    _total_pages[0] = max(_total_pages[0], _current_page[0])
    
    # Draw footer only on page 3
    if _current_page[0] == 3:
        _draw_footer(canv)


def export_to_pdf(data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Export data to PDF with exact design matching reference image.
    
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
    
    # Create PDF document with tight margins
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=0.5*inch,
        leftMargin=0.5*inch,
        topMargin=0.4*inch,
        bottomMargin=0.5*inch
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Build story (list of flowables)
    story = []
    
    # Add input section
    input_elements = _create_input_section(data, styles)
    story.extend(input_elements)
    
    # Add medication section
    medication_elements = _create_medication_section(data, styles)
    story.extend(medication_elements)
    
    # Add gene section
    gene_elements = _create_gene_section(data, styles)
    story.extend(gene_elements)
    
    # Add negative sections
    negative_elements = _create_negative_sections(data, styles)
    story.extend(negative_elements)
    
    # Reset footer tracking for new PDF
    global _total_pages, _current_page
    _total_pages[0] = 0
    _current_page[0] = 0
    
    # Build PDF with footer callbacks
    # The footer will be drawn on the last page only
    # We track the maximum page number, and draw footer when current page equals max
    doc.build(story, onFirstPage=_first_page_footer, onLaterPages=_later_pages_footer)
    
    logger.info(f"PDF report exported to: {output_path}")
    return output_path