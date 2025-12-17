"""
PDF Export Module - Dynamic with Staphylococcus-style layout
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from utils import fix_text_encoding

logger = logging.getLogger(__name__)

try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False
    logger.warning("xhtml2pdf not available. Install with: pip install xhtml2pdf")


def _format_text_field(text: Optional[str]) -> str:
    """Format any text field to fix encoding issues - applies to all fields."""
    return fix_text_encoding(text)


def _format_dose_duration(dose_duration: Optional[str]) -> str:
    """Format dose_duration string for display."""
    if not dose_duration:
        return "Not specified"
    
    # Parse the format: 'dose,route,frequency,duration'
    parts = dose_duration.split(',')
    if len(parts) >= 4:
        dose, route, frequency, duration = parts[0], parts[1], parts[2], parts[3]
        result = f"{_format_text_field(dose)} {_format_text_field(route)}"
        if frequency and frequency.lower() != 'null':
            result += f", {_format_text_field(frequency)}"
        if duration and duration.lower() != 'null':
            result += f" for {_format_text_field(duration)}"
        return result
    elif len(parts) >= 3:
        dose, route, frequency = parts[0], parts[1], parts[2]
        result = f"{_format_text_field(dose)} {_format_text_field(route)}"
        if frequency and frequency.lower() != 'null':
            result += f", {_format_text_field(frequency)}"
        return result
    else:
        return _format_text_field(dose_duration)


def _format_considerations(text: Optional[str]) -> str:
    """Format considerations text for better display."""
    if not text or text.lower() in ['not specified', 'no additional considerations', 'none']:
        return "No additional considerations"
    
    # Use the utility function to fix encoding
    return fix_text_encoding(text)


def _generate_html_template(data: Dict[str, Any]) -> str:
    """Generate HTML template from pipeline output data - Staphylococcus-style layout."""
    
    input_params = data.get('input_parameters', {})
    result = data.get('result', {})
    therapy_plan = result.get('antibiotic_therapy_plan', {})
    pharmacist_analysis = result.get('pharmacist_analysis_on_resistant_gene', [])
    extraction_date = data.get('extraction_date', '')
    
    # Format extraction date
    if extraction_date:
        try:
            dt = datetime.fromisoformat(extraction_date.replace('Z', '+00:00'))
            formatted_date = dt.strftime('%B %d, %Y at %I:%M %p')
        except:
            formatted_date = extraction_date
    else:
        formatted_date = "Not available"
    
    # Get input parameters and format them
    pathogen_name = _format_text_field(input_params.get('pathogen_name', 'N/A'))
    resistant_gene = _format_text_field(input_params.get('resistant_gene', 'N/A'))
    pathogen_count = _format_text_field(input_params.get('pathogen_count', 'N/A'))
    severity_codes = _format_text_field(input_params.get('severity_codes', 'N/A'))
    age = input_params.get('age', 'N/A')
    if isinstance(age, (int, float)):
        age_str = f"{age} years"
    else:
        age_str = _format_text_field(str(age))
    
    # Start HTML with Staphylococcus-style layout
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
    <meta charset="UTF-8">
    <title>Antibiotic Therapy Plan Report</title>
    <style>
        body {{
            font-family: Arial, Helvetica, sans-serif;
            font-size: 10pt;
            line-height: 1.6;
            color: #1a1a1a;
            margin: 0;
            padding: 20px;
            background-color: #ffffff;
        }}
        
        .header {{
            text-align: left;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #ddd;
        }}
        
        .header h1 {{
            color: #1a1a1a;
            margin: 0 0 8px 0;
            font-size: 18pt;
            font-weight: 700;
        }}
        
        .header .subtitle {{
            color: #1a1a1a;
            font-size: 12pt;
            margin-bottom: 5px;
            font-weight: 600;
        }}
        
        .header .date {{
            color: #666;
            font-size: 9pt;
            font-weight: 400;
        }}
        
        .input-info {{
            background-color: #ffffff;
            border: none;
            border-radius: 0;
            padding: 0;
            margin-bottom: 20px;
            font-size: 9.5pt;
        }}
        
        .input-info table {{
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #ccc;
        }}
        
        .input-info td {{
            padding: 10px 12px;
            border-right: 1px solid #ccc;
            border-bottom: 1px solid #ccc;
            background-color: #ffffff;
            font-weight: 600;
            color: #1a1a1a;
            vertical-align: middle;
        }}
        
        .input-info tr:last-child td {{
            border-bottom: none;
        }}
        
        .input-info td:last-child {{
            border-right: none;
        }}
        
        .therapy-section {{
            margin-bottom: 30px;
            page-break-inside: avoid;
        }}
        
        .therapy-section h2 {{
            color: #1a365d;
            background-color: #e6f3ff;
            padding: 12px 20px;
            margin: 40px 0 15px 0;
            font-size: 14pt;
            font-weight: 600;
            border-left: 5px solid #1a365d;
            border-radius: 3px;
            page-break-after: avoid;
            page-break-inside: avoid;
        }}
        
        .therapy-section:first-child h2 {{
            margin-top: 20px;
        }}
        
        /* Staphylococcus-style table layout - Compact */
        .staph-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 8px 0;
            border: 1px solid #1a365d;
            page-break-inside: avoid !important;
            page-break-after: avoid !important;
            display: table;
            orphans: 3;
            widows: 3;
        }}
        
        .staph-table th {{
            background-color: #1a365d;
            color: white;
            padding: 8px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 9.5pt;
            border: 1px solid #1a365d;
        }}
        
        .staph-table td {{
            padding: 6px 10px;
            border: 1px solid #ddd;
            vertical-align: top;
            font-size: 9.5pt;
            page-break-inside: avoid !important;
            orphans: 3;
            widows: 3;
        }}
        
        .staph-table tr {{
            page-break-inside: avoid !important;
            page-break-after: avoid !important;
            orphans: 3;
            widows: 3;
        }}
        
        .staph-table .medication-row {{
            background-color: transparent;
            font-weight: 600;
            color: #1a365d;
            page-break-inside: avoid;
        }}
        
        .staph-table .medication-row td {{
            padding: 8px 10px;
            font-size: 10pt;
            background-color: transparent;
            page-break-inside: avoid;
        }}
        
        .staph-table .coverage-row {{
            page-break-inside: avoid;
        }}
        
        .staph-table .coverage-row td {{
            background-color: transparent;
            color: #c53030;
            font-weight: 500;
            padding: 6px 10px;
            font-size: 9pt;
            page-break-inside: avoid;
        }}
        
        .staph-table .considerations-row {{
            page-break-inside: avoid;
        }}
        
        .staph-table .considerations-row td {{
            background-color: transparent;
            color: #744210;
            padding: 6px 10px;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        
        .antibiotic-table-wrapper {{
            page-break-inside: avoid !important;
            page-break-after: avoid !important;
            page-break-before: auto;
            margin-bottom: 8px;
            display: block;
            overflow: hidden;
        }}
        
        .antibiotic-table-wrapper table {{
            page-break-inside: avoid !important;
            border-collapse: separate;
            border-spacing: 0;
        }}
        
        .antibiotic-table-wrapper table tbody {{
            page-break-inside: avoid !important;
            display: table-row-group;
        }}
        
        .antibiotic-table-wrapper table tr {{
            page-break-inside: avoid !important;
            page-break-after: avoid !important;
            display: table-row;
        }}
        
        .antibiotic-table-wrapper table td {{
            page-break-inside: avoid !important;
            display: table-cell;
        }}
        
        .staph-table .coverage-header {{
            font-weight: 600;
            color: #c53030;
            margin-right: 5px;
        }}
        
        .staph-table .considerations-header {{
            font-weight: 600;
            color: #8b4513;
            margin-right: 5px;
        }}
        
        .or-separator {{
            text-align: center;
            margin: 2px 0;
            font-weight: 600;
            color: #666;
            font-size: 9pt;
            font-style: italic;
        }}
        
        .therapy-section {{
            margin-bottom: 25px;
            page-break-inside: avoid;
        }}
        
        .pharmacist-analysis {{
            margin-top: 25px;
            padding: 15px;
            background-color: #f0f4f8;
            border: 2px solid #1a365d;
            border-radius: 5px;
        }}
        
        .pharmacist-analysis h2 {{
            color: #1a365d;
            margin-top: 0;
            margin-bottom: 12px;
            font-size: 13pt;
            font-weight: 600;
            padding-bottom: 8px;
            border-bottom: 2px solid #1a365d;
        }}
        
        .gene-analysis {{
            margin-bottom: 15px;
            padding: 12px;
            background-color: white;
            border-left: 4px solid #1a365d;
            border-radius: 3px;
        }}
        
        .gene-analysis h3 {{
            color: #1a365d;
            margin-top: 0;
            margin-bottom: 10px;
            font-size: 11pt;
            font-weight: 600;
        }}
        
        .gene-analysis .gene-name {{
            color: #c53030;
            font-weight: 600;
        }}
        
        .gene-analysis .mechanism {{
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-left: 3px solid #4a5568;
            font-style: italic;
            font-size: 9pt;
        }}
        
        .gene-analysis .clinical-impact {{
            margin-top: 10px;
            line-height: 1.5;
            font-size: 9pt;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Antibiotic Therapy Plan Report</h1>
        <div class="subtitle">Focus Therapy: {pathogen_name}</div>
        <div class="date">Generated on: {formatted_date}</div>
    </div>
    
    <div class="input-info">
        <table>
            <tr>
                <td style="width: 18%;">Pathogen Name:</td>
                <td style="width: 32%;">{pathogen_name}</td>
                <td style="width: 18%;">Resistance Gene(s):</td>
                <td style="width: 32%;">{resistant_gene}</td>
            </tr>
            <tr>
                <td style="width: 18%;">Pathogen Count:</td>
                <td style="width: 32%;">{pathogen_count}</td>
                <td style="width: 18%;">Severity Codes:</td>
                <td style="width: 32%;">{severity_codes}</td>
            </tr>
            <tr>
                <td style="width: 18%;">Patient Age:</td>
                <td style="width: 32%;">{age_str}</td>
                <td style="width: 18%;"></td>
                <td style="width: 32%;"></td>
            </tr>
        </table>
    </div>
"""
    
    # Process therapy plan data dynamically but in Staphylococcus layout
    html += _generate_therapy_section(therapy_plan)
    
    # Add pharmacist analysis if available
    if pharmacist_analysis:
        html += _generate_pharmacist_analysis(pharmacist_analysis)
    
    html += """
</body>
</html>"""
    
    return html


def _generate_therapy_section(therapy_plan: Dict[str, Any]) -> str:
    """Generate therapy section in Staphylococcus-style layout."""
    html = ""
    
    # First Choice Antibiotics
    first_choice = therapy_plan.get('first_choice', [])
    if first_choice:
        html += '<div class="therapy-section">\n'
        html += '<h2>First Line Treatment Options</h2>\n'
        
        for idx, ab in enumerate(first_choice):
            if idx > 0:
                html += '<div class="or-separator">OR</div>\n'
            
            medical_name = _format_text_field(ab.get('medical_name', 'Unknown'))
            route = _format_text_field(ab.get('route_of_administration', 'Not specified'))
            dose_duration = _format_dose_duration(ab.get('dose_duration'))
            coverage_for = _format_text_field(ab.get('coverage_for', 'Not specified'))
            general_considerations = ab.get('general_considerations', 'No additional considerations')
            renal_adjustment = ab.get('renal_adjustment', '')
            
            # Combine considerations with renal adjustment if available
            considerations = general_considerations
            if renal_adjustment and renal_adjustment.lower() not in ['not specified', 'none', '']:
                formatted_renal = _format_text_field(renal_adjustment)
                considerations = f"{general_considerations} Renal adjustment: {formatted_renal}"
            
            html += f"""
        <div class="antibiotic-table-wrapper" style="page-break-inside: avoid !important; border: 1px solid #1a365d; margin: 8px 0;">
        <div class="medication-row" style="display: table; width: 100%; page-break-inside: avoid !important; background-color: #f8f9fa; font-weight: 600; color: #1a365d; border-bottom: 1px solid #ddd;">
            <div style="display: table-cell; width: 30%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{medical_name}</div>
            <div style="display: table-cell; width: 12%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{route}</div>
            <div style="display: table-cell; width: 58%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{dose_duration}</div>
        </div>
        <div class="coverage-row" style="display: block; width: 100%; page-break-inside: avoid !important; background-color: transparent; color: #c53030; font-weight: 500; padding: 6px 10px; font-size: 9pt; border-bottom: 1px solid #ddd;">
            <span class="coverage-header">Coverage For:</span> {coverage_for}
        </div>
        <div class="considerations-row" style="display: block; width: 100%; page-break-inside: avoid !important; background-color: transparent; color: #744210; padding: 6px 10px; font-size: 9pt; line-height: 1.4;">
            <span class="considerations-header">Considerations:</span> {_format_considerations(considerations)}
        </div>
        </div>
"""
        html += '</div>\n'
    
    # Second Choice Antibiotics
    second_choice = therapy_plan.get('second_choice', [])
    if second_choice:
        html += '<div class="therapy-section">\n'
        html += '<h2>Second Choice Treatment Options</h2>\n'
        
        for idx, ab in enumerate(second_choice):
            if idx > 0:
                html += '<div class="or-separator">OR</div>\n'
            
            medical_name = _format_text_field(ab.get('medical_name', 'Unknown'))
            route = _format_text_field(ab.get('route_of_administration', 'Not specified'))
            dose_duration = _format_dose_duration(ab.get('dose_duration'))
            coverage_for = _format_text_field(ab.get('coverage_for', 'Not specified'))
            general_considerations = ab.get('general_considerations', 'No additional considerations')
            renal_adjustment = ab.get('renal_adjustment', '')
            
            # Combine considerations with renal adjustment if available
            considerations = general_considerations
            if renal_adjustment and renal_adjustment.lower() not in ['not specified', 'none', '']:
                formatted_renal = _format_text_field(renal_adjustment)
                considerations = f"{general_considerations} Renal adjustment: {formatted_renal}"
            
            html += f"""
        <div class="antibiotic-table-wrapper" style="page-break-inside: avoid !important; border: 1px solid #1a365d; margin: 8px 0;">
        <div class="medication-row" style="display: table; width: 100%; page-break-inside: avoid !important; background-color: #f8f9fa; font-weight: 600; color: #1a365d; border-bottom: 1px solid #ddd;">
            <div style="display: table-cell; width: 30%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{medical_name}</div>
            <div style="display: table-cell; width: 12%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{route}</div>
            <div style="display: table-cell; width: 58%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{dose_duration}</div>
        </div>
        <div class="coverage-row" style="display: block; width: 100%; page-break-inside: avoid !important; background-color: transparent; color: #c53030; font-weight: 500; padding: 6px 10px; font-size: 9pt; border-bottom: 1px solid #ddd;">
            <span class="coverage-header">Coverage For:</span> {coverage_for}
        </div>
        <div class="considerations-row" style="display: block; width: 100%; page-break-inside: avoid !important; background-color: transparent; color: #744210; padding: 6px 10px; font-size: 9pt; line-height: 1.4;">
            <span class="considerations-header">Considerations:</span> {_format_considerations(considerations)}
        </div>
        </div>
"""
        html += '</div>\n'
    
    # Alternative Antibiotics
    alternative = therapy_plan.get('alternative_antibiotic', [])
    if alternative:
        html += '<div class="therapy-section">\n'
        html += '<h2>Alternative Treatment Options</h2>\n'
        
        for idx, ab in enumerate(alternative):
            if idx > 0:
                html += '<div class="or-separator">OR</div>\n'
            
            medical_name = _format_text_field(ab.get('medical_name', 'Unknown'))
            route = _format_text_field(ab.get('route_of_administration', 'Not specified'))
            dose_duration = _format_dose_duration(ab.get('dose_duration'))
            coverage_for = _format_text_field(ab.get('coverage_for', 'Not specified'))
            general_considerations = ab.get('general_considerations', 'No additional considerations')
            renal_adjustment = ab.get('renal_adjustment', '')
            
            # Combine considerations with renal adjustment if available
            considerations = general_considerations
            if renal_adjustment and renal_adjustment.lower() not in ['not specified', 'none', '']:
                formatted_renal = _format_text_field(renal_adjustment)
                considerations = f"{general_considerations} Renal adjustment: {formatted_renal}"
            
            html += f"""
        <div class="antibiotic-table-wrapper" style="page-break-inside: avoid !important; border: 1px solid #1a365d; margin: 8px 0;">
        <div class="medication-row" style="display: table; width: 100%; page-break-inside: avoid !important; background-color: #f8f9fa; font-weight: 600; color: #1a365d; border-bottom: 1px solid #ddd;">
            <div style="display: table-cell; width: 30%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{medical_name}</div>
            <div style="display: table-cell; width: 12%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{route}</div>
            <div style="display: table-cell; width: 58%; padding: 8px 10px; font-size: 10pt; page-break-inside: avoid !important;">{dose_duration}</div>
        </div>
        <div class="coverage-row" style="display: block; width: 100%; page-break-inside: avoid !important; background-color: transparent; color: #c53030; font-weight: 500; padding: 6px 10px; font-size: 9pt; border-bottom: 1px solid #ddd;">
            <span class="coverage-header">Coverage For:</span> {coverage_for}
        </div>
        <div class="considerations-row" style="display: block; width: 100%; page-break-inside: avoid !important; background-color: transparent; color: #744210; padding: 6px 10px; font-size: 9pt; line-height: 1.4;">
            <span class="considerations-header">Considerations:</span> {_format_considerations(considerations)}
        </div>
        </div>
"""
        html += '</div>\n'
    
    return html


def _generate_pharmacist_analysis(pharmacist_analysis: list) -> str:
    """Generate pharmacist analysis section."""
    html = """
    <div class="pharmacist-analysis">
        <h2>Pharmacist Analysis on Resistance Gene</h2>
"""
    
    for analysis in pharmacist_analysis:
        gene_name = _format_text_field(analysis.get('detected_resistant_gene_name', 'Unknown'))
        affected_classes = analysis.get('potential_medication_class_affected') or analysis.get('potential_medication_classes_affected', 'Not specified')
        considerations = analysis.get('general_considerations', 'No analysis available')
        
        html += f"""
        <div class="gene-analysis">
            <h3>Resistance Gene: <span class="gene-name">{gene_name}</span></h3>
            
            <div class="mechanism">
                <strong>Affected Medication Classes:</strong> {_format_considerations(affected_classes)}
            </div>
            
            <div class="clinical-impact">
                <strong>Clinical Analysis:</strong><br/>
                {_format_considerations(considerations)}
            </div>
        </div>
"""
    
    html += """
    </div>
"""
    return html


def export_to_pdf(data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Export pipeline output data to PDF.
    
    Args:
        data: Pipeline output dictionary (from JSON file)
        output_path: Optional path for output PDF. If None, generates based on timestamp.
        
    Returns:
        Path to generated PDF file
    """
    if not XHTML2PDF_AVAILABLE:
        raise ImportError("xhtml2pdf is not available. Install with: pip install xhtml2pdf")
    
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f'antibiotic_therapy_report_{timestamp}.pdf')
    
    # Generate HTML
    html_content = _generate_html_template(data)
    
    # Convert to PDF
    try:
        from io import BytesIO
        
        # Ensure proper UTF-8 encoding
        html_bytes = html_content.encode('utf-8', errors='replace')
        
        result_file = BytesIO()
        pdf = pisa.CreatePDF(
            BytesIO(html_bytes),
            result_file,
            encoding='utf-8'
        )
        
        if pdf.err:
            raise Exception(f"Error creating PDF: {pdf.err}")
        
        # Write to file
        with open(output_path, 'wb') as f:
            f.write(result_file.getvalue())
        
        logger.info(f"PDF exported successfully to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting PDF: {e}", exc_info=True)
        raise


def export_to_html(data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Export pipeline output data to HTML file (for preview).
    
    Args:
        data: Pipeline output dictionary (from JSON file)
        output_path: Optional path for output HTML. If None, generates based on timestamp.
        
    Returns:
        Path to generated HTML file
    """
    # Generate output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('output')
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f'antibiotic_therapy_report_{timestamp}.html')
    
    # Generate HTML
    html_content = _generate_html_template(data)
    
    # Write to file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"HTML exported successfully to: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting HTML: {e}", exc_info=True)
        raise