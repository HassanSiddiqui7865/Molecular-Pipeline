"""
PDF Export Module using xhtml2pdf (pisa) - Pure Python HTML/CSS to PDF conversion
No system dependencies, works on all platforms
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import html
from io import BytesIO

logger = logging.getLogger(__name__)

try:
    from xhtml2pdf import pisa
    XHTML2PDF_AVAILABLE = True
except ImportError:
    XHTML2PDF_AVAILABLE = False
    logger.warning("xhtml2pdf (pisa) not available. Install with: pip install xhtml2pdf")


def _format_text_field(text: Optional[str]) -> str:
    """Format any text field to handle None values and escape HTML."""
    if text is None:
        return "N/A"
    text_str = str(text).strip()
    if not text_str:
        return "N/A"
    return html.escape(text_str)


def _create_html_template(data: Dict[str, Any]) -> str:
    """Create HTML template for the PDF report."""
    input_params = data.get('input_parameters', {})
    result = data.get('result', {})
    therapy_plan = result.get('antibiotic_therapy_plan', {})
    
    # Get ICD codes with names
    from utils import get_icd_names_from_state, get_pathogens_from_input, format_pathogens, get_resistance_genes_from_input, format_resistance_genes
    icd_code_names = get_icd_names_from_state(data) if isinstance(data, dict) else 'N/A'
    if not icd_code_names or icd_code_names == 'not specified' or icd_code_names == 'N/A':
        severity_codes = input_params.get('severity_codes', [])
        if severity_codes:
            icd_code_names = ', '.join(severity_codes) if isinstance(severity_codes, list) else str(severity_codes)
        else:
            icd_code_names = 'N/A'
    
    pathogens = get_pathogens_from_input(input_params)
    pathogen_display = format_pathogens(pathogens) if pathogens else 'N/A'
    resistant_genes = get_resistance_genes_from_input(input_params)
    resistant_gene = format_resistance_genes(resistant_genes) if resistant_genes else 'N/A'
    
    # Get sample type from input parameters (for header display)
    sample = input_params.get('sample', 'N/A')
    if sample and sample != 'N/A':
        sample = sample.upper()
    else:
        sample = "N/A"
    
    # Get gene analysis
    gene_analysis_list = result.get('pharmacist_analysis_on_resistant_gene', [])
    
    # Build medication HTML
    medications_html = _build_medications_html(therapy_plan)
    
    # Build gene analysis HTML
    gene_html = _build_gene_html(gene_analysis_list)
    
    # FIXED: Create proper header section like in the image
    header_html = f"""
    <!-- Header Section -->
    <div class="header-section">
        <div class="header-main">
            <div class="header-title">#{sample if sample != 'N/A' else 'N/A'}</div>
        </div>
    </div>
    
    <!-- Patient Info Section -->
    <div class="patient-info-section">
        <table class="patient-info-table">
            <tr>
                <td class="info-label">Patient Name:</td>
                <td class="info-value">{_format_text_field(input_params.get('patient_name', 'N/A'))}</td>
                <td class="info-label">Lab Accession:</td>
                <td class="info-value">{_format_text_field(input_params.get('lab_accession', 'N/A'))}</td>
            </tr>
            <tr>
                <td class="info-label">Provider:</td>
                <td class="info-value">{_format_text_field(input_params.get('provider', 'N/A'))}</td>
                <td class="info-label">Date Collected:</td>
                <td class="info-value">{_format_text_field(input_params.get('date_collected', 'N/A'))}</td>
            </tr>
            <tr>
                <td class="info-label">Phone:</td>
                <td class="info-value">{_format_text_field(input_params.get('phone', 'N/A'))}</td>
                <td class="info-label">Date Received:</td>
                <td class="info-value">{_format_text_field(input_params.get('date_received', 'N/A'))}</td>
            </tr>
            <tr>
                <td class="info-label">Patient DOB:</td>
                <td class="info-value">{_format_text_field(input_params.get('patient_dob', 'N/A'))}</td>
                <td class="info-label">Date Reported:</td>
                <td class="info-value">{_format_text_field(input_params.get('date_reported', 'N/A'))}</td>
            </tr>
            <tr>
                <td class="info-label">Patient Gender:</td>
                <td class="info-value">{_format_text_field(input_params.get('patient_gender', input_params.get('gender', 'N/A')))}</td>
                <td class="info-label">Specimen Type:</td>
                <td class="info-value">{_format_text_field('N/A')}</td>
            </tr>
            <tr>
                <td class="info-label">Drug Allergies:</td>
                <td class="info-value">{_format_text_field(', '.join(input_params.get('drug_allergies', input_params.get('allergy', []))) if input_params.get('drug_allergies') or input_params.get('allergy') else 'N/A')}</td>
                <td class="info-label">Specimen Site:</td>
                <td class="info-value">{_format_text_field(input_params.get('specimen_site', 'N/A'))}</td>
            </tr>
            <tr>
                <td class="info-label">Age:</td>
                <td class="info-value">{_format_text_field(input_params.get('age', 'N/A'))}</td>
                <td class="info-label">Systemic:</td>
                <td class="info-value">{_format_text_field('Yes' if input_params.get('systemic') is True else ('No' if input_params.get('systemic') is False else 'N/A'))}</td>
            </tr>
        </table>
    </div>
    
    <!-- Test Results Section -->
    <div class="test-results-section">
        <div class="section-header">TEST RESULTS</div>
        <table class="test-results-table">
            <tr>
                <td class="result-label">Pathogen:</td>
                <td class="result-value">{_format_text_field(pathogen_display)}</td>
            </tr>
            <tr>
                <td class="result-label">Resistant Gene:</td>
                <td class="result-value">{_format_text_field(resistant_gene)}</td>
            </tr>
            <tr>
                <td class="result-label">Severity Codes:</td>
                <td class="result-value">{_format_text_field(icd_code_names if icd_code_names and icd_code_names != 'not specified' else 'N/A')}</td>
            </tr>
        </table>
    </div>
    """
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        {_get_css_styles()}
    </style>
</head>
<body>
    {header_html}
    
    <!-- Medication Section -->
    {medications_html if medications_html else ''}
    
    <!-- Gene Section -->
    {gene_html if gene_html else ''}
    
    <!-- Negative Sections -->
    {_build_negative_sections_html(data)}
    
    <!-- Footer -->
    <div class="footer">
        <p>The estimated microbial load is the approximate copies of target nucleic acid, present in the original sample (copies per mL), categorized as follows: HIGH (>1 million copies/mL). MODERATE (500,000-1 million copies per mL), and LOW (100,000-500,000 copies per mL). Levels less than LOW are generally not reported, unless deemed potentially significant. Loads less than LOW generally represent normal flora/contaminants.</p>
        <p>CLIA# 45D2257672 Processing and Detection Methodology: DNA/RNA extraction from the sample was performed. Reverse transcriptase polymerase chain reaction (TaqMan qPCR) was utilized for detection.</p>
    </div>
</body>
</html>
"""
    return html_content


def _build_medications_html(therapy_plan: Dict[str, Any]) -> str:
    """Build HTML for medication recommendations."""
    first_choice = therapy_plan.get('first_choice', [])
    second_choice = therapy_plan.get('second_choice', [])
    alternative = therapy_plan.get('alternative_antibiotic', [])
    
    if not (first_choice or second_choice or alternative):
        return ""
    
    html_parts = []
    
    # Check if we have any medications
    if first_choice or second_choice or alternative:
        html_parts.append('<div class="medication-section">')
        html_parts.append('<div class="section-header">MEDICATION RECOMMENDATIONS</div>')
        
        from utils import fix_text_encoding
        
        # Process first_choice medications
        if first_choice:
            for idx, med in enumerate(first_choice):
                name = fix_text_encoding(med.get('medical_name', ''))
                route = fix_text_encoding(med.get('route_of_administration', ''))
                dose = fix_text_encoding(med.get('dose_duration', ''))
                coverage = fix_text_encoding(med.get('coverage_for', ''))
                renal = fix_text_encoding(med.get('renal_adjustment', ''))
                considerations = fix_text_encoding(med.get('general_considerations', ''))
                
                html_parts.append('<div class="medication-card">')
                # Show "First Line" tag only on first medication of this category
                if idx == 0:
                    html_parts.append('<table class="medication-header"><tr>')
                    html_parts.append('<td class="medication-tag-section first-line-tag">First Line</td>')
                    html_parts.append(f'<td class="medication-name-section"><span class="medication-name">{_format_text_field(name)}</span></td>')
                    html_parts.append('</tr></table>')
                else:
                    html_parts.append('<table class="medication-header"><tr>')
                    html_parts.append(f'<td class="medication-name-section full-width"><span class="medication-name">{_format_text_field(name)}</span></td>')
                    html_parts.append('</tr></table>')
                
                html_parts.append('<div class="medication-details">')
                html_parts.append('<table class="medication-table">')
                html_parts.append('<tr>')
                html_parts.append(f'<td class="detail-label">Route:</td>')
                html_parts.append(f'<td class="detail-value">{_format_text_field(route)}</td>')
                html_parts.append('</tr>')
                html_parts.append('<tr>')
                html_parts.append(f'<td class="detail-label">Dose:</td>')
                html_parts.append(f'<td class="detail-value">{_format_text_field(dose)}</td>')
                html_parts.append('</tr>')
                if coverage:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Coverage For:</td>')
                    html_parts.append(f'<td class="detail-value coverage">{_format_text_field(coverage)}</td>')
                    html_parts.append('</tr>')
                if renal:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Renal Adjustment:</td>')
                    html_parts.append(f'<td class="detail-value">{_format_text_field(renal)}</td>')
                    html_parts.append('</tr>')
                if considerations:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Considerations:</td>')
                    html_parts.append(f'<td class="detail-value">{_format_text_field(considerations)}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</table>')
                
                # Add references for this antibiotic
                sources = med.get('mentioned_in_sources', [])
                if sources and isinstance(sources, list) and len(sources) > 0:
                    html_parts.append('<div class="medication-references">')
                    html_parts.append('<div class="references-label">References:</div>')
                    for ref in sources:
                        if ref:
                            ref_escaped = html.escape(ref)
                            html_parts.append(f'<div class="reference-line"><a href="{ref_escaped}" class="reference-link">{ref_escaped}</a></div>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
                html_parts.append('</div>')
        
        # Process second_choice medications
        if second_choice:
            for idx, med in enumerate(second_choice):
                name = fix_text_encoding(med.get('medical_name', ''))
                route = fix_text_encoding(med.get('route_of_administration', ''))
                dose = fix_text_encoding(med.get('dose_duration', ''))
                coverage = fix_text_encoding(med.get('coverage_for', ''))
                renal = fix_text_encoding(med.get('renal_adjustment', ''))
                considerations = fix_text_encoding(med.get('general_considerations', ''))
                
                html_parts.append('<div class="medication-card">')
                # Show "Second Line" tag only on first medication of this category
                if idx == 0:
                    html_parts.append('<table class="medication-header"><tr>')
                    html_parts.append('<td class="medication-tag-section second-line-tag">Second Line</td>')
                    html_parts.append(f'<td class="medication-name-section"><span class="medication-name">{_format_text_field(name)}</span></td>')
                    html_parts.append('</tr></table>')
                else:
                    html_parts.append('<table class="medication-header"><tr>')
                    html_parts.append(f'<td class="medication-name-section full-width"><span class="medication-name">{_format_text_field(name)}</span></td>')
                    html_parts.append('</tr></table>')
                
                html_parts.append('<div class="medication-details">')
                html_parts.append('<table class="medication-table">')
                html_parts.append('<tr>')
                html_parts.append(f'<td class="detail-label">Route:</td>')
                html_parts.append(f'<td class="detail-value">{_format_text_field(route)}</td>')
                html_parts.append('</tr>')
                html_parts.append('<tr>')
                html_parts.append(f'<td class="detail-label">Dose:</td>')
                html_parts.append(f'<td class="detail-value">{_format_text_field(dose)}</td>')
                html_parts.append('</tr>')
                if coverage:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Coverage For:</td>')
                    html_parts.append(f'<td class="detail-value coverage">{_format_text_field(coverage)}</td>')
                    html_parts.append('</tr>')
                if renal:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Renal Adjustment:</td>')
                    html_parts.append(f'<td class="detail-value">{_format_text_field(renal)}</td>')
                    html_parts.append('</tr>')
                if considerations:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Considerations:</td>')
                    html_parts.append(f'<td class="detail-value">{_format_text_field(considerations)}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</table>')
                
                # Add references for this antibiotic
                sources = med.get('mentioned_in_sources', [])
                if sources and isinstance(sources, list) and len(sources) > 0:
                    html_parts.append('<div class="medication-references">')
                    html_parts.append('<div class="references-label">References:</div>')
                    for ref in sources:
                        if ref:
                            ref_escaped = html.escape(ref)
                            html_parts.append(f'<div class="reference-line"><a href="{ref_escaped}" class="reference-link">{ref_escaped}</a></div>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
                html_parts.append('</div>')
        
        # Process alternative_antibiotic medications
        if alternative:
            for idx, med in enumerate(alternative):
                name = fix_text_encoding(med.get('medical_name', ''))
                route = fix_text_encoding(med.get('route_of_administration', ''))
                dose = fix_text_encoding(med.get('dose_duration', ''))
                coverage = fix_text_encoding(med.get('coverage_for', ''))
                renal = fix_text_encoding(med.get('renal_adjustment', ''))
                considerations = fix_text_encoding(med.get('general_considerations', ''))
                
                html_parts.append('<div class="medication-card">')
                # Show "Alternate" tag only on first medication of this category
                if idx == 0:
                    html_parts.append('<table class="medication-header"><tr>')
                    html_parts.append('<td class="medication-tag-section alternate-tag">Alternate</td>')
                    html_parts.append(f'<td class="medication-name-section"><span class="medication-name">{_format_text_field(name)}</span></td>')
                    html_parts.append('</tr></table>')
                else:
                    html_parts.append('<table class="medication-header"><tr>')
                    html_parts.append(f'<td class="medication-name-section full-width"><span class="medication-name">{_format_text_field(name)}</span></td>')
                    html_parts.append('</tr></table>')
                
                html_parts.append('<div class="medication-details">')
                html_parts.append('<table class="medication-table">')
                html_parts.append('<tr>')
                html_parts.append(f'<td class="detail-label">Route:</td>')
                html_parts.append(f'<td class="detail-value">{_format_text_field(route)}</td>')
                html_parts.append('</tr>')
                html_parts.append('<tr>')
                html_parts.append(f'<td class="detail-label">Dose:</td>')
                html_parts.append(f'<td class="detail-value">{_format_text_field(dose)}</td>')
                html_parts.append('</tr>')
                if coverage:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Coverage For:</td>')
                    html_parts.append(f'<td class="detail-value coverage">{_format_text_field(coverage)}</td>')
                    html_parts.append('</tr>')
                if renal:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Renal Adjustment:</td>')
                    html_parts.append(f'<td class="detail-value">{_format_text_field(renal)}</td>')
                    html_parts.append('</tr>')
                if considerations:
                    html_parts.append('<tr>')
                    html_parts.append(f'<td class="detail-label">Considerations:</td>')
                    html_parts.append(f'<td class="detail-value">{_format_text_field(considerations)}</td>')
                    html_parts.append('</tr>')
                html_parts.append('</table>')
                
                # Add references for this antibiotic
                sources = med.get('mentioned_in_sources', [])
                if sources and isinstance(sources, list) and len(sources) > 0:
                    html_parts.append('<div class="medication-references">')
                    html_parts.append('<div class="references-label">References:</div>')
                    for ref in sources:
                        if ref:
                            ref_escaped = html.escape(ref)
                            html_parts.append(f'<div class="reference-line"><a href="{ref_escaped}" class="reference-link">{ref_escaped}</a></div>')
                    html_parts.append('</div>')
                
                html_parts.append('</div>')
                html_parts.append('</div>')
        
        html_parts.append('</div>')
    
    return '\n'.join(html_parts)


def _build_gene_html(gene_analysis_list: list) -> str:
    """Build HTML for resistance gene information."""
    if not gene_analysis_list:
        return ""
    
    html_parts = ['<div class="gene-section">']
    html_parts.append('<div class="section-header">RESISTANCE GENE INFORMATION</div>')
    
    from utils import fix_text_encoding
    for gene_analysis in gene_analysis_list:
        gene_name = fix_text_encoding(gene_analysis.get('detected_resistant_gene_name', ''))
        medication_classes = fix_text_encoding(gene_analysis.get('potential_medication_class_affected', ''))
        considerations = fix_text_encoding(gene_analysis.get('general_considerations', ''))
        
        html_parts.append('<div class="gene-info-card">')
        html_parts.append('<table class="gene-info-table">')
        html_parts.append('<tr>')
        html_parts.append('<td class="gene-label"><strong>Detected Resistance Gene:</strong></td>')
        html_parts.append(f'<td class="gene-value">{_format_text_field(gene_name)}</td>')
        html_parts.append('</tr>')
        html_parts.append('<tr>')
        html_parts.append('<td class="gene-label"><strong>Potential Medication Classes Affected:</strong></td>')
        html_parts.append(f'<td class="gene-value">{_format_text_field(medication_classes)}</td>')
        html_parts.append('</tr>')
        if considerations:
            html_parts.append('<tr>')
            html_parts.append('<td class="gene-label"><strong>General Considerations:</strong></td>')
            html_parts.append(f'<td class="gene-value">{_format_text_field(considerations)}</td>')
            html_parts.append('</tr>')
        html_parts.append('</table>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def _build_negative_sections_html(data: Dict[str, Any] = None) -> str:
    """Build HTML for negative organisms and genes sections using static lists."""
    negative_organisms = [
        'Acinetobacter baumannii', 'Bacteroides fragilis', 'Candida glabrata', 'Candida albicans',
        'Candida auris', 'Candida krusei', 'Candida lusitaniae', 'Candida parapsilosis',
        'Candida tropicalis', 'Citrobacter freundii', 'Clostridium novyi', 'Clostridium perfringens',
        'Clostridium septicum', 'Enterobacter cloacae', 'Enterococcus faecium', 'Escherichia coli',
        'Group A strep', 'Group B strep', 'Group C and G Strep', 'Herpes Zoster',
        'HSV-1 (Herpes Simplex)', 'HSV-2 (Herpes Simplex)', 'kingella kingae', 'Klebsiella aerogenes',
        'Klebsiella oxytoca', 'Klebsiella pneumoniae', 'Morganella morganii', 'Not an organism',
        'Proteus mirabilis', 'Proteus vulgaris', 'Pseudomonas aeruginosa', 'Trichophyton spp.'
    ]
    
    negative_genes = [
        'ampC', 'Ant-2', 'Aph 2', 'aph3', 'CTX-M1', 'CTX-M2', 'dfrA1', 'dfrA5',
        'Erm B', 'ErmA', 'femA', 'Gyrase A', 'KPC', 'mefA', 'NDM', 'OXA-48',
        'Par C', 'QnrA', 'QnrB', 'SHV', 'Sul 2', 'Sul1', 'TEM', 'Tet O',
        'tetB', 'vanA1', 'vanA2', 'VanB'
    ]
    
    html_parts = ['<div class="negative-sections">']
    
    # Negative organisms
    html_parts.append('<div class="negative-section">')
    html_parts.append('<div class="section-header">NEGATIVE ORGANISMS TESTED</div>')
    html_parts.append('<div class="negative-content">')
    for i, org in enumerate(negative_organisms):
        if i > 0:
            html_parts.append(', ')
        html_parts.append(_format_text_field(org))
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    # Negative genes
    html_parts.append('<div class="negative-section">')
    html_parts.append('<div class="section-header">NEGATIVE RESISTANCE GENES TESTED</div>')
    html_parts.append('<div class="negative-content">')
    for i, gene in enumerate(negative_genes):
        if i > 0:
            html_parts.append(', ')
        html_parts.append(_format_text_field(gene))
    html_parts.append('</div>')
    html_parts.append('</div>')
    
    html_parts.append('</div>')
    return '\n'.join(html_parts)


def _get_css_styles() -> str:
    """Get CSS styles for the PDF."""
    return """
        @page {
            size: letter;
            margin: 0.4in 0.4in 0.4in 0.4in;
        }
        
        body {
            font-family: Helvetica, Arial, sans-serif;
            font-size: 9pt;
            color: #000;
            line-height: 1.2;
            margin: 0;
            padding: 0;
        }
        
        /* Header Section */
        .header-section {
            margin-bottom: 12px;
        }
        
        .header-main {
            background-color: #4472C4;
            padding: 8px 0;
            text-align: center;
            margin-bottom: 8px;
        }
        
        .header-title {
            color: white;
            font-size: 16pt;
            font-weight: bold;
        }
        
        /* Patient Info Section */
        .patient-info-section {
            margin-bottom: 12px;
        }
        
        .patient-info-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 9pt;
        }
        
        .patient-info-table tr {
            border-bottom: 0.5pt solid #ddd;
        }
        
        .patient-info-table td {
            padding: 3px 4px;
            vertical-align: top;
        }
        
        .info-label {
            font-weight: bold;
            width: 1.2in;
            color: #333;
        }
        
        .info-value {
            width: 2.2in;
            color: #000;
        }
        
        /* Test Results Section */
        .test-results-section {
            margin-bottom: 15px;
        }
        
        .test-results-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 9pt;
        }
        
        .test-results-table tr {
            border-bottom: 0.5pt solid #ddd;
        }
        
        .test-results-table td {
            padding: 4px 6px;
            vertical-align: top;
        }
        
        .result-label {
            font-weight: bold;
            width: 1.5in;
            color: #333;
        }
        
        .result-value {
            color: #000;
        }
        
        /* Section Headers */
        .section-header {
            background-color: #4472C4;
            color: white;
            text-align: center;
            padding: 6px 0;
            font-size: 11pt;
            font-weight: bold;
            margin-bottom: 8px;
        }
        
        /* Medication Section */
        .medication-section {
            margin-bottom: 15px;
        }
        
        .medication-card {
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-bottom: 15px;
            overflow: hidden;
            text-align: left;
            min-height: 80px;
            width: 100%;
            box-sizing: border-box;
        }
        
        .medication-header {
            background-color: #ffffff;
            border-bottom: 1px solid #ccc;
            width: 100%;
            box-sizing: border-box;
            border-collapse: collapse;
            margin: 0;
            padding: 0;
            border-spacing: 0;
        }
        
        .medication-header td {
            vertical-align: middle;
            margin: 0;
        }
        
        .medication-tag-section {
            color: white;
            width: 20%;
            vertical-align: middle;
            text-align: center;
            font-size: 11pt;
            font-weight: bold;
            line-height: 1.3;
            padding: 4px !important;
            margin: 0;
        }
        
        .medication-tag-section.first-line-tag {
            background-color: #367FA9;
        }
        
        .medication-tag-section.second-line-tag {
            background-color: #5B9BD5;
        }
        
        .medication-tag-section.alternate-tag {
            background-color: #70AD47;
        }
        
        .medication-name-section {
            background-color: #ffffff;
            width: 80%;
            vertical-align: middle;
            padding: 4px !important;
            margin: 0;
        }
        
        .medication-name-section.full-width {
            width: 100%;
        }
        
        .medication-name {
            font-weight: bold;
            font-size: 13pt;
            color: #000;
            text-align: left;
            margin: 0;
            padding: 0;
            line-height: 1.4;
        }
        
        .medication-details {
            padding: 8px;
        }
        
        .medication-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .medication-table tr {
            border-bottom: 0.5pt solid #eee;
        }
        
        .medication-table td {
            padding: 3px 4px;
            vertical-align: top;
        }
        
        .detail-label {
            font-weight: bold;
            color: #666;
            width: 1in;
        }
        
        .detail-value {
            color: #000;
        }
        
        .detail-value.coverage {
            color: #FF0000;
            font-weight: bold;
        }
        
        /* Medication References */
        .medication-references {
            margin-top: 2px;
            padding-top: 2px;
            border-top: 0.5pt solid #eee;
        }
        
        .references-label {
            font-weight: bold;
            font-size: 8pt;
            color: #666;
            margin-bottom: 2px;
            display: block;
        }
        
        .reference-line {
            margin: 0;
            margin-bottom: 1px;
            color: #333;
            word-wrap: break-word;
            font-size: 7pt;
            line-height: 1.2;
            padding: 0;
            border: none;
            background: none;
        }
        
        .reference-link {
            color: #0066cc;
            text-decoration: none;
        }
        
        /* Gene Section */
        .gene-section {
            margin-bottom: 15px;
        }
        
        .gene-info-card {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            margin-bottom: 8px;
            background-color: #f9f9f9;
        }
        
        .gene-info-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .gene-info-table tr {
            border-bottom: 0.5pt solid #ddd;
        }
        
        .gene-info-table td {
            padding: 4px 6px;
            vertical-align: top;
        }
        
        .gene-label {
            font-weight: bold;
            color: #333;
            width: 2.5in;
        }
        
        .gene-value {
            color: #000;
        }
        
        /* Negative Sections */
        .negative-sections {
            margin-bottom: 15px;
        }
        
        .negative-section {
            margin-bottom: 12px;
        }
        
        .negative-content {
            border: 0.5pt solid #000;
            padding: 8px 10px;
            font-size: 8.5pt;
            line-height: 1.4;
            background-color: #fff;
        }
        
        /* Footer */
        .footer {
            font-size: 7.5pt;
            line-height: 1.3;
            margin-top: 20px;
            padding-top: 10px;
            border-top: 0.5pt solid #ccc;
            color: #666;
        }
        
        .footer p {
            margin: 0 0 4px 0;
        }
    """


def export_to_pdf(data: Dict[str, Any], output_path: Optional[str] = None, save_to_disk: Optional[bool] = None) -> tuple[str, BytesIO]:
    """
    Export data to PDF using xhtml2pdf (pisa) - Pure Python, no system dependencies.
    
    Args:
        data: Dictionary containing input_parameters and result
        output_path: Optional path for output file (only used if save_to_disk is True)
        save_to_disk: Whether to save PDF to disk. If None, uses config setting.
        
    Returns:
        Tuple of (pdf_path_or_name, pdf_bytes): 
        - pdf_path_or_name: Path if saved to disk, or filename if not saved
        - pdf_bytes: BytesIO object containing the PDF data
    """
    if not XHTML2PDF_AVAILABLE:
        raise ImportError(
            "xhtml2pdf (pisa) is not installed. Install with: pip install xhtml2pdf\n"
            "This is a pure Python library with no system dependencies."
        )
    
    # Check if we should save to disk
    if save_to_disk is None:
        from config import get_output_config
        output_config = get_output_config()
        save_to_disk = output_config.get('save_enabled', True)
    
    # Generate HTML content
    html_content = _create_html_template(data)
    
    # Create BytesIO buffer for PDF
    pdf_buffer = BytesIO()
    
    # Convert HTML string to PDF
    pisa_status = pisa.CreatePDF(
        BytesIO(html_content.encode('utf-8')),
        dest=pdf_buffer,
        encoding='utf-8'
    )
    
    # Check for errors
    if pisa_status.err:
        raise Exception(f"Error generating PDF: {pisa_status.err}")
    
    # Reset buffer position to beginning
    pdf_buffer.seek(0)
    
    # Optionally save to disk
    pdf_path_or_name = None
    if save_to_disk:
        # Generate output path if not provided
        if not output_path:
            from config import get_output_config
            output_config = get_output_config()
            output_dir = Path(output_config.get('directory', 'output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = str(output_dir / f"report_{timestamp}.pdf")
        
        # Save to disk
        try:
            with open(output_path, "wb") as f:
                # Copy buffer content to file
                pdf_buffer.seek(0)
                f.write(pdf_buffer.read())
                pdf_buffer.seek(0)  # Reset for return
            logger.info(f"PDF report saved to: {output_path}")
            pdf_path_or_name = output_path
        except Exception as e:
            logger.warning(f"Failed to save PDF to disk: {e}, but PDF is available for download")
            # Generate a filename for reference even if save failed
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            pdf_path_or_name = f"report_{timestamp}.pdf"
    else:
        # Just generate a filename for reference
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        pdf_path_or_name = f"report_{timestamp}.pdf"
        logger.info(f"PDF generated in memory (not saved to disk): {pdf_path_or_name}")
    
    return pdf_path_or_name, pdf_buffer