"""
Utility functions for the Molecular Pipeline.
"""
from typing import Optional


def fix_text_encoding(text: Optional[str]) -> str:
    """
    Fix encoding issues in text fields - replaces problematic Unicode characters
    that appear as squares in PDFs with standard ASCII equivalents.
    
    Args:
        text: Input text that may contain problematic Unicode characters
        
    Returns:
        Cleaned text with all problematic characters replaced
    """
    if not text:
        return ""
    
    # Comprehensive replacement of ALL problematic Unicode characters
    import re
    
    # Replace ALL Unicode dash/hyphen variants with standard ASCII hyphen
    # This catches en dash, em dash, horizontal bar, minus sign, and ALL variants
    dash_pattern = re.compile(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D\u2500-\u2501\u2E3A\u2E3B]')
    text = dash_pattern.sub('-', text)
    
    # Also replace common dash characters explicitly
    text = text.replace('–', '-')  # En dash
    text = text.replace('—', '-')  # Em dash
    text = text.replace('−', '-')  # Minus sign
    text = text.replace('‐', '-')  # Hyphen
    
    # Replace bullet points and special characters that might appear as squares
    bullet_pattern = re.compile(r'[\u2022\u25CF\u25E6\u2023\u2043\u2219\u25AA\u25AB]')
    text = bullet_pattern.sub('-', text)
    text = text.replace('•', '-')  # Bullet
    text = text.replace('·', '-')  # Middle dot
    
    # Replace problematic spaces
    space_pattern = re.compile(r'[\u00A0\u2000-\u200F\u2028-\u2029\uFEFF]')
    text = space_pattern.sub(' ', text)
    
    # Replace ellipsis
    text = text.replace('\u2026', '...')  # Horizontal ellipsis
    text = text.replace('…', '...')       # Ellipsis
    
    # Replace any box-drawing or geometric characters that might appear as squares
    # This includes box drawing characters (2500-257F) and geometric shapes (25A0-25FF)
    # Also catch the specific black square character (U+25A0)
    box_pattern = re.compile(r'[\u2500-\u257F\u25A0-\u25FF\u2580-\u259F]')
    text = box_pattern.sub('-', text)
    
    # Explicitly replace common square characters
    text = text.replace('\u25A0', '-')  # Black square
    text = text.replace('\u25A1', '-')  # White square
    text = text.replace('\u25AA', '-')  # Black small square
    text = text.replace('\u25AB', '-')  # White small square
    text = text.replace('■', '-')       # Black square (alternative)
    text = text.replace('□', '-')       # White square (alternative)
    
    # Replace any remaining non-printable or problematic characters
    # But preserve common Unicode characters like Greek letters (β, etc.)
    # Only remove control characters and truly problematic ones
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Clean up multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)
    
    return text.strip()


def format_resistance_genes(resistant_gene: Optional[str]) -> str:
    """
    Format resistance genes from comma-separated string to readable format.
    
    Args:
        resistant_gene: Comma-separated resistance genes (e.g., "vanA, mecA" or "vanA")
        
    Returns:
        Formatted string for use in prompts (e.g., "vanA and mecA" or "vanA")
    """
    if not resistant_gene or not resistant_gene.strip():
        return "unknown"
    
    # Split by comma and clean up
    genes = [gene.strip() for gene in resistant_gene.split(',')]
    genes = [gene for gene in genes if gene]  # Remove empty strings
    
    if not genes:
        return "unknown"
    
    if len(genes) == 1:
        return genes[0]
    elif len(genes) == 2:
        return f"{genes[0]} and {genes[1]}"
    else:
        # Format as "gene1, gene2, and gene3"
        return ", ".join(genes[:-1]) + f", and {genes[-1]}"


def format_icd_codes(severity_codes: Optional[str]) -> str:
    """
    Format ICD codes from comma-separated string to readable format.
    
    Args:
        severity_codes: Comma-separated ICD codes (e.g., "A41.9, B95.3" or "A41.9")
        
    Returns:
        Formatted string for use in prompts (e.g., "A41.9 and B95.3" or "A41.9")
    """
    if not severity_codes or not severity_codes.strip():
        return "not specified"
    
    # Split by comma and clean up
    codes = [code.strip().upper() for code in severity_codes.split(',')]
    codes = [code for code in codes if code]  # Remove empty strings
    
    if not codes:
        return "not specified"
    
    if len(codes) == 1:
        return codes[0]
    elif len(codes) == 2:
        return f"{codes[0]} and {codes[1]}"
    else:
        # Format as "code1, code2, and code3"
        return ", ".join(codes[:-1]) + f", and {codes[-1]}"

