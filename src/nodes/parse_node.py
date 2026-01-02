"""
Parse node for LangGraph - Validates and structures extracted data.
"""
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


def parse_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that preserves extracted data without strict validation.
    Simply passes through the data structure from extraction.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with source_results and extraction dates
    """
    try:
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        
        # Emit progress for parsing start
        if progress_callback:
            progress_callback('parse', 0, 'Parsing extracted data...')
        
        source_results_raw = state.get('source_results', [])
        
        # Simply preserve all data as-is, no strict validation
        preserved_results = []
        total_sources = len(source_results_raw)
        
        for idx, result_data in enumerate(source_results_raw):
            # Preserve the data structure exactly as extracted
            preserved_result = {
                'source_url': result_data.get('source_url', ''),
                'source_title': result_data.get('source_title', ''),
                'source_index': result_data.get('source_index', 0),
                'antibiotic_therapy_plan': result_data.get('antibiotic_therapy_plan', {
                    'first_choice': [],
                    'second_choice': [],
                    'alternative_antibiotic': []
                }),
                'pharmacist_analysis_on_resistant_gene': result_data.get('pharmacist_analysis_on_resistant_gene', [])
            }
            preserved_results.append(preserved_result)
            
            # Emit progress for each source parsed
            if progress_callback and total_sources > 0:
                sub_progress = ((idx + 1) / total_sources) * 100.0
                progress_callback('parse', sub_progress, f'Parsed {idx + 1}/{total_sources} sources')
        
        # Add extraction date
        extraction_date = datetime.now()
        
        # Emit progress for parsing complete
        if progress_callback:
            progress_callback('parse', 100, f'Parsing complete ({len(preserved_results)} sources)')
        
        return {
            'source_results': preserved_results,
            'extraction_date': extraction_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in parse_node: {e}", exc_info=True)
        raise

