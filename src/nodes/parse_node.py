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
        source_results_raw = state.get('source_results', [])
        
        # Simply preserve all data as-is, no strict validation
        preserved_results = []
        for result_data in source_results_raw:
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
        
        # Add extraction date
        extraction_date = datetime.now()
        
        return {
            'source_results': preserved_results,
            'extraction_date': extraction_date.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in parse_node: {e}", exc_info=True)
        raise

