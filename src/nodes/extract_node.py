"""
Extract node for LangGraph - Extracts antibiotic therapy information from search results.
Uses LlamaIndex Pydantic program for structured extraction.
"""
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from schemas import SearchResult, CombinedExtractionResult
from prompts import EXTRACTION_PROMPT_TEMPLATE
from utils import (format_resistance_genes, get_icd_names_from_state, 
                   get_pathogens_from_input, format_pathogens, 
                   get_resistance_genes_from_input, create_llm, retry_with_max_attempts, RetryError)

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None


def _extract_with_llamaindex(
    content: str,
    pathogen_name: str,
    resistant_gene: str,
    severity_codes: str,
    age: Optional[int],
    sample: str,
    systemic: bool,
    pathogens: Optional[List[Dict[str, str]]] = None,
    resistant_genes_list: Optional[List[str]] = None,
    allergies: Optional[List[str]] = None,
    source_title: str = "",
    retry_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Extract using LlamaIndex Pydantic program with unlimited retry logic.
    
    Args:
        retry_delay: Initial delay between retries in seconds (default: 2.0)
    """
    llm = create_llm()
    if not llm:
        logger.error("LlamaIndex LLM not available")
        return _empty_result()
    
    # Build conditional resistance gene sections
    if resistant_gene:
        resistance_context = f" | Resistance: {resistant_gene}"
        resistance_task = f" with {resistant_gene} resistance"
        resistance_genes_section = f"""RESISTANCE GENES (for each in {resistant_gene}):
- detected_resistant_gene_name: Exact name ("mecA", "tetM", "dfrA", "Ant-la")
- potential_medication_class_affected: Classes affected. Look for: "beta-lactams", "penicillins", "cephalosporins", "tetracyclines", "trimethoprim", "aminoglycosides", "TMP-SMX". Infer from mechanism (e.g., "methicillin resistance" → beta-lactams) or specific drugs mentioned. Use knowledge: mecA→beta-lactams, tetM→tetracyclines, dfrA→trimethoprim/TMP-SMX, Ant-la→aminoglycosides. Use null only if no info exists.
- general_considerations: Mechanism/how gene confers resistance. Use null only if no mechanism info exists.

"""
        resistance_filtering_rule = f"""3. Filtering: DO NOT extract antibiotics ineffective due to {resistant_gene}. Example: mecA present → skip oxacillin, nafcillin, methicillin, cefazolin, other beta-lactams ineffective against MRSA. Only extract antibiotics retaining activity.
"""
    else:
        resistance_context = ""
        resistance_task = ""
        resistance_genes_section = ""
        resistance_filtering_rule = ""
    
    # Build conditional allergy sections
    from utils import format_allergies
    allergy_display = format_allergies(allergies) if allergies else None
    if allergy_display:
        allergy_context = f" | Allergies: {allergy_display}"
        allergy_filtering_rule = f"""4. Filtering: DO NOT extract antibiotics that patient is allergic to ({allergy_display}). Example: penicillin allergy → skip penicillins, amoxicillin, ampicillin, and consider cross-reactivity with cephalosporins. sulfa allergy → skip sulfonamides, TMP-SMX. Only extract antibiotics that are safe for the patient.
"""
    else:
        allergy_context = ""
        allergy_filtering_rule = ""
    
    # Format prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        pathogen_display=pathogen_name,
        resistance_context=resistance_context,
        resistance_task=resistance_task,
        allergy_context=allergy_context,
        severity_codes=severity_codes,
        age=f"{age} years" if age else 'Not specified',
        sample=sample or 'Not specified',
        systemic='Yes' if systemic else 'No',
        content=content,
        resistance_genes_section=resistance_genes_section,
        resistance_filtering_rule=resistance_filtering_rule,
        allergy_filtering_rule=allergy_filtering_rule
    )
    
    def _perform_extraction():
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=CombinedExtractionResult,
            llm=llm,
            prompt_template_str="{input_str}",
            verbose=False
        )
        result = program(input_str=prompt)
        if not result:
            return None
        result_dict = result.model_dump()
        return result_dict
    
    try:
        result_dict = retry_with_max_attempts(
            operation=_perform_extraction,
            operation_name=f"LLM extraction from {source_title}",
            max_attempts=5,
            retry_delay=retry_delay,
            should_retry_on_empty=True
        )
        
        # Post-process to fix is_combined, route extraction, and frequency conversion
        result_dict = _post_process_extraction_result(result_dict)
        
        # Log summary
        therapy = result_dict.get('antibiotic_therapy_plan', {})
        logger.info(
            f"Extracted from '{source_title[:50]}...': "
            f"first={len(therapy.get('first_choice', []))}, "
            f"second={len(therapy.get('second_choice', []))}, "
            f"alt={len(therapy.get('alternative_antibiotic', []))}, "
            f"genes={len(result_dict.get('pharmacist_analysis_on_resistant_gene', []))}"
        )
        
        return result_dict
        
    except RetryError as e:
        logger.error(f"Extraction failed after max attempts for {source_title}: {e}")
        raise


def _empty_result() -> Dict[str, Any]:
    """Return empty extraction result structure."""
    return {
        'antibiotic_therapy_plan': {
            'first_choice': [],
            'second_choice': [],
            'alternative_antibiotic': [],
            'not_known': []
        },
        'pharmacist_analysis_on_resistant_gene': []
    }


def _post_process_extraction_result(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process extraction results to set is_combined based on "plus" in medical_name.
    """
    therapy_plan = result_dict.get('antibiotic_therapy_plan', {})
    
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
        antibiotics = therapy_plan.get(category, [])
        if not isinstance(antibiotics, list):
            continue
            
        for ab in antibiotics:
            if not isinstance(ab, dict):
                continue
            
            medical_name = ab.get('medical_name', '')
            
            # Set is_combined based on "plus" in medical_name
            if medical_name and ' plus ' in medical_name.lower():
                ab['is_combined'] = True
            else:
                ab['is_combined'] = False
    
    return result_dict


def extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract antibiotic therapy and resistance gene information from search results.
    Uses LlamaIndex for structured extraction.
    """
    if not LLAMAINDEX_AVAILABLE:
        logger.error("LlamaIndex unavailable. Cannot perform extraction.")
        return {'source_results': []}
    
    try:
        search_results = state.get('search_results', [])
        input_params = state.get('input_parameters', {})
        
        if not search_results:
            logger.warning("No search results to extract from")
            return {'source_results': []}
        
        # Prepare context
        pathogens = get_pathogens_from_input(input_params)
        pathogen_display = format_pathogens(pathogens)
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistant_gene_display = format_resistance_genes(resistant_genes)
        from utils import get_allergies_from_input
        allergies = get_allergies_from_input(input_params)
        severity_codes = get_icd_names_from_state(state)
        age = input_params.get('age')
        sample = input_params.get('sample', '')
        systemic = input_params.get('systemic', True)
        
        def process_source(idx: int, result_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single search result."""
            result = SearchResult(**result_data)
            logger.info(f"[{idx}/{len(search_results)}] Processing: {result.title[:50]}...")
            
            # Add unique ID to prevent caching
            source_content = (
                f"Title: {result.title}\n"
                f"Content: {result.snippet}\n"
                f"[ID: {uuid.uuid4()}]"
            )
            
            # Extract with retry logic
            extraction = _extract_with_llamaindex(
                content=source_content,
                pathogen_name=pathogen_display,
                resistant_gene=resistant_gene_display,
                severity_codes=severity_codes,
                age=age,
                sample=sample,
                systemic=systemic,
                pathogens=pathogens,
                resistant_genes_list=resistant_genes,
                allergies=allergies,
                source_title=result.title
            )
            
            return {
                'source_url': result.url,
                'source_title': result.title,
                'source_index': idx,
                'antibiotic_therapy_plan': extraction.get('antibiotic_therapy_plan', {}),
                'pharmacist_analysis_on_resistant_gene': extraction.get(
                    'pharmacist_analysis_on_resistant_gene', []
                )
            }
        
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        total_sources = len(search_results)
        
        # Process concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_idx = {
                executor.submit(process_source, idx + 1, res): idx + 1
                for idx, res in enumerate(search_results)
            }
            
            results_dict = {}
            completed_count = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_dict[idx] = future.result()
                    completed_count += 1
                    
                    # Emit progress for each completed source
                    if progress_callback:
                        sub_progress = (completed_count / total_sources) * 100.0
                        progress_callback('extract', sub_progress, f'Extracted {completed_count}/{total_sources} sources')
                    
                except RetryError as e:
                    error_msg = f"Extraction failed for source {idx}: {e.operation_name} - {str(e)}"
                    logger.error(error_msg)
                    # Record error in state
                    if 'errors' not in state:
                        state['errors'] = []
                    state['errors'].append(error_msg)
                    # Return empty result for this source instead of stopping entire pipeline
                    return {
                        'source_url': result.url,
                        'source_title': result.title,
                        'source_index': idx,
                        'antibiotic_therapy_plan': {},
                        'pharmacist_analysis_on_resistant_gene': []
                    }
                except Exception as e:
                    logger.error(f"[{idx}] Processing error: {e}", exc_info=True)
                    completed_count += 1
                    
                    # Emit progress even on error
                    if progress_callback:
                        sub_progress = (completed_count / total_sources) * 100.0
                        progress_callback('extract', sub_progress, f'Processed {completed_count}/{total_sources} sources (error on {idx})')
                    
                    # Store empty result on error
                    results_dict[idx] = {
                        'source_url': search_results[idx-1].get('url', ''),
                        'source_title': search_results[idx-1].get('title', ''),
                        'source_index': idx,
                        **_empty_result()
                    }
        
        # Sort by index
        source_results = [results_dict[i] for i in sorted(results_dict.keys())]
        
        # Save results
        _save_extraction_results(input_params, source_results)
        
        return {'source_results': source_results}
        
    except Exception as e:
        logger.error(f"Error in extract_node: {e}", exc_info=True)
        raise


def _save_extraction_results(input_params: Dict, source_results: List[Dict]) -> None:
    """Save extraction results to file."""
    try:
        from config import get_output_config
        output_config = get_output_config()
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "extraction_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'input_parameters': input_params,
                'source_results': source_results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Extraction results saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save extraction results: {e}")