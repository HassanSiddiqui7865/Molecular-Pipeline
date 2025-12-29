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
                   get_resistance_genes_from_input)
from config import get_ollama_config

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    from llama_index.llms.ollama import Ollama
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None
    Ollama = None


def _create_llm() -> Optional[Ollama]:
    """Create LlamaIndex Ollama LLM instance."""
    if not LLAMAINDEX_AVAILABLE:
        return None
    
    try:
        config = get_ollama_config()
        return Ollama(
            model=config['model'].replace('ollama/', ''),
            base_url=config['api_base'],
            temperature=config['temperature'],
            request_timeout=600.0
        )
    except Exception as e:
        logger.error(f"Failed to create Ollama LLM: {e}")
        return None


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
    source_title: str = "",
    retry_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Extract using LlamaIndex Pydantic program with unlimited retry logic.
    
    Args:
        retry_delay: Initial delay between retries in seconds (default: 2.0)
    """
    llm = _create_llm()
    if not llm:
        logger.error("LlamaIndex LLM not available")
        return _empty_result()
    
    # Format prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        pathogen_display=pathogen_name,
        resistant_gene=resistant_gene,
        severity_codes=severity_codes,
        age=f"{age} years" if age else 'Not specified',
        sample=sample or 'Not specified',
        systemic='Yes' if systemic else 'No',
        content=content
    )
    
    attempt = 0
    while True:
        attempt += 1
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=CombinedExtractionResult,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            
            result = program(input_str=prompt)
            
            if not result:
                logger.warning(f"Empty result from {source_title} (attempt {attempt}), retrying...")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            result_dict = result.model_dump()
            
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
            
        except Exception as e:
            logger.warning(
                f"Extraction error from {source_title} (attempt {attempt}): {e}"
            )
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


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
        
        # Process concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_idx = {
                executor.submit(process_source, idx + 1, res): idx + 1
                for idx, res in enumerate(search_results)
            }
            
            results_dict = {}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results_dict[idx] = future.result()
                except Exception as e:
                    logger.error(f"[{idx}] Processing error: {e}", exc_info=True)
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