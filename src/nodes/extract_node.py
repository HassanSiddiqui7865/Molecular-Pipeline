"""
Extract node for LangGraph - Extracts antibiotic therapy information from search results.
Uses LlamaIndex Pydantic program for structured extraction.
"""
import json
import logging
import hashlib
import copy
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from schemas import SearchResult, CombinedExtractionResult
from prompts import EXTRACTION_PROMPT_TEMPLATE
from utils import (format_resistance_genes, get_icd_names_from_state, 
                   get_pathogens_from_input, format_pathogens, 
                   get_resistance_genes_from_input, create_llm, retry_with_max_attempts, RetryError,
                   chunk_text_custom, _get_overlap_text, convert_json_to_toon, call_llm_program)

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
    resistant_gene: Optional[str],
    severity_codes: str,
    age: Optional[int],
    panel: str,
    systemic: Optional[bool],
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
    
    # Chunk the content for processing based on tokens (GPT-OSS 20B: ~4k-8k tokens typical)
    chunk_size_tokens = 2500  # Target 2.5k tokens per chunk for extraction
    chunk_overlap_tokens = 200  # 200 tokens overlap for context
    chunks = chunk_text_custom(content, chunk_size_tokens=chunk_size_tokens, chunk_overlap_tokens=chunk_overlap_tokens)
    
    from utils import _get_token_count
    total_tokens = _get_token_count(content)
    logger.info(f"Processing {len(chunks)} chunks for extraction from '{source_title[:50]}...' (total: {total_tokens} tokens, {len(content)} chars)")
    
    # Accumulate results from all chunks
    all_antibiotics = {
        'first_choice': [],
        'second_choice': [],
        'alternative_antibiotic': []
    }
    all_resistance_genes = []
    
    # Track previous chunk text for overlap context
    previous_chunk_text = ""
    
    # Each chunk depends on previous chunks' extractions being merged into all_antibiotics/all_resistance_genes.
    # Chunk 1's results are merged before chunk 2 is processed, so chunk 2 can see and enhance chunk 1's data.
    for i, chunk in enumerate(chunks):
        def _process_chunk():
            # Build cross-chunk context from merged data (more accurate than raw chunks)
            # Only show context if we actually have extracted data (not for first chunk)
            cross_chunk_context = ""
            has_extracted_data = (
                all_antibiotics.get('first_choice') or 
                all_antibiotics.get('second_choice') or 
                all_antibiotics.get('alternative_antibiotic') or 
                all_resistance_genes
            )
            
            if has_extracted_data:
                # Build from merged data (already deduplicated and enhanced)
                merged_extraction = {
                    'antibiotic_therapy_plan': {
                        'first_choice': copy.deepcopy(all_antibiotics['first_choice']),
                        'second_choice': copy.deepcopy(all_antibiotics['second_choice']),
                        'alternative_antibiotic': copy.deepcopy(all_antibiotics['alternative_antibiotic'])
                    },
                    'pharmacist_analysis_on_resistant_gene': copy.deepcopy(all_resistance_genes)
                }
                
                # Convert merged extraction to TOON format (raises exception if TOON not available)
                toon_format = convert_json_to_toon(merged_extraction)
                
                cross_chunk_context = f"\n\n```toon\n{toon_format}\n```\n\n"
            
            # Add overlap text from previous chunk for context (if not first chunk)
            overlap_context = ""
            if i > 0 and previous_chunk_text:
                overlap_text = _get_overlap_text(previous_chunk_text, chunk_overlap_tokens)
                if overlap_text:
                    overlap_context = f"\n\n{overlap_text}\n\n"
            
            # Format chunk info
            chunk_info = f" (chunk {i+1} of {len(chunks)})" if len(chunks) > 1 else ""
            
            # Format prompt with chunk, overlap context, and cross-chunk context
            prompt = EXTRACTION_PROMPT_TEMPLATE.format(
                pathogen_display=pathogen_name,
                resistance_context=resistance_context,
                resistance_task=resistance_task,
                allergy_context=allergy_context,
                severity_codes=severity_codes,
                age=f"{age} years" if age else 'Not specified',
                panel=panel or 'Not specified',
                content=overlap_context + chunk if overlap_context else chunk,
                chunk_info=chunk_info,
                cross_chunk_context=cross_chunk_context,
                resistance_genes_section=resistance_genes_section,
                resistance_filtering_rule=resistance_filtering_rule,
                allergy_filtering_rule=allergy_filtering_rule
            )
            
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=CombinedExtractionResult,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            
            # Use logging wrapper for LLM call
            metadata = {
                'chunk_number': i + 1,
                'total_chunks': len(chunks),
                'source_title': source_title[:100] if source_title else '',
                'operation': 'extraction'
            }
            result = call_llm_program(program, prompt, operation_name=f"Extraction chunk {i+1}/{len(chunks)}", metadata=metadata)
            
            if not result:
                return None
            result_dict = result.model_dump()
            return result_dict
        
        try:
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} SEQUENTIALLY for '{source_title[:50]}...'")
            # Process chunk synchronously - wait for completion before moving to next chunk
            chunk_result = retry_with_max_attempts(
                operation=_process_chunk,
                operation_name=f"LLM extraction from chunk {i+1} of {source_title}",
                max_attempts=5,
                retry_delay=retry_delay,
                should_retry_on_empty=True
            )
            
            if chunk_result:
                # Store this chunk's text for overlap context in next chunk
                previous_chunk_text = chunk
                
                # Merge results into all_antibiotics/all_resistance_genes BEFORE processing next chunk.
                # This ensures chunk 2 can see and enhance chunk 1's extractions.
                # Merge across ALL categories first, then organize by category
                # This prevents duplicates when same antibiotic appears in different categories
                therapy_plan = chunk_result.get('antibiotic_therapy_plan', {})
                
                # Collect all new antibiotics from all categories
                all_new_antibiotics = []
                for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                    antibiotics = therapy_plan.get(category, [])
                    if isinstance(antibiotics, list):
                        for ab in antibiotics:
                            if isinstance(ab, dict) and ab.get('medical_name'):
                                all_new_antibiotics.append((ab, category))
                
                # Now merge each new antibiotic with existing ones (checking ALL categories)
                for new_ab, new_category in all_new_antibiotics:
                    if not isinstance(new_ab, dict):
                        continue
                    
                    new_name = new_ab.get('medical_name', '').strip()
                    new_route = new_ab.get('route_of_administration', '').strip() if new_ab.get('route_of_administration') else ''
                    
                    if not new_name:
                        continue
                    
                    # Check if we already have this antibiotic (same name and route) in ANY category
                    existing_ab = None
                    existing_category = None
                    for cat in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                        for existing in all_antibiotics[cat]:
                            if isinstance(existing, dict):
                                existing_name = existing.get('medical_name', '').strip()
                                existing_route = existing.get('route_of_administration', '').strip() if existing.get('route_of_administration') else ''
                                if existing_name == new_name and existing_route == new_route:
                                    existing_ab = existing
                                    existing_category = cat
                                    break
                        if existing_ab:
                            break
                    
                    if existing_ab:
                        # Enhance existing entry with new information
                        # Replace null fields with new values, merge non-null fields
                        
                        if not existing_ab.get('dose_duration') and new_ab.get('dose_duration'):
                            existing_ab['dose_duration'] = new_ab['dose_duration']
                        
                        if not existing_ab.get('renal_adjustment') and new_ab.get('renal_adjustment'):
                            existing_ab['renal_adjustment'] = new_ab['renal_adjustment']
                        
                        if not existing_ab.get('general_considerations') and new_ab.get('general_considerations'):
                            existing_ab['general_considerations'] = new_ab['general_considerations']
                        elif existing_ab.get('general_considerations') and new_ab.get('general_considerations'):
                            existing_cons = existing_ab['general_considerations']
                            new_cons = new_ab['general_considerations']
                            if new_cons not in existing_cons and len(existing_cons + '; ' + new_cons) <= 300:
                                existing_ab['general_considerations'] = f"{existing_cons}; {new_cons}"
                        
                        if not existing_ab.get('coverage_for') and new_ab.get('coverage_for'):
                            existing_ab['coverage_for'] = new_ab['coverage_for']
                        
                        # If new category is higher priority, move to that category
                        category_priority = {'first_choice': 3, 'second_choice': 2, 'alternative_antibiotic': 1}
                        if category_priority.get(new_category, 0) > category_priority.get(existing_category, 0):
                            all_antibiotics[existing_category].remove(existing_ab)
                            all_antibiotics[new_category].append(existing_ab)
                    else:
                        # New antibiotic, add it to the appropriate category
                        all_antibiotics[new_category].append(new_ab)
                
                # Merge resistance genes - avoid duplicates
                resistance_genes = chunk_result.get('pharmacist_analysis_on_resistant_gene', [])
                if isinstance(resistance_genes, list):
                    for new_gene in resistance_genes:
                        if not isinstance(new_gene, dict):
                            continue
                        
                        new_gene_name = new_gene.get('detected_resistant_gene_name', '').strip()
                        if not new_gene_name:
                            continue
                        
                        # Check if we already have this gene
                        existing_gene = next(
                            (g for g in all_resistance_genes 
                             if isinstance(g, dict) and g.get('detected_resistant_gene_name', '').strip() == new_gene_name),
                            None
                        )
                        
                        if existing_gene:
                            # Enhance existing gene with new information
                            if not existing_gene.get('potential_medication_class_affected') and new_gene.get('potential_medication_class_affected'):
                                existing_gene['potential_medication_class_affected'] = new_gene['potential_medication_class_affected']
                            if not existing_gene.get('general_considerations') and new_gene.get('general_considerations'):
                                existing_gene['general_considerations'] = new_gene['general_considerations']
                            elif existing_gene.get('general_considerations') and new_gene.get('general_considerations'):
                                # Merge considerations
                                existing_cons = existing_gene['general_considerations']
                                new_cons = new_gene['general_considerations']
                                if new_cons not in existing_cons and len(existing_cons + '; ' + new_cons) <= 200:
                                    existing_gene['general_considerations'] = f"{existing_cons}; {new_cons}"
                        else:
                            # New gene, add it
                            all_resistance_genes.append(new_gene)
        
        except RetryError as e:
            logger.warning(f"Chunk {i+1} extraction failed for {source_title}: {e}, continuing with other chunks")
            continue
    
    # Merge results from all chunks
    result_dict = {
        'antibiotic_therapy_plan': all_antibiotics,
        'pharmacist_analysis_on_resistant_gene': all_resistance_genes
    }
    
    # Post-process to fix is_combined, route extraction, and frequency conversion
    result_dict = _post_process_extraction_result(result_dict)
    
    # Log summary
    therapy = result_dict.get('antibiotic_therapy_plan', {})
    logger.info(
        f"Extracted from '{source_title[:50]}...' ({len(chunks)} chunks): "
        f"first={len(therapy.get('first_choice', []))}, "
        f"second={len(therapy.get('second_choice', []))}, "
        f"alt={len(therapy.get('alternative_antibiotic', []))}, "
        f"genes={len(result_dict.get('pharmacist_analysis_on_resistant_gene', []))}"
    )
    
    return result_dict


def _empty_result() -> Dict[str, Any]:
    """Return empty extraction result structure."""
    return {
        'antibiotic_therapy_plan': {
            'first_choice': [],
            'second_choice': [],
            'alternative_antibiotic': []
        },
        'pharmacist_analysis_on_resistant_gene': []
    }


def _post_process_extraction_result(result_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Post-process extraction results to set is_combined based on "plus" in medical_name.
    """
    therapy_plan = result_dict.get('antibiotic_therapy_plan', {})
    
    for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
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
        panel = input_params.get('panel', '')
        systemic = input_params.get('systemic')
        
        def process_source(idx: int, result_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single search result."""
            result = SearchResult(**result_data)
            logger.info(f"[{idx}/{len(search_results)}] Processing: {result.title[:50]}...")
            
            # Use deterministic ID based on source content (hash of title + snippet)
            # This ensures consistent extraction while still preventing caching issues
            content_hash = hashlib.md5(f"{result.title}|{result.snippet}".encode('utf-8')).hexdigest()[:8]
            source_content = (
                f"Title: {result.title}\n"
                f"Content: {result.snippet}\n"
                f"[ID: {content_hash}]"
            )
            
            # Extract with retry logic
            extraction = _extract_with_llamaindex(
                content=source_content,
                pathogen_name=pathogen_display,
                resistant_gene=resistant_gene_display,
                severity_codes=severity_codes,
                age=age,
                panel=panel,
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
                    # Get search result data for this index
                    result_data = search_results[idx - 1]
                    result = SearchResult(**result_data) if isinstance(result_data, dict) else result_data
                    # Return empty result for this source instead of stopping entire pipeline
                    results_dict[idx] = {
                        'source_url': result.url,
                        'source_title': result.title,
                        'source_index': idx,
                        'antibiotic_therapy_plan': {},
                        'pharmacist_analysis_on_resistant_gene': []
                    }
                    completed_count += 1
                    
                    # Emit progress even on error
                    if progress_callback:
                        sub_progress = (completed_count / total_sources) * 100.0
                        progress_callback('extract', sub_progress, f'Processed {completed_count}/{total_sources} sources (error on {idx})')
                    continue
                except Exception as e:
                    logger.error(f"[{idx}] Processing error: {e}", exc_info=True)
                    completed_count += 1
                    
                    # Emit progress even on error
                    if progress_callback:
                        sub_progress = (completed_count / total_sources) * 100.0
                        progress_callback('extract', sub_progress, f'Processed {completed_count}/{total_sources} sources (error on {idx})')
                    
                    # Get search result data for this index
                    result_data = search_results[idx - 1]
                    result = SearchResult(**result_data) if isinstance(result_data, dict) else result_data
                    # Store empty result on error
                    results_dict[idx] = {
                        'source_url': result.url,
                        'source_title': result.title,
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
        
        # Check if saving is enabled
        if not output_config.get('save_enabled', True):
            logger.debug("Saving extraction results disabled (production mode)")
            return
        
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