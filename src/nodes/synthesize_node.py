"""
Synthesize node for LangGraph - Groups antibiotics by name and unifies entries using LLM.
Uses LlamaIndex for structured extraction.
"""
import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict

from schemas import UnifiedResistanceGenesResult, UnifiedAntibioticEntryForSynthesis
from prompts import ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE, RESISTANCE_GENE_UNIFICATION_PROMPT_TEMPLATE
from utils import format_resistance_genes, get_icd_names_from_state, create_llm, clean_null_strings, normalize_antibiotic_name, retry_with_max_attempts, RetryError

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None






def _unify_antibiotic_group_with_llm(
    antibiotic_name: str,
    entries: List[Dict[str, Any]],
    route_of_administration: str = '',
    retry_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Unify multiple entries of the same antibiotic with the same route using LLM.
    
    Args:
        antibiotic_name: The antibiotic name
        entries: List of entries for this antibiotic from different sources (all with same route)
        route_of_administration: The route of administration for this group
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        Unified antibiotic entry
    """
    if not entries:
        return {}
    
    # If only one entry, return it as-is (no unification needed)
    if len(entries) == 1:
        entry = entries[0].copy()
        # Remove source-specific fields
        entry.pop('source_index', None)
        entry.pop('original_category', None)
        return entry
    
    # Build prompt with all entries
    entries_text = []
    for i, entry in enumerate(entries, 1):
        source_idx = entry.get('source_index', i)
        original_category = entry.get('original_category', 'unknown')
        
        entries_text.append(
            f"Source {source_idx} (Category: {original_category}):\n"
            f"  medical_name: {entry.get('medical_name', 'null')}\n"
            f"  coverage_for: {entry.get('coverage_for') or 'null'}\n"
            f"  route_of_administration: {entry.get('route_of_administration') or 'null'}\n"
            f"  dose_duration: {entry.get('dose_duration') or 'null'}\n"
            f"  renal_adjustment: {entry.get('renal_adjustment') or 'null'}\n"
            f"  general_considerations: {entry.get('general_considerations') or 'null'}\n"
            f"  is_combined: {entry.get('is_combined', False)}"
        )
    
    entries_list = "\n\n".join(entries_text)
    
    # Use prompt template from prompts.py
    # All entries in this group have the same route
    route_display = route_of_administration if route_of_administration else 'null'
    prompt = ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE.format(
        antibiotic_name=antibiotic_name,
        route_of_administration=route_display,
        entries_list=entries_list
    )

    llm = create_llm()
    if not llm:
        logger.warning("LlamaIndex LLM not available, using first entry")
        entry = entries[0].copy()
        entry.pop('source_index', None)
        entry.pop('original_category', None)
        return entry
    
    def _perform_unification():
        # Use LlamaIndex with schema from schemas.py
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=UnifiedAntibioticEntryForSynthesis,
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
            operation=_perform_unification,
            operation_name=f"LLM unification for {antibiotic_name}",
            max_attempts=5,
            retry_delay=retry_delay,
            should_retry_on_empty=True
        )
        
        # Trust LLM's output completely - it will set dose_duration to null if incomplete per prompt
        unified = {
            'medical_name': result_dict.get('medical_name', antibiotic_name),
            'coverage_for': result_dict.get('coverage_for') if result_dict.get('coverage_for') else None,
            'route_of_administration': result_dict.get('route_of_administration') if result_dict.get('route_of_administration') else None,
            'dose_duration': result_dict.get('dose_duration') if result_dict.get('dose_duration') else None,
            'renal_adjustment': result_dict.get('renal_adjustment') if result_dict.get('renal_adjustment') else None,
            'general_considerations': result_dict.get('general_considerations') if result_dict.get('general_considerations') else None,
            'is_combined': result_dict.get('is_combined', False),
            'is_complete': result_dict.get('is_complete', False)  # LLM determines this based on prompt
        }
        
        # Ensure is_complete is always present
        if 'is_complete' not in unified:
            unified['is_complete'] = False
        
        # Clean null strings
        for key, value in unified.items():
            if isinstance(value, str) and value.lower() in ['null', 'none', 'not specified', '']:
                unified[key] = None
        
        # CRITICAL FALLBACK: If LLM returned null for dose_duration but entries have it, use the most complete one
        if unified.get('dose_duration') is None:
            available_dose_durations = [e.get('dose_duration') for e in entries if e.get('dose_duration')]
            if available_dose_durations:
                # Prioritize complete ones (with duration), then incomplete ones
                complete_ones = [d for d in available_dose_durations if 'for' in d.lower() or 'week' in d.lower() or 'day' in d.lower()]
                if complete_ones:
                    # Use the most comprehensive complete one
                    unified['dose_duration'] = max(complete_ones, key=lambda x: len(x))
                    logger.info(f"Fallback: Used dose_duration from entries for {antibiotic_name}: {unified['dose_duration']}")
                else:
                    # Use the longest incomplete one
                    unified['dose_duration'] = max(available_dose_durations, key=lambda x: len(x))
                    logger.info(f"Fallback: Used incomplete dose_duration from entries for {antibiotic_name}: {unified['dose_duration']}")
        
        return unified
        
    except RetryError as e:
        logger.error(f"Unification failed after max attempts for {antibiotic_name}: {e}")
        raise


def _determine_final_category(
    entries: List[Dict[str, Any]],
    total_sources: int
) -> str:
    """
    Determine final category based on category distribution in entries.
    
    Args:
        entries: List of entries for this antibiotic
        total_sources: Total number of sources
        
    Returns:
        Final category: 'first_choice', 'second_choice', or 'alternative_antibiotic'
    """
    category_counts = {
        'first_choice': 0,
        'second_choice': 0,
        'alternative_antibiotic': 0,
        'not_known': 0
    }
    
    for entry in entries:
        category = entry.get('original_category', 'not_known')
        if category in category_counts:
            category_counts[category] += 1
    
    # Calculate percentages (exclude not_known)
    total_valid = category_counts['first_choice'] + category_counts['second_choice'] + category_counts['alternative_antibiotic']
    
    if total_valid == 0:
        return 'alternative_antibiotic'
    
    first_pct = category_counts['first_choice'] / total_valid
    second_pct = category_counts['second_choice'] / total_valid
    alt_pct = category_counts['alternative_antibiotic'] / total_valid
    
    # Use weighted scoring (first_choice = 4, second_choice = 3, alternative = 2)
    weighted_scores = {
        'first_choice': category_counts['first_choice'] * 4.0,
        'second_choice': category_counts['second_choice'] * 3.0,
        'alternative_antibiotic': category_counts['alternative_antibiotic'] * 2.0
    }
    
    # Strong consensus (>60%)
    if first_pct > 0.6:
        return 'first_choice'
    elif second_pct > 0.6:
        return 'second_choice'
    elif alt_pct > 0.6:
        return 'alternative_antibiotic'
    
    # Weighted majority
    max_category = max(weighted_scores.items(), key=lambda x: x[1])[0]
    return max_category


def _unify_resistance_genes_with_llm(
    resistance_genes_data: List[Dict[str, Any]],
    retry_delay: float = 2.0
) -> List[Dict[str, Any]]:
    """
    Unify resistance genes using LLM - processes all genes in one go.
    
    Args:
        resistance_genes_data: List of resistance gene entries from different sources
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        List of unified resistance gene entries
    """
    if not resistance_genes_data:
        return []
    
    # Group by gene name (keep original names, don't normalize)
    gene_groups = defaultdict(list)
    for entry in resistance_genes_data:
        gene_name = entry.get('detected_resistant_gene_name', '').strip()
        if gene_name:
            gene_groups[gene_name].append(entry)
    
    # Build prompt with all genes grouped
    genes_text = []
    for gene_name, entries in gene_groups.items():
        entries_list = []
        for i, entry in enumerate(entries, 1):
            entries_list.append(
                f"Source {i}:\n"
                f"  detected_resistant_gene_name: {entry.get('detected_resistant_gene_name', 'null')}\n"
                f"  potential_medication_class_affected: {entry.get('potential_medication_class_affected') or 'null'}\n"
                f"  general_considerations: {entry.get('general_considerations') or 'null'}"
            )
        genes_text.append(f"{gene_name}:\n" + "\n\n".join(entries_list))
    
    genes_list = "\n\n".join(genes_text)
    
    # Use prompt template from prompts.py
    prompt = RESISTANCE_GENE_UNIFICATION_PROMPT_TEMPLATE.format(
        genes_list=genes_list
    )

    llm = create_llm()
    if not llm:
        logger.warning("LlamaIndex LLM not available, using first entry from each group")
        unified_genes = []
        for gene_name, entries in gene_groups.items():
            if entries:
                unified_genes.append(entries[0].copy())
        return unified_genes
    
    def _perform_gene_unification():
        # Use LlamaIndex with schema from schemas.py
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=UnifiedResistanceGenesResult,
            llm=llm,
            prompt_template_str="{input_str}",
            verbose=False
        )
        result = program(input_str=prompt)
        if not result:
            return None
        result_dict = result.model_dump()
        resistance_genes = result_dict.get('resistance_genes', [])
        if not resistance_genes:
            return []  # Empty list is valid, but we'll retry if needed
        return [clean_null_strings(entry) for entry in resistance_genes]
    
    try:
        unified_genes = retry_with_max_attempts(
            operation=_perform_gene_unification,
            operation_name="LLM resistance gene unification",
            max_attempts=5,
            retry_delay=retry_delay,
            should_retry_on_empty=True
        )
        return unified_genes
        
    except RetryError as e:
        logger.error(f"Resistance gene unification failed after max attempts: {e}")
        raise


def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize node that groups antibiotics by name and unifies entries using LLM.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with result
    """
    try:
        source_results = state.get('source_results', [])
        
        if not source_results:
            logger.warning("No source results to synthesize")
            return {
                'result': {
                    'antibiotic_therapy_plan': {
                        'first_choice': [],
                        'second_choice': [],
                        'alternative_antibiotic': []
                    },
                    'pharmacist_analysis_on_resistant_gene': []
                }
            }
        
        total_sources = len(source_results)
        
        antibiotic_groups = defaultdict(list)
        
        for source_result in source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            source_index = source_result.get('source_index', 0)
            
            # Process all categories
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                antibiotics = therapy_plan.get(category, [])
                if not isinstance(antibiotics, list):
                    continue
                
                for antibiotic in antibiotics:
                    if not isinstance(antibiotic, dict):
                        continue
                    
                    medical_name = antibiotic.get('medical_name', '').strip()
                    if not medical_name:
                        continue
                    
                    # Normalize name for grouping
                    normalized_name = normalize_antibiotic_name(medical_name)
                    if not normalized_name:
                        continue
                    
                    # Get route_of_administration for grouping key
                    route = antibiotic.get('route_of_administration', '').strip() if antibiotic.get('route_of_administration') else ''
                    # Normalize route (handle None, empty strings, etc.)
                    route_key = route if route else 'null'
                    
                    # Create grouping key: (normalized_name, route)
                    # This ensures same drug with different routes are in different groups
                    group_key = (normalized_name, route_key)
                    
                    # Add entry with source info
                    antibiotic_groups[group_key].append({
                        **antibiotic,
                        'source_index': source_index,
                        'original_category': category
                    })
            
        logger.info(f"Grouped {len(antibiotic_groups)} unique antibiotics from {total_sources} sources")
        
        # Get progress callback from metadata
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        
        # Step 2: Unify antibiotics - use LLM if count > 1, otherwise use as-is
        unified_antibiotics = []
        total_antibiotics = len(antibiotic_groups)
        unified_count = 0
        
        for group_key, entries in antibiotic_groups.items():
            normalized_name, route_key = group_key
            # Get source indices for this antibiotic
            source_indices = sorted(set(e.get('source_index', 0) for e in entries if e.get('source_index', 0) > 0))
            
            # Get the route for this group (all entries should have the same route)
            route = entries[0].get('route_of_administration', '') if entries else ''
            
            # Unify if multiple entries
            if len(entries) > 1:
                logger.info(f"Unifying {len(entries)} entries for {normalized_name} with route {route}")
                unified_entry = _unify_antibiotic_group_with_llm(
                    antibiotic_name=entries[0].get('medical_name', normalized_name),
                    entries=entries,
                    route_of_administration=route
                )
            else:
                # Single entry - use as-is
                unified_entry = entries[0].copy()
                unified_entry.pop('source_index', None)
                unified_entry.pop('original_category', None)
                # Ensure is_complete is always present (default to False if not set)
                if 'is_complete' not in unified_entry:
                    unified_entry['is_complete'] = False
            
            # Determine final category based on category distribution
            final_category = _determine_final_category(entries, total_sources)
            
            # Add source information
            unified_entry['mentioned_in_sources'] = source_indices
            
            # Add to appropriate category
            unified_entry['final_category'] = final_category
            unified_antibiotics.append(unified_entry)
            
            # Emit progress for this antibiotic
            if progress_callback and total_antibiotics > 0:
                unified_count += 1
                sub_progress = (unified_count / total_antibiotics) * 100.0
                progress_callback('synthesize', sub_progress, f'Unified {unified_count}/{total_antibiotics} antibiotics')
        
        # Step 3: Organize into categories
        result_categories = {
            'first_choice': [],
            'second_choice': [],
            'alternative_antibiotic': []
        }
        
        for ab in unified_antibiotics:
            category = ab.pop('final_category', 'alternative_antibiotic')
            # Ensure is_complete is always present in every entry
            if 'is_complete' not in ab:
                ab['is_complete'] = False
            if category in result_categories:
                result_categories[category].append(ab)
        
        # Step 4: Process resistance genes
        resistance_genes_all = []
        for source_result in source_results:
            resistance_genes = source_result.get('pharmacist_analysis_on_resistant_gene', [])
            if isinstance(resistance_genes, list):
                resistance_genes_all.extend(resistance_genes)
        
        unified_resistance_genes = []
        if resistance_genes_all:
            logger.info(f"Unifying {len(resistance_genes_all)} resistance gene entries")
            unified_resistance_genes = _unify_resistance_genes_with_llm(resistance_genes_all)
        
        # Step 5: Convert source indices to URLs
        source_index_to_url = {}
        for source_result in source_results:
            source_index = source_result.get('source_index', 0)
            source_url = source_result.get('source_url', '')
            if source_index > 0 and source_url:
                source_index_to_url[source_index] = source_url
        
        def convert_indices_to_urls(antibiotics_list):
            for ab in antibiotics_list:
                if 'mentioned_in_sources' in ab:
                    ab['mentioned_in_sources'] = [
                        source_index_to_url.get(idx, '') 
                        for idx in ab['mentioned_in_sources'] 
                        if source_index_to_url.get(idx, '')
                    ]
        
        convert_indices_to_urls(result_categories['first_choice'])
        convert_indices_to_urls(result_categories['second_choice'])
        convert_indices_to_urls(result_categories['alternative_antibiotic'])
        
        # Build result
        result = {
            'antibiotic_therapy_plan': {
                'first_choice': result_categories['first_choice'],
                'second_choice': result_categories['second_choice'],
                'alternative_antibiotic': result_categories['alternative_antibiotic']
            },
            'pharmacist_analysis_on_resistant_gene': unified_resistance_genes
        }
        
        logger.info(
            f"Synthesized {len(unified_antibiotics)} antibiotics: "
            f"{len(result_categories['first_choice'])} first_choice, "
            f"{len(result_categories['second_choice'])} second_choice, "
            f"{len(result_categories['alternative_antibiotic'])} alternative"
        )
        
        # Save results to file
        input_params = state.get('input_parameters', {})
        icd_transformation = state.get('icd_transformation', {})
        _save_synthesize_results(input_params, result, icd_transformation)
        
        return {
            'result': result
        }
        
    except RetryError as e:
        error_msg = f"Synthesize node failed: {e.operation_name} - {str(e)}"
        logger.error(error_msg)
        # Record error in state and stop pipeline
        errors = state.get('errors', [])
        errors.append(error_msg)
        raise Exception(error_msg) from e
    except Exception as e:
        logger.error(f"Error in synthesize_node: {e}", exc_info=True)
        # Record error in state
        errors = state.get('errors', [])
        errors.append(f"Synthesize node error: {str(e)}")
        raise


def _save_synthesize_results(input_params: Dict, result: Dict, icd_transformation: Dict = None) -> None:
    """Save synthesize results to file."""
    try:
        from config import get_output_config
        output_config = get_output_config()
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "synthesize_result.json"
        
        output_data = {
            'input_parameters': input_params,
            'result': result
        }
        
        # Include ICD transformation with formatted codes (Code (Name))
        if icd_transformation:
            from utils import get_icd_names_from_state
            # Create a temporary state dict to get formatted ICD codes
            temp_state = {'icd_transformation': icd_transformation}
            icd_codes_formatted = get_icd_names_from_state(temp_state)
            output_data['icd_transformation'] = {
                **icd_transformation,
                'icd_codes_formatted': icd_codes_formatted  # Add formatted string with names
            }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Synthesize results saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save synthesize results: {e}")
