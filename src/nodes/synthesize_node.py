"""
Synthesize node for LangGraph - Groups antibiotics by name and unifies entries using LLM.
Uses LlamaIndex for structured extraction.
"""
import logging
import json
import time
from typing import Dict, Any, List, Optional
from collections import defaultdict

from schemas import UnifiedResistanceGenesResult, UnifiedAntibioticEntryForSynthesis
from prompts import ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE, RESISTANCE_GENE_UNIFICATION_PROMPT_TEMPLATE
from utils import format_resistance_genes, get_icd_names_from_state, create_llm, clean_null_strings

logger = logging.getLogger(__name__)

# LlamaIndex imports with fallback
try:
    from llama_index.core.program import LLMTextCompletionProgram
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    logger.error("LlamaIndex not available. Install: pip install llama-index llama-index-llms-ollama")
    LLAMAINDEX_AVAILABLE = False
    LLMTextCompletionProgram = None


def _normalize_antibiotic_name(name: str) -> str:
    """
    Normalize antibiotic name for comparison (case-insensitive, handle hyphens/dashes).
    
    Args:
        name: Antibiotic name to normalize
        
    Returns:
        Normalized name for comparison
    """
    if not name:
        return ""
    # Convert to lowercase, replace different dash types with standard hyphen
    normalized = name.lower().strip()
    normalized = normalized.replace('–', '-').replace('—', '-').replace('−', '-')
    return normalized




def _unify_antibiotic_group_with_llm(
    antibiotic_name: str,
    entries: List[Dict[str, Any]],
    retry_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Unify multiple entries of the same antibiotic using LLM.
    
    Args:
        antibiotic_name: The antibiotic name
        entries: List of entries for this antibiotic from different sources
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
    prompt = ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE.format(
        antibiotic_name=antibiotic_name,
        entries_list=entries_list
    )

    llm = create_llm()
    if not llm:
        logger.warning("LlamaIndex LLM not available, using first entry")
        entry = entries[0].copy()
        entry.pop('source_index', None)
        entry.pop('original_category', None)
        return entry
    
    attempt = 0
    while True:
        attempt += 1
        try:
            # Use LlamaIndex with schema from schemas.py
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=UnifiedAntibioticEntryForSynthesis,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            
            result = program(input_str=prompt)
            
            if not result:
                logger.warning(f"Empty result for {antibiotic_name} (attempt {attempt}), retrying...")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            continue
        
            result_dict = result.model_dump()
            
            unified = {
                'medical_name': result_dict.get('medical_name', antibiotic_name),
                'coverage_for': result_dict.get('coverage_for') if result_dict.get('coverage_for') else None,
                'route_of_administration': result_dict.get('route_of_administration') if result_dict.get('route_of_administration') else None,
                'dose_duration': result_dict.get('dose_duration') if result_dict.get('dose_duration') else None,
                'renal_adjustment': result_dict.get('renal_adjustment') if result_dict.get('renal_adjustment') else None,
                'general_considerations': result_dict.get('general_considerations') if result_dict.get('general_considerations') else None,
                'is_combined': result_dict.get('is_combined', False),
                'is_complete': result_dict.get('is_complete', False) 
            }
            
            # Ensure is_complete is always present
            if 'is_complete' not in unified:
                unified['is_complete'] = False
            
            # Clean null strings
            for key, value in unified.items():
                if isinstance(value, str) and value.lower() in ['null', 'none', 'not specified', '']:
                    unified[key] = None
            
            return unified
            
        except Exception as e:
            logger.warning(f"Error unifying {antibiotic_name} (attempt {attempt}): {e}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


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
    
    attempt = 0
    while True:
        attempt += 1
        try:
            # Use LlamaIndex with schema from schemas.py
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=UnifiedResistanceGenesResult,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            
            result = program(input_str=prompt)
            
            if not result:
                logger.warning(f"Empty result for resistance genes (attempt {attempt}), retrying...")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            result_dict = result.model_dump()
            resistance_genes = result_dict.get('resistance_genes', [])
            
            if resistance_genes:
                return [clean_null_strings(entry) for entry in resistance_genes]
            else:
                logger.warning(f"LLM returned no resistance genes (attempt {attempt}), retrying...")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
                
        except Exception as e:
            logger.warning(f"Error unifying resistance genes (attempt {attempt}): {e}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


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
        
        # Step 1: Group all antibiotics by normalized name across all sources and categories
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
                    normalized_name = _normalize_antibiotic_name(medical_name)
                    if not normalized_name:
                        continue
                    
                    # Add entry with source info
                    antibiotic_groups[normalized_name].append({
                        **antibiotic,
                        'source_index': source_index,
                        'original_category': category
                    })
            
        logger.info(f"Grouped {len(antibiotic_groups)} unique antibiotics from {total_sources} sources")
        
        # Step 2: Unify antibiotics - use LLM if count > 1, otherwise use as-is
        unified_antibiotics = []
        
        for normalized_name, entries in antibiotic_groups.items():
            # Get source indices for this antibiotic
            source_indices = sorted(set(e.get('source_index', 0) for e in entries if e.get('source_index', 0) > 0))
            
            # Unify if multiple entries
            if len(entries) > 1:
                logger.info(f"Unifying {len(entries)} entries for {normalized_name}")
                unified_entry = _unify_antibiotic_group_with_llm(
                    antibiotic_name=entries[0].get('medical_name', normalized_name),
                    entries=entries
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
        
        return {
            'result': result
        }
        
    except Exception as e:
        logger.error(f"Error in synthesize_node: {e}", exc_info=True)
        raise
