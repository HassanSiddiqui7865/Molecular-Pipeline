"""
Synthesize node for LangGraph - Aggregates and combines results from all sources using LangChain.
"""
import logging
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict

from langchain_core.language_models.chat_models import BaseChatModel
from schemas import UnifiedResistanceGene, UnifiedResistanceGenesResult
from utils import format_resistance_genes, format_icd_codes

logger = logging.getLogger(__name__)


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


def _clean_null_strings(data: Any) -> Any:
    """Recursively convert string 'null' values to actual None/null."""
    if isinstance(data, dict):
        return {k: _clean_null_strings(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_clean_null_strings(item) for item in data]
    elif isinstance(data, str) and data.lower() == 'null':
        return None
    else:
        return data


def _format_resistance_genes_as_toon(resistance_genes_data: List[Dict[str, Any]]) -> str:
    """
    Format resistance genes data as TOON string to reduce token usage.
    
    Args:
        resistance_genes_data: List of resistance gene dictionaries
        
    Returns:
        TOON string representation
    """
    try:
        from py_toon_format import encode
        return encode(resistance_genes_data)
    except ImportError:
        # Fallback to JSON if TOON library not available
        logger.warning("py-toon-format not installed, falling back to JSON. Install with: pip install py-toon-format")
        return json.dumps(resistance_genes_data, indent=2, ensure_ascii=False)


def _rank_antibiotic_with_llm(
    medical_name: str,
    first_choice_count: int,
    second_choice_count: int,
    alternative_count: int,
    total_sources: int,
    resistant_gene: str,
    severity_codes: str,
    llm: BaseChatModel
) -> str:
    """
    Rank a single antibiotic using LLM based on category counts.
    
    Args:
        medical_name: Normalized antibiotic name
        first_choice_count: Number of sources where it appears as first_choice
        second_choice_count: Number of sources where it appears as second_choice
        alternative_count: Number of sources where it appears as alternative_antibiotic
        total_sources: Total number of sources
        resistant_gene: Resistance gene name
        severity_codes: ICD severity codes
        llm: LangChain BaseChatModel
        
    Returns:
        Final category: 'first_choice', 'second_choice', or 'alternative_antibiotic'
    """
    # Build category counts text (only include if count > 0)
    category_info = []
    if first_choice_count > 0:
        category_info.append(f"{first_choice_count} out of {total_sources} sources as first_choice")
    if second_choice_count > 0:
        category_info.append(f"{second_choice_count} out of {total_sources} sources as second_choice")
    if alternative_count > 0:
        category_info.append(f"{alternative_count} out of {total_sources} sources as alternative_antibiotic")
    
    category_counts_text = ", ".join(category_info) if category_info else "No category assignments found"
    
    prompt = f"""You are a medical expert ranking an antibiotic for a patient with {resistant_gene} resistance.

Antibiotic: {medical_name}
Resistance Gene(s): {resistant_gene}
ICD Code(s): {severity_codes}

Category Distribution:
According to the sources, this antibiotic appeared as: {category_counts_text}

TASK: Based on the category distribution, medical knowledge, and the resistance pattern, determine the FINAL category for this antibiotic:
- first_choice: Best/preferred option for this condition
- second_choice: Good alternative if first-choice unavailable
- alternative_antibiotic: Other viable option

Consider:
1. The category distribution (if it appears mostly as first_choice, it should likely be first_choice)
2. Medical guidelines and evidence for {resistant_gene} resistance
3. Clinical appropriateness for the condition
4. Patient's ICD codes ({severity_codes}) - prioritize antibiotics appropriate for these conditions and severity levels

Return ONLY the final category: 'first_choice', 'second_choice', or 'alternative_antibiotic'"""
    
    try:
        from pydantic import BaseModel, Field
        class CategoryResult(BaseModel):
            final_category: str = Field(..., description="Final category: 'first_choice', 'second_choice', or 'alternative_antibiotic'")
        
        structured_llm = llm.with_structured_output(CategoryResult)
        result = structured_llm.invoke(prompt)
        return result.final_category if result else 'alternative_antibiotic'
    except Exception as e:
        logger.warning(f"Error ranking {medical_name} with LLM: {e}, defaulting to alternative_antibiotic")
        # Fallback logic: use highest count category
        if first_choice_count >= second_choice_count and first_choice_count >= alternative_count:
            return 'first_choice'
        elif second_choice_count >= alternative_count:
            return 'second_choice'
        else:
            return 'alternative_antibiotic'


def _unify_antibiotic_fields_with_llm(
    medical_name: str,
    all_entries: List[Dict[str, Any]],
    llm: BaseChatModel
) -> Dict[str, Any]:
    """
    Unify fields from all entries for the same antibiotic using LLM.
    LLM intelligently synthesizes fields from multiple sources, avoiding randomness from simple merging.
    
    Args:
        medical_name: Normalized antibiotic name
        all_entries: List of all entries for the same antibiotic from different sources
        llm: LangChain BaseChatModel
        
    Returns:
        Unified entry with all fields intelligently synthesized
    """
    if not all_entries:
        return {}
    
    # Prepare entries data for LLM
    entries_data = []
    for i, entry in enumerate(all_entries, 1):
        entries_data.append({
            'source': f"Source {entry.get('source_index', i)}",
            'coverage_for': entry.get('coverage_for'),
            'route_of_administration': entry.get('route_of_administration'),
            'dose_duration': entry.get('dose_duration'),
            'renal_adjustment': entry.get('renal_adjustment'),
            'general_considerations': entry.get('general_considerations')
        })
    
    # Format entries as readable text
    entries_text = "\n".join([
        f"{entry['source']}:\n"
        f"  - coverage_for: {entry['coverage_for'] or 'null'}\n"
        f"  - route_of_administration: {entry['route_of_administration'] or 'null'}\n"
        f"  - dose_duration: {entry['dose_duration'] or 'null'}\n"
        f"  - renal_adjustment: {entry['renal_adjustment'] or 'null'}\n"
        f"  - general_considerations: {entry['general_considerations'] or 'null'}\n"
        for entry in entries_data
    ])
    
    prompt = f"""You are a medical expert synthesizing information for the antibiotic: {medical_name}

You have information from multiple sources. Synthesize the fields intelligently:

SOURCE DATA:
{entries_text}

INSTRUCTIONS:
1. For each field, synthesize the BEST information from all sources
2. If a field is null in one source but has data in another, use the data
3. If multiple sources have different values, choose the most comprehensive/accurate one
4. Combine information when appropriate (e.g., combine general_considerations from multiple sources)
5. If a field is null in ALL sources, keep it as null
6. Avoid randomness - choose the most clinically relevant information

OUTPUT FIELDS:
- coverage_for: Specific indication/condition (e.g., "VRE bacteremia")
- route_of_administration: 'IV', 'PO', 'IM', 'IV/PO', or null
- dose_duration: 'dose,route,frequency,duration' format or null
- renal_adjustment: Dose adjustment guidance or null
- general_considerations: Combined clinical notes from all sources or null

Return the unified fields. Use null (not strings) for missing information."""

    try:
        from pydantic import BaseModel, Field
        class UnifiedFieldsResult(BaseModel):
            coverage_for: Optional[str] = Field(None, description="Unified coverage indication")
            route_of_administration: Optional[str] = Field(None, description="Unified route")
            dose_duration: Optional[str] = Field(None, description="Unified dose and duration")
            renal_adjustment: Optional[str] = Field(None, description="Unified renal adjustment")
            general_considerations: Optional[str] = Field(None, description="Unified general considerations")
        
        structured_llm = llm.with_structured_output(UnifiedFieldsResult)
        result = structured_llm.invoke(prompt)
        
        unified = {
            'coverage_for': result.coverage_for if result.coverage_for else None,
            'route_of_administration': result.route_of_administration if result.route_of_administration else None,
            'dose_duration': result.dose_duration if result.dose_duration else None,
            'renal_adjustment': result.renal_adjustment if result.renal_adjustment else None,
            'general_considerations': result.general_considerations if result.general_considerations else None
        }
        
        # Clean null strings
        for key, value in unified.items():
            if isinstance(value, str) and value.lower() in ['null', 'none', 'not specified', '']:
                unified[key] = None
        
        return unified
    except Exception as e:
        logger.warning(f"Error unifying fields for {medical_name} with LLM: {e}, falling back to simple merge")
        # Fallback to simple merge
        unified = all_entries[0].copy()
        fields_to_unify = ['coverage_for', 'route_of_administration', 'dose_duration', 'renal_adjustment', 'general_considerations']
        for field in fields_to_unify:
            current_value = unified.get(field)
            if not current_value or (isinstance(current_value, str) and current_value.lower() in ['null', 'none', 'not specified', '']):
                for entry in all_entries[1:]:
                    other_value = entry.get(field)
                    if other_value and isinstance(other_value, str) and other_value.lower() not in ['null', 'none', 'not specified', '']:
                        unified[field] = other_value
                        break
        return unified


def _unify_all_with_llm(
    antibiotics_to_unify: List[Dict[str, Any]],
    resistance_genes_to_unify: List[Dict[str, Any]],
    input_params: Dict[str, Any],
    llm: BaseChatModel
) -> Dict[str, Any]:
    """
    Unify all antibiotics and resistance genes using LLM.
    Synthesizes fields, deduplicates, and assigns final categories.
    
    Args:
        antibiotics_to_unify: List of aggregated antibiotics with their entries
        resistance_genes_to_unify: List of aggregated resistance genes with their entries
        input_params: Input parameters (pathogen_name, resistant_gene, etc.)
        llm: LangChain BaseChatModel
        
    Returns:
        Dictionary with unified first_choice, second_choice, alternative_antibiotic, and resistance_genes
    """
    pathogen_name = input_params.get('pathogen_name', 'Unknown')
    resistant_gene_raw = input_params.get('resistant_gene', 'Unknown')
    # Format resistance genes (handle comma-separated)
    resistant_gene = format_resistance_genes(resistant_gene_raw)
    severity_codes_raw = input_params.get('severity_codes', '')
    # Format ICD codes (handle comma-separated)
    severity_codes = format_icd_codes(severity_codes_raw)
    
    # Calculate total sources from all entries
    all_source_indices = set()
    for ab_data in antibiotics_to_unify:
        for entry in ab_data['entries']:
            source_idx = entry.get('source_index', 0)
            if source_idx > 0:
                all_source_indices.add(source_idx)
    total_sources = len(all_source_indices) if all_source_indices else 1
    
    logger.info(f"Unifying {len(antibiotics_to_unify)} antibiotics and {len(resistance_genes_to_unify)} resistance genes from {total_sources} sources")
    
    # Process each unique antibiotic
    antibiotics_result = {
        'first_choice': [],
        'second_choice': [],
        'alternative_antibiotic': []
    }
    
    for ab_data in antibiotics_to_unify:
        all_entries = ab_data['entries']
        # Use normalized name for comparison, but get medical_name from entries
        normalized_name = ab_data['normalized_name']
        medical_name = all_entries[0].get('medical_name', normalized_name) if all_entries else normalized_name
        
        # Verify all entries have the same normalized name (safety check)
        for entry in all_entries:
            entry_name = entry.get('medical_name', '')
            if entry_name:
                entry_normalized = _normalize_antibiotic_name(entry_name)
                if entry_normalized != normalized_name:
                    logger.warning(f"Name mismatch: {normalized_name} vs {entry_normalized} for entry {entry.get('source_index')}")
        
        # Count appearances in each category
        first_choice_count = sum(1 for e in all_entries if e.get('original_category') == 'first_choice')
        second_choice_count = sum(1 for e in all_entries if e.get('original_category') == 'second_choice')
        alternative_count = sum(1 for e in all_entries if e.get('original_category') == 'alternative_antibiotic')
        
        # TASK 1: Rank the antibiotic using LLM
        final_category = _rank_antibiotic_with_llm(
            medical_name=medical_name,
            first_choice_count=first_choice_count,
            second_choice_count=second_choice_count,
            alternative_count=alternative_count,
            total_sources=total_sources,
            resistant_gene=resistant_gene,
            severity_codes=severity_codes,
            llm=llm
        )
        
        # TASK 2: Unify fields from all sources using LLM
        unified_entry = _unify_antibiotic_fields_with_llm(
            medical_name=medical_name,
            all_entries=all_entries,
            llm=llm
        )
        
        # Build final entry
        final_entry = {
            'medical_name': medical_name,
            'coverage_for': unified_entry.get('coverage_for'),
            'route_of_administration': unified_entry.get('route_of_administration'),
            'dose_duration': unified_entry.get('dose_duration'),
            'renal_adjustment': unified_entry.get('renal_adjustment'),
            'general_considerations': unified_entry.get('general_considerations'),
            'mentioned_in_sources': ab_data['mentioned_in_sources']
        }
        
        # Clean null strings
        final_entry = _clean_null_strings(final_entry)
        
        # Add to appropriate category
        antibiotics_result[final_category].append(final_entry)
    
    # Apply limits after processing all antibiotics
    # Sort by source count within each category
    def get_source_count(entry):
        return len(entry.get('mentioned_in_sources', []))
    
    # Sort each category by source count (descending)
    antibiotics_result['first_choice'].sort(key=get_source_count, reverse=True)
    antibiotics_result['second_choice'].sort(key=get_source_count, reverse=True)
    antibiotics_result['alternative_antibiotic'].sort(key=get_source_count, reverse=True)
    
    # Apply limits
    antibiotics_result['first_choice'] = antibiotics_result['first_choice'][:5]
    antibiotics_result['second_choice'] = antibiotics_result['second_choice'][:4]
    # alternative_antibiotic: unlimited
    
    logger.info(f"After processing: {len(antibiotics_result['first_choice'])} first_choice, {len(antibiotics_result['second_choice'])} second_choice, {len(antibiotics_result['alternative_antibiotic'])} alternative")
    
    # Prepare resistance genes data
    resistance_genes_data = []
    for rg_data in resistance_genes_to_unify:
        resistance_genes_data.append({
            'normalized_name': rg_data['normalized_name'],
            'all_entries': rg_data['entries']
        })
    
    # Process resistance genes separately
    resistance_genes_result = []
    
    if resistance_genes_data:
        # Format as readable text instead of JSON
        resistance_genes_toon = _format_resistance_genes_as_toon(resistance_genes_data)
        
        logger.info(f"Preparing resistance genes prompt with {len(resistance_genes_data)} genes, TOON length: {len(resistance_genes_toon)} chars")
        
        resistance_genes_prompt = f"""Synthesize resistance gene information for {pathogen_name} with {resistant_gene} resistance from multiple sources. If multiple resistance genes are specified, synthesize information for each gene separately.

INSTRUCTIONS:

Combine information from ALL sources:

- detected_resistant_gene_name: Standard gene name (e.g., "vanA"). Must match one of the specified resistance genes: {resistant_gene}
- potential_medication_class_affected: Which antibiotic classes are affected. Combine all classes mentioned.
- general_considerations: Combine all information including:
  * Mechanism of resistance (how it works)
  * Clinical implications
  * Treatment implications
  * All clinical notes from all sources

Use null (not strings) for missing information.

RESISTANCE GENES DATA (TOON format - compact token-efficient format):
{resistance_genes_toon}"""
        
        # Use structured output for resistance genes only
        structured_llm_rg = llm.with_structured_output(UnifiedResistanceGenesResult)
        
        try:
            logger.info("Invoking LLM for resistance genes synthesis...")
            rg_result = structured_llm_rg.invoke(resistance_genes_prompt)
            
            if rg_result:
                resistance_genes_result = [_clean_null_strings(entry.model_dump()) for entry in rg_result.resistance_genes]
                logger.info(f"Resistance genes LLM returned: {len(resistance_genes_result)} genes")
            else:
                logger.warning("Resistance genes LLM returned None")
        except Exception as e:
            logger.error(f"Error during resistance genes synthesis: {e}", exc_info=True)
    
    # Combine results
    return {
        'first_choice': antibiotics_result['first_choice'],
        'second_choice': antibiotics_result['second_choice'],
        'alternative_antibiotic': antibiotics_result['alternative_antibiotic'],
        'resistance_genes': resistance_genes_result
    }


def synthesize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synthesize node that aggregates antibiotics from all sources and creates a result.
    
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
        
        # Get LangChain ChatModel - Ollama
        from config import get_ollama_llm
        llm = get_ollama_llm()
        
        # Aggregate antibiotics by normalized name and category
        aggregated = {
            'first_choice': defaultdict(lambda: {'count': 0, 'entries': []}),
            'second_choice': defaultdict(lambda: {'count': 0, 'entries': []}),
            'alternative_antibiotic': defaultdict(lambda: {'count': 0, 'entries': []})
        }
        
        # Aggregate resistance genes
        resistance_genes_agg = defaultdict(lambda: {'count': 0, 'entries': []})
        
        # Process all sources
        total_sources = len(source_results)
        
        for source_result in source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            # Process each category
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                antibiotics = therapy_plan.get(category, [])
                if not isinstance(antibiotics, list):
                    continue
                
                for antibiotic in antibiotics:
                    if not isinstance(antibiotic, dict):
                        continue
                    
                    medical_name = antibiotic.get('medical_name', '').strip()
                    if not medical_name:
                        continue
                    
                    # Normalize name for comparison (even though names should already be normalized)
                    normalized_name = _normalize_antibiotic_name(medical_name)
                    if not normalized_name:
                        continue
                    
                    # Use normalized name as key for aggregation to ensure proper grouping
                    aggregated[category][normalized_name]['count'] += 1
                    aggregated[category][normalized_name]['entries'].append({
                        **antibiotic,
                        'source_index': source_result.get('source_index', 0),
                        'original_category': category
                    })
            
            # Process resistance genes
            resistance_genes = source_result.get('pharmacist_analysis_on_resistant_gene', [])
            if isinstance(resistance_genes, list):
                for gene_entry in resistance_genes:
                    if not isinstance(gene_entry, dict):
                        continue
                    
                    gene_name = gene_entry.get('detected_resistant_gene_name', '').strip()
                    if not gene_name:
                        continue
                    
                    normalized_gene = _normalize_antibiotic_name(gene_name)
                    resistance_genes_agg[normalized_gene]['count'] += 1
                    resistance_genes_agg[normalized_gene]['entries'].append(gene_entry)
        
        # Aggregate antibiotics across ALL categories
        cross_category_aggregated = defaultdict(lambda: {'total_count': 0, 'entries_by_category': defaultdict(list), 'all_entries': []})
        
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            for normalized_name, data in aggregated[category].items():
                cross_category_aggregated[normalized_name]['total_count'] += data['count']
                cross_category_aggregated[normalized_name]['entries_by_category'][category].extend(data['entries'])
                cross_category_aggregated[normalized_name]['all_entries'].extend(data['entries'])
        
        # Collect all antibiotics with their aggregated entries
        antibiotics_to_unify = []
        for normalized_name, data in cross_category_aggregated.items():
            antibiotics_to_unify.append({
                'normalized_name': normalized_name,
                'mentioned_in_sources': sorted(set(e.get('source_index', 0) for e in data['all_entries'] if e.get('source_index', 0) > 0)),
                'entries': data['all_entries']
            })
        
        logger.info(f"Aggregated {len(antibiotics_to_unify)} unique antibiotics across all categories (total mentions: {sum(d['total_count'] for d in cross_category_aggregated.values())})")
        
        # Collect all resistance genes with their aggregated entries
        resistance_genes_to_unify = []
        sorted_genes = sorted(
            resistance_genes_agg.items(),
            key=lambda x: -x[1]['count']
        )
        for normalized_gene, data in sorted_genes:
            resistance_genes_to_unify.append({
                'normalized_name': normalized_gene,
                'entries': data['entries']
            })
        
        unified_result_data = _unify_all_with_llm(
            antibiotics_to_unify,
            resistance_genes_to_unify,
            state.get('input_parameters', {}),
            llm
        )
        
        unified_first_choice = unified_result_data['first_choice']
        unified_second_choice = unified_result_data['second_choice']
        unified_alternative = unified_result_data['alternative_antibiotic']
        unified_resistance_genes = unified_result_data['resistance_genes']
        
        # Build a mapping from source_index to source_url
        source_index_to_url = {}
        for source_result in source_results:
            source_index = source_result.get('source_index', 0)
            source_url = source_result.get('source_url', '')
            if source_index > 0 and source_url:
                source_index_to_url[source_index] = source_url
        
        # Convert source indices to URLs for all antibiotics
        def convert_indices_to_urls(antibiotics_list):
            for ab in antibiotics_list:
                if 'mentioned_in_sources' in ab:
                    # Convert list of indices to list of URLs
                    ab['mentioned_in_sources'] = [
                        source_index_to_url.get(idx, '') 
                        for idx in ab['mentioned_in_sources'] 
                        if source_index_to_url.get(idx, '')
                    ]
        
        convert_indices_to_urls(unified_first_choice)
        convert_indices_to_urls(unified_second_choice)
        convert_indices_to_urls(unified_alternative)
        
        result = {
            'antibiotic_therapy_plan': {
                'first_choice': unified_first_choice,
                'second_choice': unified_second_choice,
                'alternative_antibiotic': unified_alternative
            },
            'pharmacist_analysis_on_resistant_gene': unified_resistance_genes
        }
        
        logger.info(f"Synthesized {len(unified_first_choice)} first_choice, {len(unified_second_choice)} second_choice, {len(unified_alternative)} alternative antibiotics from {total_sources} sources")
        
        return {
            'result': result
        }
        
    except Exception as e:
        logger.error(f"Error in synthesize_node: {e}", exc_info=True)
        raise
