"""
Synthesize node for LangGraph - Aggregates and combines results from all sources using LangChain.
"""
import logging
import json
from typing import Dict, Any, List, Optional
from collections import defaultdict

from langchain_core.language_models.chat_models import BaseChatModel
from schemas import UnifiedResistanceGenesResult
from utils import format_resistance_genes, get_icd_names_from_state

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


def _unify_all_antibiotic_fields_with_llm(
    antibiotics_data: List[Dict[str, Any]],
    llm: BaseChatModel
) -> Dict[str, Dict[str, Any]]:
    """
    Unify fields for all antibiotics in batch using LLM.
    
    Args:
        antibiotics_data: List of dicts with 'medical_name' and 'all_entries'
        llm: LangChain BaseChatModel
        
    Returns:
        Dict mapping medical_name -> unified fields dict
    """
    if not antibiotics_data:
        return {}
    
    # Build antibiotics data for prompt
    antibiotics_text = []
    for ab_data in antibiotics_data:
        medical_name = ab_data['medical_name']
        all_entries = ab_data['all_entries']
        
        entries_text = []
        for i, entry in enumerate(all_entries, 1):
            source_idx = entry.get('source_index', i)
            entries_text.append(
                f"  Source {source_idx}:\n"
                f"    coverage_for: {entry.get('coverage_for') or 'null'}\n"
                f"    route: {entry.get('route_of_administration') or 'null'}\n"
                f"    dose_duration: {entry.get('dose_duration') or 'null'}\n"
                f"    renal_adjustment: {entry.get('renal_adjustment') or 'null'}\n"
                f"    general_considerations: {entry.get('general_considerations') or 'null'}"
            )
        
        antibiotics_text.append(f"{medical_name}:\n" + "\n".join(entries_text))
    
    antibiotics_list = "\n\n".join(antibiotics_text)
    
    prompt = f"""Synthesize fields for multiple antibiotics from multiple sources.

ANTIBIOTICS DATA:
{antibiotics_list}

INSTRUCTIONS:
1. For each antibiotic, synthesize BEST information from all sources
2. If field is null in one source but has data in another, use the data
3. If multiple sources have different values, choose most comprehensive/accurate
4. Combine information when appropriate (e.g., combine general_considerations)
5. If field is null in ALL sources, keep as null
6. Avoid randomness - choose most clinically relevant

OUTPUT FIELDS (for each antibiotic):
- coverage_for: Specific indication/condition (e.g., "VRE bacteremia")
- route_of_administration: 'IV', 'PO', 'IM', 'IV/PO', or null
- dose_duration: 'dose,route,frequency,duration' format or null
- renal_adjustment: Dose adjustment guidance or null
- general_considerations: Combined clinical notes or null

Return unified fields for ALL antibiotics. Use null (not strings) for missing information."""

    try:
        from pydantic import BaseModel, Field
        class UnifiedFields(BaseModel):
            medical_name: str
            coverage_for: Optional[str] = Field(None, description="Unified coverage indication")
            route_of_administration: Optional[str] = Field(None, description="Unified route")
            dose_duration: Optional[str] = Field(None, description="Unified dose and duration")
            renal_adjustment: Optional[str] = Field(None, description="Unified renal adjustment")
            general_considerations: Optional[str] = Field(None, description="Unified general considerations")
        
        class BatchUnifiedFieldsResult(BaseModel):
            unified_antibiotics: List[UnifiedFields] = Field(..., description="Unified fields for all antibiotics")
        
        structured_llm = llm.with_structured_output(BatchUnifiedFieldsResult)
        result = structured_llm.invoke(prompt)
        
        if result and result.unified_antibiotics:
            unified_map = {}
            for unified in result.unified_antibiotics:
                unified_map[unified.medical_name] = {
                    'coverage_for': unified.coverage_for if unified.coverage_for else None,
                    'route_of_administration': unified.route_of_administration if unified.route_of_administration else None,
                    'dose_duration': unified.dose_duration if unified.dose_duration else None,
                    'renal_adjustment': unified.renal_adjustment if unified.renal_adjustment else None,
                    'general_considerations': unified.general_considerations if unified.general_considerations else None
                }
                # Clean null strings
                for key, value in unified_map[unified.medical_name].items():
                    if isinstance(value, str) and value.lower() in ['null', 'none', 'not specified', '']:
                        unified_map[unified.medical_name][key] = None
            return unified_map
        else:
            logger.warning("LLM returned no unified fields, using fallback")
            return _fallback_unify_fields(antibiotics_data)
    except Exception as e:
        logger.warning(f"Error in batch field unification: {e}, using fallback")
        return _fallback_unify_fields(antibiotics_data)


def _fallback_unify_fields(antibiotics_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Fallback field unification using simple merge."""
    unified_map = {}
    for ab_data in antibiotics_data:
        medical_name = ab_data['medical_name']
        all_entries = ab_data['all_entries']
        
        if not all_entries:
            unified_map[medical_name] = {}
            continue
        
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
        
        unified_map[medical_name] = {
            'coverage_for': unified.get('coverage_for'),
            'route_of_administration': unified.get('route_of_administration'),
            'dose_duration': unified.get('dose_duration'),
            'renal_adjustment': unified.get('renal_adjustment'),
            'general_considerations': unified.get('general_considerations')
        }
    
    return unified_map


def _rank_all_antibiotics_with_llm(
    antibiotics_data: List[Dict[str, Any]],
    total_sources: int,
    resistant_gene: str,
    severity_codes: str,
    llm: BaseChatModel
) -> Dict[str, str]:
    """
    Rank all antibiotics in batch using LLM based on category counts.
    
    Args:
        antibiotics_data: List of dicts with 'medical_name', 'first_choice_count', 'second_choice_count', 'alternative_count'
        total_sources: Total number of sources
        resistant_gene: Resistance gene name
        severity_codes: ICD severity codes
        llm: LangChain BaseChatModel
        
    Returns:
        Dict mapping medical_name -> final_category ('first_choice', 'second_choice', or 'alternative_antibiotic')
    """
    if not antibiotics_data:
        return {}
    
    # Build antibiotics list for prompt
    antibiotics_text = []
    for ab_data in antibiotics_data:
        medical_name = ab_data['medical_name']
        first_choice_count = ab_data.get('first_choice_count', 0)
        second_choice_count = ab_data.get('second_choice_count', 0)
        alternative_count = ab_data.get('alternative_count', 0)
        
        category_info = []
        if first_choice_count > 0:
            category_info.append(f"{first_choice_count}/{total_sources} as first_choice")
        if second_choice_count > 0:
            category_info.append(f"{second_choice_count}/{total_sources} as second_choice")
        if alternative_count > 0:
            category_info.append(f"{alternative_count}/{total_sources} as alternative")
        
        category_text = ", ".join(category_info) if category_info else "No category assignments"
        antibiotics_text.append(f"- {medical_name}: {category_text}")
    
    antibiotics_list = "\n".join(antibiotics_text)
    
    prompt = f"""Rank antibiotics for {resistant_gene} resistance.

PATIENT: Resistance: {resistant_gene} | ICD: {severity_codes}

ANTIBIOTICS:
{antibiotics_list}

TASK: Determine FINAL category for each antibiotic:
- first_choice: Best/preferred option
- second_choice: Good alternative
- alternative_antibiotic: Other viable option

RULES:
1. Consider category distribution (mostly first_choice → first_choice)
2. Medical guidelines for {resistant_gene} resistance
3. Clinical appropriateness for condition
4. ICD codes ({severity_codes}) - prioritize appropriate antibiotics

Return category for ALL antibiotics."""
    
    try:
        from pydantic import BaseModel, Field
        class RankedAntibiotic(BaseModel):
            medical_name: str
            final_category: str = Field(..., description="'first_choice', 'second_choice', or 'alternative_antibiotic'")
        
        class BatchRankingResult(BaseModel):
            rankings: List[RankedAntibiotic] = Field(..., description="Rankings for all antibiotics")
        
        structured_llm = llm.with_structured_output(BatchRankingResult)
        result = structured_llm.invoke(prompt)
        
        if result and result.rankings:
            return {r.medical_name: r.final_category for r in result.rankings}
        else:
            logger.warning("LLM returned no rankings, using fallback logic")
            return _fallback_rankings(antibiotics_data)
    except Exception as e:
        logger.warning(f"Error in batch ranking: {e}, using fallback logic")
        return _fallback_rankings(antibiotics_data)


def _fallback_rankings(antibiotics_data: List[Dict[str, Any]]) -> Dict[str, str]:
    """Fallback ranking logic using highest count category."""
    rankings = {}
    for ab_data in antibiotics_data:
        medical_name = ab_data['medical_name']
        first_choice_count = ab_data.get('first_choice_count', 0)
        second_choice_count = ab_data.get('second_choice_count', 0)
        alternative_count = ab_data.get('alternative_count', 0)
        
        if first_choice_count >= second_choice_count and first_choice_count >= alternative_count:
            rankings[medical_name] = 'first_choice'
        elif second_choice_count >= alternative_count:
            rankings[medical_name] = 'second_choice'
        else:
            rankings[medical_name] = 'alternative_antibiotic'
    return rankings


def _unify_all_with_llm(
    antibiotics_to_unify: List[Dict[str, Any]],
    resistance_genes_to_unify: List[Dict[str, Any]],
    input_params: Dict[str, Any],
    llm: BaseChatModel,
    state: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Unify all antibiotics and resistance genes using LLM.
    Re-ranks antibiotics and collects ALL dosage mentions (not just unified).
    
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
    # Get ICD names from state (transformed), fallback to codes
    if state:
        severity_codes = get_icd_names_from_state(state)
    else:
        severity_codes_raw = input_params.get('severity_codes', '')
        from utils import format_icd_codes
        severity_codes = format_icd_codes(severity_codes_raw)
    sample = input_params.get('sample', '')
    systemic = input_params.get('systemic', True)
    age = input_params.get('age')
    
    # Calculate total sources from all entries
    all_source_indices = set()
    for ab_data in antibiotics_to_unify:
        for entry in ab_data['entries']:
            source_idx = entry.get('source_index', 0)
            if source_idx > 0:
                all_source_indices.add(source_idx)
    total_sources = len(all_source_indices) if all_source_indices else 1
    
    logger.info(f"Unifying {len(antibiotics_to_unify)} antibiotics and {len(resistance_genes_to_unify)} resistance genes from {total_sources} sources")
    
    # Prepare data for ranking and field collection
    ranking_data = []
    field_collection_data = []
    
    for ab_data in antibiotics_to_unify:
        all_entries = ab_data['entries']
        normalized_name = ab_data['normalized_name']
        medical_name = all_entries[0].get('medical_name', normalized_name) if all_entries else normalized_name
        
        # Count appearances in each category for ranking
        first_choice_count = sum(1 for e in all_entries if e.get('original_category') == 'first_choice')
        second_choice_count = sum(1 for e in all_entries if e.get('original_category') == 'second_choice')
        alternative_count = sum(1 for e in all_entries if e.get('original_category') == 'alternative_antibiotic')
        
        ranking_data.append({
            'medical_name': medical_name,
            'first_choice_count': first_choice_count,
            'second_choice_count': second_choice_count,
            'alternative_count': alternative_count
        })
        
        field_collection_data.append({
            'medical_name': medical_name,
            'all_entries': all_entries,
            'mentioned_in_sources': ab_data['mentioned_in_sources']
        })
    
    # Single LLM call: Rank + Unify fields (with all dosage mentions)
    logger.info(f"Ranking and unifying {len(field_collection_data)} antibiotics in one LLM call...")
    
    # Build prompt with all data
    antibiotics_text = []
    for ab_data in field_collection_data:
        medical_name = ab_data['medical_name']
        all_entries = ab_data['all_entries']
        
        # Count category appearances
        first_choice_count = sum(1 for e in all_entries if e.get('original_category') == 'first_choice')
        second_choice_count = sum(1 for e in all_entries if e.get('original_category') == 'second_choice')
        alternative_count = sum(1 for e in all_entries if e.get('original_category') == 'alternative_antibiotic')
        
        category_info = []
        if first_choice_count > 0:
            category_info.append(f"{first_choice_count}/{total_sources} as first_choice")
        if second_choice_count > 0:
            category_info.append(f"{second_choice_count}/{total_sources} as second_choice")
        if alternative_count > 0:
            category_info.append(f"{alternative_count}/{total_sources} as alternative")
        category_text = ", ".join(category_info) if category_info else "No category assignments"
        
        # Collect all entries data
        entries_data = []
        for entry in all_entries:
            entries_data.append({
                'source': f"Source {entry.get('source_index', '?')}",
                'coverage_for': entry.get('coverage_for') or 'null',
                'route': entry.get('route_of_administration') or 'null',
                'dose_duration': entry.get('dose_duration') or 'null',
                'renal_adjustment': entry.get('renal_adjustment') or 'null',
                'general_considerations': entry.get('general_considerations') or 'null'
            })
        
        entries_text = "\n".join([
            f"  {e['source']}: coverage={e['coverage_for']}, route={e['route']}, dose={e['dose_duration']}, renal={e['renal_adjustment']}, considerations={e['general_considerations']}"
            for e in entries_data
        ])
        
        antibiotics_text.append(f"{medical_name} (Category: {category_text}):\n{entries_text}")
    
    antibiotics_list = "\n\n".join(antibiotics_text)
    
    # Build patient context with sample and systemic info
    patient_context = f"Resistance: {resistant_gene} | ICD: {severity_codes}"
    if sample:
        patient_context += f" | Sample: {sample}"
    if systemic is not None:
        patient_context += f" | Systemic: {'Yes' if systemic else 'No'}"
    
    prompt = f"""Unify and rank antibiotics for {resistant_gene} resistance.

PATIENT: {patient_context}

ANTIBIOTICS FROM MULTIPLE SOURCES:
{antibiotics_list}

TASK: Group medically equivalent antibiotics and unify their data.

STEP 1 - GROUP: Identify same antibiotics across sources. Group as ONE unified entry. Use standard name.

STEP 2 - UNIFY: For each group, combine information from all sources. Keep only data relevant to patient parameters:
- coverage_for: Keep only indications that match patient condition (ICD: {severity_codes}). Do not combine all - filter to match use case.
- dose_duration: Select dosage that matches patient parameters (ICD: {severity_codes}, Gene: {resistant_gene}, Age: {age if age else 'N/A'})
- Fill missing fields from other sources in group, but keep them relevant to patient parameters

STEP 3 - DOSAGE: Select ONE optimal dose_duration from provided sources that matches patient parameters:
- Use ONLY dosages shown in source data above
- MUST match patient condition (ICD: {severity_codes})
- MUST consider resistance gene: {resistant_gene}
- MUST consider patient age: {age if age else 'N/A'}
- DO NOT invent, create, or hallucinate dosages
- DO NOT combine unrelated dosages - choose the one most appropriate for patient
- If source shows null/empty/"not specified", return null
{f"- Consider sample type: For \"{sample}\" samples, prioritize dosages appropriate for that sample type" if sample else ""}
{f"- Consider systemic: For systemic treatment, prefer IV/PO/IM routes. For local/topical treatment, prefer local routes" if systemic is not None else ""}
- Single: "dose,route,freq,dur" format
- If multiple dosages exist for different conditions (both in patient ICD codes), use pipe: "dose1,route1,freq1,dur1|dose2,route2,freq2,dur2"
- If multiple exist, select most appropriate for patient parameters
- If ALL sources have null, return null

STEP 4 - RANK: Determine final_category based on category distribution, medical guidelines, and clinical appropriateness.

STEP 5 - COMPLETENESS: Determine if the record is fully complete. A record is considered COMPLETE if ALL of these critical fields are non-null:
- medical_name (always required)
- coverage_for (required for prescribing)
- dose_duration (required for prescribing)
- route_of_administration (required for prescribing)

Optional fields (renal_adjustment, general_considerations) can be null and the record can still be complete.
Mark is_complete as True only if ALL critical fields are present and non-null.

OUTPUT for each unified antibiotic:
- medical_name: Standard name
- final_category: 'first_choice', 'second_choice', or 'alternative_antibiotic'
- coverage_for: Coverage indication relevant to patient condition
- route_of_administration: Unified route ('IV', 'PO', 'IM', 'IV/PO', or null)
- dose_duration: ONE optimal dosage from sources that matches patient parameters
- renal_adjustment: Unified renal adjustment (combine all, or null)
- general_considerations: Unified considerations (combine all, or null)
- is_complete: Boolean - True if medical_name, coverage_for, dose_duration, and route_of_administration are ALL non-null. False otherwise.

FIELD FORMATS:
- coverage_for: Keep only indications relevant to patient condition ({severity_codes}). Do not combine all - keep only what matches the use case.
- route_of_administration: 'IV', 'PO', 'IM', 'IV/PO'. If multiple, use 'IV/PO'. If not mentioned, null.
- dose_duration: 
  * Use ONLY dosages from provided sources that match patient parameters (ICD: {severity_codes}, Gene: {resistant_gene}, Age: {age if age else 'N/A'})
  * DO NOT invent or combine unrelated dosages
  * Single: "dose,route,frequency,duration" (comma-separated)
  * If multiple dosages for different conditions (both in patient ICD codes), use pipe: "dose1,route1,freq1,dur1|dose2,route2,freq2,dur2"
  * If source shows null/empty/"not specified", return null
  * Choose the dosage most appropriate for the patient condition
- renal_adjustment: Combine all. Format: "Adjust dose in CrCl < X mL/min" or similar. If not mentioned, null.
- general_considerations: Combine all clinical notes, warnings, monitoring from all sources. If nothing, null.

RULES:
- Group same antibiotics from different sources
- Keep only data relevant to patient condition ({severity_codes}), resistance gene ({resistant_gene}), and age ({age if age else 'N/A'})
- Do not combine all coverage_for - keep only what matches the use case
- Use ONLY data from provided sources - do not invent
- Make data complete by filling gaps from other sources in group
- Return ONE entry per medically unique antibiotic
- Dosage must match patient parameters - do not combine unrelated dosages

Return ALL unified antibiotics."""
    
    try:
        from pydantic import BaseModel, Field
        class RankedAndUnifiedAntibiotic(BaseModel):
            medical_name: str = Field(..., description="Standard name for the antibiotic (use most complete/standard form after grouping medically equivalent entries)")
            final_category: str = Field(..., description="'first_choice', 'second_choice', or 'alternative_antibiotic' based on category distribution and medical guidelines")
            coverage_for: Optional[str] = Field(None, description="Coverage indication relevant to patient condition. Keep only what matches the use case - do not combine all mentions.")
            route_of_administration: Optional[str] = Field(None, description="Unified route - combine all from all sources in the group")
            dose_duration: Optional[str] = Field(None, description="ONE optimal dosage FROM PROVIDED SOURCES ONLY that matches patient parameters (ICD codes, resistance gene, age). DO NOT invent or hallucinate. DO NOT combine unrelated dosages. Single: 'dose,route,freq,dur'. If multiple dosages for different conditions (both in patient ICD codes), use pipe: 'dose1,route1,freq1,dur1|dose2,route2,freq2,dur2'. If sources say null or 'not specified', return null (NOT an invented dosage).")
            renal_adjustment: Optional[str] = Field(None, description="Unified renal adjustment - combine all from all sources in the group")
            general_considerations: Optional[str] = Field(None, description="Unified general considerations - combine all from all sources in the group to make data complete")
            is_complete: bool = Field(..., description="True if medical_name, coverage_for, dose_duration, and route_of_administration are ALL non-null. False if any of these critical fields is null.")
        
        class CombinedResult(BaseModel):
            antibiotics: List[RankedAndUnifiedAntibiotic] = Field(..., description="Medically equivalent antibiotics grouped and unified, with complete data from all sources")
        
        structured_llm = llm.with_structured_output(CombinedResult)
        result = structured_llm.invoke(prompt)
        
        rankings_map = {}
        unified_fields_map = {}
        completeness_map = {}
        
        if result and result.antibiotics:
            for ab in result.antibiotics:
                medical_name = ab.medical_name
                rankings_map[medical_name] = ab.final_category
                unified_fields_map[medical_name] = {
                    'coverage_for': ab.coverage_for if ab.coverage_for else None,
                    'route_of_administration': ab.route_of_administration if ab.route_of_administration else None,
                    'dose_duration': ab.dose_duration if ab.dose_duration else None,
                    'renal_adjustment': ab.renal_adjustment if ab.renal_adjustment else None,
                    'general_considerations': ab.general_considerations if ab.general_considerations else None
                }
                # Clean null strings
                for key, value in unified_fields_map[medical_name].items():
                    if isinstance(value, str) and value.lower() in ['null', 'none', 'not specified', '']:
                        unified_fields_map[medical_name][key] = None
                
                # Store completeness from LLM
                completeness_map[medical_name] = ab.is_complete
        else:
            logger.warning("LLM returned no results, using fallback")
            rankings_map = _fallback_rankings(ranking_data)
            unified_fields_map = _fallback_unify_fields(field_collection_data)
            completeness_map = {}
    except Exception as e:
        logger.warning(f"Error in combined ranking/unification: {e}, using fallback")
        rankings_map = _fallback_rankings(ranking_data)
        unified_fields_map = _fallback_unify_fields(field_collection_data)
        completeness_map = {}
    
    # Build final entries (deduplicated by medical_name, re-ranked)
    antibiotics_result = {
        'first_choice': [],
        'second_choice': [],
        'alternative_antibiotic': []
    }
    
    # Deduplicate: track which antibiotics we've already added
    seen_antibiotics = set()
    
    for ab_data in field_collection_data:
        medical_name = ab_data['medical_name']
        
        # Skip if already processed (deduplication)
        if medical_name in seen_antibiotics:
            continue
        seen_antibiotics.add(medical_name)
        
        # Get re-ranked category
        final_category = rankings_map.get(medical_name, 'alternative_antibiotic')
        unified_fields = unified_fields_map.get(medical_name, {})
        
        final_entry = {
            'medical_name': medical_name,
            'coverage_for': unified_fields.get('coverage_for'),
            'route_of_administration': unified_fields.get('route_of_administration'),
            'dose_duration': unified_fields.get('dose_duration'),  # Contains ALL dosage mentions
            'renal_adjustment': unified_fields.get('renal_adjustment'),
            'general_considerations': unified_fields.get('general_considerations'),
            'mentioned_in_sources': ab_data['mentioned_in_sources']
        }
        
        # Clean null strings
        final_entry = _clean_null_strings(final_entry)
        
        # Determine completeness: check if LLM provided it, otherwise calculate it
        if medical_name in completeness_map:
            final_entry['is_complete'] = completeness_map[medical_name]
        else:
            # Fallback: calculate completeness based on critical fields
            final_entry['is_complete'] = (
                final_entry.get('medical_name') is not None and
                final_entry.get('coverage_for') is not None and
                final_entry.get('dose_duration') is not None and
                final_entry.get('route_of_administration') is not None
            )
        
        # Add to re-ranked category
        antibiotics_result[final_category].append(final_entry)
    
    # Sort by source count within each category (no limits - include all)
    def get_source_count(entry):
        return len(entry.get('mentioned_in_sources', []))
    
    # Sort each category by source count (descending)
    antibiotics_result['first_choice'].sort(key=get_source_count, reverse=True)
    antibiotics_result['second_choice'].sort(key=get_source_count, reverse=True)
    antibiotics_result['alternative_antibiotic'].sort(key=get_source_count, reverse=True)
    
    logger.info(f"After processing: {len(antibiotics_result['first_choice'])} first_choice, {len(antibiotics_result['second_choice'])} second_choice, {len(antibiotics_result['alternative_antibiotic'])} alternative")
    
    # Prepare resistance genes data - format same as antibiotics
    resistance_genes_data = []
    for rg_data in resistance_genes_to_unify:
        all_entries = rg_data['entries']
        normalized_name = rg_data['normalized_name']
        gene_name = all_entries[0].get('detected_resistant_gene_name', normalized_name) if all_entries else normalized_name
        
        resistance_genes_data.append({
            'gene_name': gene_name,
            'all_entries': all_entries
        })
    
    # Process resistance genes separately
    resistance_genes_result = []
    
    if resistance_genes_data:
        # Build resistance genes list with same formatting as antibiotics
        resistance_genes_text = []
        for rg_data in resistance_genes_data:
            gene_name = rg_data['gene_name']
            all_entries = rg_data['all_entries']
            
            # Collect all entries data
            entries_data = []
            for entry in all_entries:
                entries_data.append({
                    'source': f"Source {entry.get('source_index', '?')}",
                    'detected_resistant_gene_name': entry.get('detected_resistant_gene_name') or 'null',
                    'potential_medication_class_affected': entry.get('potential_medication_class_affected') or 'null',
                    'general_considerations': entry.get('general_considerations') or 'null'
                })
            
            entries_text = "\n".join([
                f"  {e['source']}: gene_name={e['detected_resistant_gene_name']}, medication_class={e['potential_medication_class_affected']}, considerations={e['general_considerations']}"
                for e in entries_data
            ])
            
            resistance_genes_text.append(f"{gene_name}:\n{entries_text}")
        
        resistance_genes_list = "\n\n".join(resistance_genes_text)
        
        logger.info(f"Preparing resistance genes prompt with {len(resistance_genes_data)} genes")
        
        resistance_genes_prompt = f"""Synthesize resistance gene information for {pathogen_name} with {resistant_gene} resistance from multiple sources. If multiple resistance genes are specified, synthesize information for each gene separately.

RESISTANCE GENES FROM MULTIPLE SOURCES:
{resistance_genes_list}

TASK: Group same resistance genes across sources and unify their data.

STEP 1 - GROUP: Identify same resistance genes across sources. Group as ONE unified entry. Use standard name.

STEP 2 - UNIFY: For each group, combine information from all sources:
- detected_resistant_gene_name: Standard gene name (e.g., "vanA"). Must match one of the specified resistance genes: {resistant_gene}
- potential_medication_class_affected: Which antibiotic classes are affected. Combine all classes mentioned from all sources.
- general_considerations: Combine all information including:
  * Mechanism of resistance (how it works)
  * Clinical implications
  * Treatment implications
  * All clinical notes from all sources

RULES:
- Group same resistance genes from different sources
- Unify by combining all information
- Use ONLY data from provided sources - do not invent
- Make data complete by filling gaps from other sources in group
- Return ONE entry per unique resistance gene

Use null (not strings) for missing information.

Return ALL unified resistance genes."""
        
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
        cross_category_aggregated = defaultdict(lambda: {'total_count': 0, 'all_entries': []})
        
        for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
            for normalized_name, data in aggregated[category].items():
                cross_category_aggregated[normalized_name]['total_count'] += data['count']
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
            llm,
            state
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
