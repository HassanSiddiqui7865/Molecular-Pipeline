"""
Rank node for LangGraph - Validates and ranks all antibiotics from all sources.
Processes each source separately, validates relevance, ranks not_known, and removes non-useful ones.
"""
import logging
from typing import Dict, Any, List, Tuple

from schemas import RankedAntibioticsResult
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


def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that validates and ranks all antibiotics from all sources.
    Processes each source separately with its own LLM call.
    """
    try:
        source_results = state.get('source_results', [])
        input_params = state.get('input_parameters', {})
        rank_memory = state.get('rank_memory', {}) or {}
        search_results = state.get('search_results', [])
        
        if not source_results:
            logger.info("No source results to rank")
            return {'rank_memory': rank_memory}
        
        # Get pathogens
        from utils import get_pathogens_from_input, format_pathogens
        pathogens = get_pathogens_from_input(input_params)
        pathogen_name = format_pathogens(pathogens)
        
        # Get resistance genes
        from utils import get_resistance_genes_from_input, format_resistance_genes
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistant_gene = format_resistance_genes(resistant_genes)
        
        # Get ICD names from state (transformed), fallback to codes
        severity_codes = get_icd_names_from_state(state)
        age = input_params.get('age')
        sample = input_params.get('sample', '')
        systemic = input_params.get('systemic', True)
        
        # Get LLM
        from config import get_ollama_llm
        llm = get_ollama_llm()
        
        # Process each source separately
        updated_source_results = []
        total_removed = 0
        
        for source_result in source_results:
            source_index = source_result.get('source_index', 0)
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            source_title = source_result.get('source_title', '')
            
            # Get source context
            source_snippet = ''
            if source_index > 0 and source_index <= len(search_results):
                source_info = search_results[source_index - 1]
                source_snippet = source_info.get('snippet', '')
            
            # Collect all antibiotics from this source
            source_antibiotics = []
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict) and ab_entry.get('medical_name'):
                            source_antibiotics.append((ab_entry, category))
            
            if not source_antibiotics:
                updated_source_results.append(source_result)
                continue
            
            logger.info(f"Processing {len(source_antibiotics)} antibiotics from source {source_index}: {source_title[:50]}...")
            
            # Build antibiotics list with ALL fields for context
            antibiotics_text = []
            for ab_entry, category in source_antibiotics:
                medical_name = ab_entry.get('medical_name', 'Unknown')
                coverage = ab_entry.get('coverage_for', 'null')
                route = ab_entry.get('route_of_administration', 'null')
                dose = ab_entry.get('dose_duration', 'null')
                renal = ab_entry.get('renal_adjustment', 'null')
                considerations = ab_entry.get('general_considerations', 'null')
                antibiotics_text.append(
                    f"- {medical_name} (current: {category})\n"
                    f"  coverage_for: {coverage}\n"
                    f"  route: {route}\n"
                    f"  dose_duration: {dose}\n"
                    f"  renal_adjustment: {renal}\n"
                    f"  general_considerations: {considerations}"
                )
            
            antibiotics_list_text = "\n".join(antibiotics_text)
            
            # Build patient context
            patient_context = f"{pathogen_name} | Resistance: {resistant_gene} | ICD: {severity_codes} | Age: {age if age else 'N/A'}"
            if sample:
                patient_context += f" | Sample: {sample}"
            if systemic is not None:
                patient_context += f" | Systemic: {'Yes' if systemic else 'No'}"
            
            # Create prompt for this source
            ranking_prompt = f"""Validate and rank antibiotics for {pathogen_name} with {resistant_gene} resistance.

PATIENT: {patient_context}
SOURCE: {source_title}

ANTIBIOTICS (RANK ALL {len(source_antibiotics)}):
{antibiotics_list_text}

SOURCE CONTEXT:
{source_snippet}

CATEGORIES:
- first_choice: Best/preferred for {pathogen_name} with {resistant_gene}, appropriate for condition
- second_choice: Good alternative, effective against {resistant_gene}
- alternative_antibiotic: Other viable option
- remove: NOT useful against {resistant_gene} OR inappropriate

PROCESS:
1. Review each antibiotic's coverage_for, dose_duration, route, and other fields
2. VALIDATE: is_relevant=True if useful against {resistant_gene} AND appropriate for condition/age/ICD codes, else False
3. RANK based on:
   - Effectiveness against {resistant_gene}
   - Appropriateness for ICD: {severity_codes}
   - Appropriateness for age: {age if age else 'N/A'}
   - Dosage accuracy and completeness
4. RULES:
   - Rank "not_known" into proper category
   - Validate existing categories, keep/change/remove as needed
   - Remove if NOT useful or inappropriate
   - Consider ALL fields (coverage, dosage, route) when ranking
   - CRITICAL: Return ranking for ALL {len(source_antibiotics)} antibiotics (do not skip any)

Return ALL {len(source_antibiotics)} antibiotics with ranked_category, is_relevant, ranking_reason."""
            
            # Use structured output
            structured_llm = llm.with_structured_output(RankedAntibioticsResult)
            
            try:
                ranked_result = structured_llm.invoke(ranking_prompt)
            except Exception as e:
                logger.error(f"Error ranking source {source_index}: {e}")
                updated_source_results.append(source_result)
                continue
            
            if not ranked_result or not ranked_result.ranked_antibiotics:
                logger.warning(f"LLM returned no ranked antibiotics for source {source_index}")
                updated_source_results.append(source_result)
                continue
            
            # Build mapping: medical_name -> (ranked_category, is_relevant, reason)
            # Use normalized names for matching to handle variations
            ranking_map = {}
            normalized_ranking_map = {}  # normalized_name -> original_name -> ranking_info
            
            for ranked_entry in ranked_result.ranked_antibiotics:
                medical_name = ranked_entry.medical_name
                normalized_name = _normalize_antibiotic_name(medical_name)
                
                ranking_info = {
                    'ranked_category': ranked_entry.ranked_category,
                    'is_relevant': ranked_entry.is_relevant,
                    'reason': ranked_entry.ranking_reason
                }
                
                # Store by exact name
                ranking_map[medical_name] = ranking_info
                
                # Store by normalized name for fuzzy matching
                if normalized_name not in normalized_ranking_map:
                    normalized_ranking_map[normalized_name] = {}
                normalized_ranking_map[normalized_name][medical_name] = ranking_info
                
                # Save to memory (except 'remove')
                if ranked_entry.ranked_category != 'remove' and ranked_entry.is_relevant:
                    rank_memory[medical_name] = ranked_entry.ranked_category
            
            # Find antibiotics that weren't ranked
            unranked_antibiotics = []
            for ab_entry, original_category in source_antibiotics:
                medical_name = ab_entry.get('medical_name', '')
                if not medical_name:
                    continue
                
                # Check exact match first
                if medical_name not in ranking_map:
                    # Check normalized match
                    normalized_name = _normalize_antibiotic_name(medical_name)
                    if normalized_name in normalized_ranking_map:
                        # Found by normalized name, use first match
                        matched_name = list(normalized_ranking_map[normalized_name].keys())[0]
                        ranking_map[medical_name] = normalized_ranking_map[normalized_name][matched_name]
                    else:
                        # Still not found, add to unranked list
                        unranked_antibiotics.append((ab_entry, original_category))
            
            # If there are unranked antibiotics, make a second LLM call just for them
            if unranked_antibiotics:
                logger.warning(f"Source {source_index}: {len(unranked_antibiotics)} antibiotics not ranked, making retry call...")
                
                unranked_text = []
                for ab_entry, category in unranked_antibiotics:
                    medical_name = ab_entry.get('medical_name', 'Unknown')
                    unranked_text.append(f"- {medical_name} (current: {category})")
                
                unranked_list_text = "\n".join(unranked_text)
                
                retry_prompt = f"""Rank these {len(unranked_antibiotics)} antibiotics that were missed.

PATIENT: {patient_context}
SOURCE: {source_title}

MISSED ANTIBIOTICS (RANK ALL):
{unranked_list_text}

SOURCE CONTEXT:
{source_snippet}

CATEGORIES:
- first_choice: Best/preferred for {pathogen_name} with {resistant_gene}, appropriate for condition
- second_choice: Good alternative, effective against {resistant_gene}
- alternative_antibiotic: Other viable option
- remove: NOT useful against {resistant_gene} OR inappropriate

TASK: Rank ALL {len(unranked_antibiotics)} antibiotics. Return ranked_category, is_relevant, ranking_reason for EACH one."""
                
                try:
                    retry_result = structured_llm.invoke(retry_prompt)
                    if retry_result and retry_result.ranked_antibiotics:
                        for ranked_entry in retry_result.ranked_antibiotics:
                            medical_name = ranked_entry.medical_name
                            normalized_name = _normalize_antibiotic_name(medical_name)
                            
                            ranking_info = {
                                'ranked_category': ranked_entry.ranked_category,
                                'is_relevant': ranked_entry.is_relevant,
                                'reason': ranked_entry.ranking_reason
                            }
                            
                            # Store by exact name
                            ranking_map[medical_name] = ranking_info
                            
                            # Also store by normalized name for matching
                            if normalized_name not in normalized_ranking_map:
                                normalized_ranking_map[normalized_name] = {}
                            normalized_ranking_map[normalized_name][medical_name] = ranking_info
                            
                            # Save to memory (except 'remove')
                            if ranked_entry.ranked_category != 'remove' and ranked_entry.is_relevant:
                                rank_memory[medical_name] = ranked_entry.ranked_category
                        
                        # Match unranked antibiotics by normalized name
                        for ab_entry, original_category in unranked_antibiotics:
                            medical_name = ab_entry.get('medical_name', '')
                            if not medical_name:
                                continue
                            
                            normalized_name = _normalize_antibiotic_name(medical_name)
                            if normalized_name in normalized_ranking_map:
                                # Found match, use first one
                                matched_name = list(normalized_ranking_map[normalized_name].keys())[0]
                                ranking_map[medical_name] = normalized_ranking_map[normalized_name][matched_name]
                        
                        logger.info(f"Retry ranked {len(retry_result.ranked_antibiotics)} antibiotics")
                except Exception as e:
                    logger.warning(f"Error in retry ranking for source {source_index}: {e}")
            
            # Reorganize categories and remove non-useful
            new_first_choice = []
            new_second_choice = []
            new_alternative = []
            new_not_known = []
            source_removed = 0
            
            for ab_entry, original_category in source_antibiotics:
                medical_name = ab_entry.get('medical_name', '')
                if not medical_name:
                    continue
                
                # Get ranking decision - try exact match first, then normalized
                ranking_info = ranking_map.get(medical_name)
                if not ranking_info:
                    # Try normalized name matching
                    normalized_name = _normalize_antibiotic_name(medical_name)
                    if normalized_name in normalized_ranking_map:
                        matched_name = list(normalized_ranking_map[normalized_name].keys())[0]
                        ranking_info = normalized_ranking_map[normalized_name][matched_name]
                        ranking_map[medical_name] = ranking_info  # Cache for future use
                
                if not ranking_info:
                    # Still not ranked after retry - remove it as it's likely not useful
                    logger.warning(f"Source {source_index}: {medical_name} still not ranked after retry, removing")
                    source_removed += 1
                    continue
                
                # Check if should be removed
                if ranking_info['ranked_category'] == 'remove' or not ranking_info['is_relevant']:
                    logger.info(f"Removing {medical_name} from source {source_index}: {ranking_info['reason']}")
                    source_removed += 1
                    continue
                
                # Assign to new category
                new_category = ranking_info['ranked_category']
                if new_category == 'first_choice':
                    new_first_choice.append(ab_entry)
                elif new_category == 'second_choice':
                    new_second_choice.append(ab_entry)
                elif new_category == 'alternative_antibiotic':
                    new_alternative.append(ab_entry)
                else:
                    new_not_known.append(ab_entry)
            
            total_removed += source_removed
            
            # Update therapy plan
            therapy_plan['first_choice'] = new_first_choice
            therapy_plan['second_choice'] = new_second_choice
            therapy_plan['alternative_antibiotic'] = new_alternative
            therapy_plan['not_known'] = new_not_known
            
            source_result['antibiotic_therapy_plan'] = therapy_plan
            updated_source_results.append(source_result)
            
            logger.info(f"Source {source_index}: Processed {len(source_antibiotics)} antibiotics, removed {source_removed}")
        
        logger.info(f"Ranking complete: Processed {len(source_results)} sources, removed {total_removed} antibiotics total")
        
        return {
            'source_results': updated_source_results,
            'rank_memory': rank_memory
        }
        
    except Exception as e:
        logger.error(f"Error in rank_node: {e}", exc_info=True)
        rank_memory = state.get('rank_memory', {}) or {}
        return {'rank_memory': rank_memory}
