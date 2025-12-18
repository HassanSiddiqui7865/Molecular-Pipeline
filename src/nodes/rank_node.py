"""
Rank node for LangGraph - Validates and ranks all antibiotics from all sources.
Processes each source separately, validates relevance, ranks not_known, and removes non-useful ones.
"""
import logging
from typing import Dict, Any

from schemas import RankedAntibioticsResult
from utils import format_resistance_genes, format_icd_codes

logger = logging.getLogger(__name__)


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
        
        pathogen_name = input_params.get('pathogen_name', '')
        resistant_gene_raw = input_params.get('resistant_gene', '')
        resistant_gene = format_resistance_genes(resistant_gene_raw)
        severity_codes_raw = input_params.get('severity_codes', '')
        severity_codes = format_icd_codes(severity_codes_raw)
        age = input_params.get('age')
        
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
            
            # Build antibiotics list (only names)
            antibiotics_text = []
            for ab_entry, category in source_antibiotics:
                medical_name = ab_entry.get('medical_name', 'Unknown')
                antibiotics_text.append(f"- {medical_name} (current: {category})")
            
            antibiotics_list_text = "\n".join(antibiotics_text)
            
            # Create prompt for this source
            ranking_prompt = f"""Validate and rank antibiotics for {pathogen_name} with {resistant_gene} resistance.

PATIENT: {pathogen_name} | Resistance: {resistant_gene} | ICD: {severity_codes} | Age: {age if age else 'N/A'}

SOURCE: {source_title}

ANTIBIOTICS:
{antibiotics_list_text}

SOURCE CONTEXT:
{source_snippet}

DEFINITIONS:
- first_choice: Best/preferred for {pathogen_name} with {resistant_gene}, appropriate for condition
- second_choice: Good alternative, effective against {resistant_gene}
- alternative_antibiotic: Other viable option
- remove: NOT useful against {resistant_gene} OR inappropriate

TASK:
1. VALIDATE: is_relevant = True if useful against {resistant_gene} AND appropriate for condition/age, else False
2. RANK: first_choice/second_choice/alternative_antibiotic/remove
3. RULES:
   - Rank "not_known" into proper category
   - Validate existing categories, keep/change/remove as needed
   - Remove if NOT useful or inappropriate
   - Consider age ({age}) and ICD codes ({severity_codes})

Return ALL with ranked_category, is_relevant, ranking_reason."""
            
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
            ranking_map = {}
            for ranked_entry in ranked_result.ranked_antibiotics:
                medical_name = ranked_entry.medical_name
                ranking_map[medical_name] = {
                    'ranked_category': ranked_entry.ranked_category,
                    'is_relevant': ranked_entry.is_relevant,
                    'reason': ranked_entry.ranking_reason
                }
                # Save to memory (except 'remove')
                if ranked_entry.ranked_category != 'remove' and ranked_entry.is_relevant:
                    rank_memory[medical_name] = ranked_entry.ranked_category
            
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
                
                # Get ranking decision
                ranking_info = ranking_map.get(medical_name)
                if not ranking_info:
                    # Not in ranking result, keep in not_known
                    new_not_known.append(ab_entry)
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
