"""
Rank node for LangGraph - Ranks antibiotics in 'not_known' category using LLM medical knowledge.
Takes 'not_known' entries from source_results, ranks them based on source text, condition, and age,
then re-assigns them to proper categories (first_choice, second_choice, or alternative_antibiotic).
"""
import logging
from typing import Dict, Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from config import get_ollama_config
from schemas import RankedAntibioticsResult
from utils import format_resistance_genes, format_icd_codes

logger = logging.getLogger(__name__)


def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that ranks antibiotics from 'not_known' category using LLM medical knowledge.
    Takes entries from source_results, ranks them, and updates source_results with new categories.
    
    Args:
        state: Pipeline state dictionary
        
    Returns:
        Updated state with source_results updated (not_known entries re-categorized)
    """
    try:
        source_results = state.get('source_results', [])
        input_params = state.get('input_parameters', {})
        
        if not source_results:
            logger.info("No source results to rank")
            return {}
        
        pathogen_name = input_params.get('pathogen_name', '')
        resistant_gene_raw = input_params.get('resistant_gene', '')
        # Format resistance genes (handle comma-separated)
        resistant_gene = format_resistance_genes(resistant_gene_raw)
        severity_codes_raw = input_params.get('severity_codes', '')
        # Format ICD codes (handle comma-separated)
        severity_codes = format_icd_codes(severity_codes_raw)
        age = input_params.get('age')
        
        # Collect all 'not_known' entries from all sources
        not_known_entries = []
        source_contexts = []  # Store source text for context
        
        for source_result in source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            not_known_ab = therapy_plan.get('not_known', [])
            
            if not_known_ab:
                # Get source context (title and snippet) for better ranking
                source_title = source_result.get('source_title', '')
                source_url = source_result.get('source_url', '')
                source_index = source_result.get('source_index', 0)
                
                # Try to get the original source text from search_results
                search_results = state.get('search_results', [])
                source_text = ''
                if source_index > 0 and source_index <= len(search_results):
                    source_info = search_results[source_index - 1]
                    source_text = source_info.get('snippet', '')[:500]  # First 500 chars for context
                
                for ab_entry in not_known_ab:
                    not_known_entries.append({
                        'antibiotic': ab_entry,
                        'source_index': source_index,
                        'source_title': source_title,
                        'source_url': source_url,
                        'source_text': source_text
                    })
        
        if not not_known_entries:
            logger.info("No 'not_known' antibiotics to rank")
            return {}
        
        logger.info(f"Ranking {len(not_known_entries)} antibiotics from 'not_known' category")
        
        # We will track rankings made in THIS loop only (not from existing categories)
        # This ensures consistency within the same ranking session
        # Note: Antibiotic names should already be normalized during extraction
        
        # Get LLM
        ollama_config = get_ollama_config()
        model = ollama_config['model'].replace('ollama/', '')
        base_url = ollama_config['api_base']
        
        llm: BaseChatModel = ChatOllama(
            model=model,
            base_url=base_url,
            format='json',
            temperature=0,  # Force 0 for consistency
        )
        
        # Process each 'not_known' entry individually for better accuracy
        ranked_updates = {}  # {source_index: {medical_name: new_category}}
        
        for entry_info in not_known_entries:
            ab_entry = entry_info['antibiotic']
            source_index = entry_info['source_index']
            source_title = entry_info['source_title']
            source_text = entry_info['source_text']
            
            medical_name = ab_entry.get('medical_name', '')
            if not medical_name:
                continue
            
            # STEP 1: Check if we've already ranked this antibiotic in the current loop
            # Collect all antibiotics we've ranked so far in this session
            current_session_rankings = {}
            for src_idx, updates_dict in ranked_updates.items():
                for ab_name, update_info in updates_dict.items():
                    if ab_name not in current_session_rankings:
                        current_session_rankings[ab_name] = update_info['new_category']
            
            # Check if already ranked in current session - if yes, use that (highest priority)
            if medical_name in current_session_rankings:
                existing_category = current_session_rankings[medical_name]
                logger.info(f"Antibiotic {medical_name} was already ranked in this session as {existing_category}, maintaining consistency")
                
                # Store the update with the same category
                if source_index not in ranked_updates:
                    ranked_updates[source_index] = {}
                ranked_updates[source_index][medical_name] = {
                    'new_category': existing_category,
                    'reason': f'Maintaining consistency with ranking in this session as {existing_category}'
                }
                continue
            
            # STEP 2: Check if antibiotic exists in existing categories from extraction output
            # Only check if not already in current_session_rankings
            # Since names are already normalized, we can do direct string matching
            category_counts = {
                'first_choice': 0,
                'second_choice': 0,
                'alternative_antibiotic': 0
            }
            
            for source_result in source_results:
                therapy_plan = source_result.get('antibiotic_therapy_plan', {})
                for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                    antibiotics = therapy_plan.get(category, [])
                    if isinstance(antibiotics, list):
                        for ab_entry_existing in antibiotics:
                            if isinstance(ab_entry_existing, dict):
                                ab_name = ab_entry_existing.get('medical_name', '')
                                if ab_name and ab_name.lower() == medical_name.lower():
                                    category_counts[category] += 1
            
            # Determine best category (highest count, or highest priority if tied)
            priority = {'first_choice': 1, 'second_choice': 2, 'alternative_antibiotic': 3}
            max_count = max(category_counts.values()) if category_counts.values() else 0
            
            if max_count > 0:
                best_category = None
                best_priority = 99
                for category, count in category_counts.items():
                    if count == max_count:
                        cat_priority = priority.get(category, 99)
                        if cat_priority < best_priority:
                            best_priority = cat_priority
                            best_category = category
                
                if best_category:
                    logger.info(f"Antibiotic {medical_name} found in existing categories: "
                              f"{category_counts}, choosing {best_category}")
                    
                    # Store the update with the category from existing sources
                    if source_index not in ranked_updates:
                        ranked_updates[source_index] = {}
                    ranked_updates[source_index][medical_name] = {
                        'new_category': best_category,
                        'reason': f'Found in existing extraction output: {category_counts}, assigned to {best_category}'
                    }
                    # Add to current_session_rankings for consistency in subsequent iterations
                    # This will be picked up in the next iteration when building current_session_rankings
                    continue
            
            # STEP 3: Build list of previously ranked antibiotics for LLM context
            # This includes antibiotics from existing categories and those ranked in current session
            # This ensures consistency - if we found "Linezolid" in existing categories as "first_choice",
            # and now we're ranking another antibiotic, the LLM will see "Linezolid → first_choice" in the history
            previous_rankings_text = ""
            if current_session_rankings:
                previous_rankings_list = []
                for prev_name, prev_category in current_session_rankings.items():
                    previous_rankings_list.append(f"- {prev_name} → {prev_category}")
                previous_rankings_text = "\n".join(previous_rankings_list)
            else:
                previous_rankings_text = "None (this is the first antibiotic being ranked in this session)"
            
            # Prepare ranking prompt for this specific antibiotic
            ranking_prompt = f"""You are a medical expert ranking an antibiotic for {pathogen_name} with {resistant_gene} resistance.

Patient context:
- Pathogen: {pathogen_name}
- Resistance Gene(s): {resistant_gene}
- ICD Code(s): {severity_codes}
- Age: {age if age else 'Not specified'}

Antibiotic to rank:
- Name: {medical_name}
- Coverage: {ab_entry.get('coverage_for', 'Not specified')}
- Route: {ab_entry.get('route_of_administration', 'Not specified')}
- Dose: {ab_entry.get('dose_duration', 'Not specified')}
- Considerations: {ab_entry.get('general_considerations', 'Not specified')}

Source context:
- Title: {source_title}
- Text excerpt: {source_text}

IMPORTANT - Ranking History (for consistency):
The following antibiotics have already been ranked for this same condition in this ranking session. You MUST maintain consistency:
{previous_rankings_text}

CRITICAL CONSISTENCY RULE: 
- If {medical_name} is the EXACT SAME antibiotic as any listed above, you MUST assign it to the SAME category as shown in the history above.
- For example, if "Linezolid" was already ranked as "first_choice" above, and you are now ranking "Linezolid" again, you MUST assign it as "first_choice" again, NOT "second_choice" or any other category.
- Only assign a different category if it is clearly a DIFFERENT antibiotic (different drug, not just a different source).
- Maintaining consistency is more important than source-specific nuances.

TASK: Based on your medical knowledge, the source text, patient condition, and resistance pattern, rank this antibiotic into ONE category:
- first_choice: Best/preferred option for this condition
- second_choice: Good alternative if first-choice unavailable
- alternative_antibiotic: Other viable option

Provide:
- ranked_category: 'first_choice', 'second_choice', or 'alternative_antibiotic'
- ranking_reason: Brief explanation (1-2 sentences) of why it's ranked in this category

Consider:
1. Efficacy against {pathogen_name} with {resistant_gene} resistance
2. Clinical guidelines and evidence
3. Safety profile
4. Patient factors (age, ICD codes: {severity_codes})
5. Information from the source text
6. CONSISTENCY with previously ranked antibiotics (if same/similar antibiotic, use same category)
7. Appropriateness for the patient's ICD codes ({severity_codes}) - prioritize antibiotics suitable for these conditions

Return ONLY the ranked category and reason for this ONE antibiotic."""
            
            # Use structured output
            structured_llm = llm.with_structured_output(RankedAntibioticsResult)
            
            try:
                ranked_result = structured_llm.invoke(ranking_prompt)
                
                if ranked_result and ranked_result.ranked_antibiotics:
                    ranked_entry = ranked_result.ranked_antibiotics[0]
                    new_category = ranked_entry.ranked_category
                    
                    # Store the update
                    # This will be picked up by current_session_rankings in subsequent iterations
                    if source_index not in ranked_updates:
                        ranked_updates[source_index] = {}
                    ranked_updates[source_index][medical_name] = {
                        'new_category': new_category,
                        'reason': ranked_entry.ranking_reason
                    }
                    # After LLM ranking, this antibiotic is now in ranked_updates
                    # It will be automatically included in current_session_rankings for subsequent iterations
                    
                    logger.info(f"Ranked {medical_name} from source {source_index} as {new_category}: {ranked_entry.ranking_reason[:50]}...")
            except Exception as e:
                logger.warning(f"Error ranking {medical_name} from source {source_index}: {e}")
                continue
        
        # Update source_results with new categories
        updated_source_results = []
        for source_result in source_results:
            source_index = source_result.get('source_index', 0)
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            if source_index in ranked_updates:
                # Get updates for this source
                updates = ranked_updates[source_index]
                not_known_ab = therapy_plan.get('not_known', [])
                
                # Move antibiotics from not_known to their new categories
                first_choice = list(therapy_plan.get('first_choice', []))
                second_choice = list(therapy_plan.get('second_choice', []))
                alternative = list(therapy_plan.get('alternative_antibiotic', []))
                remaining_not_known = []
                
                for ab_entry in not_known_ab:
                    ab_name = ab_entry.get('medical_name', '')
                    
                    # Check if this antibiotic is in updates
                    if ab_name in updates:
                        matched_update = updates[ab_name]
                        
                        # Move to new category
                        new_category = matched_update['new_category']
                        if new_category == 'first_choice':
                            first_choice.append(ab_entry)
                        elif new_category == 'second_choice':
                            second_choice.append(ab_entry)
                        elif new_category == 'alternative_antibiotic':
                            alternative.append(ab_entry)
                        logger.info(f"Moved {ab_name} from not_known to {new_category} in source {source_index}")
                    else:
                        # Keep in not_known if not ranked
                        remaining_not_known.append(ab_entry)
                
                # Update therapy plan
                therapy_plan['first_choice'] = first_choice
                therapy_plan['second_choice'] = second_choice
                therapy_plan['alternative_antibiotic'] = alternative
                therapy_plan['not_known'] = remaining_not_known
                
                source_result['antibiotic_therapy_plan'] = therapy_plan
            
            updated_source_results.append(source_result)
        
        logger.info(f"Ranked and re-categorized antibiotics. Updated {len(ranked_updates)} sources")
        
        return {
            'source_results': updated_source_results
        }
        
    except Exception as e:
        logger.error(f"Error in rank_node: {e}", exc_info=True)
        return {}
