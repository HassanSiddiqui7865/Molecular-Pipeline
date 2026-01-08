"""
Rank node for LangGraph - Groups antibiotics by name and places them in final ranked categories.
Hybrid: Confidence-Interval + Hierarchical Fallback
Antibiotics are already filtered during extraction based on input parameters.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

from utils import (
    normalize_antibiotic_name
)

logger = logging.getLogger(__name__)



def _calculate_rank_score(
    first_choice: int,
    second_choice: int,
    alternative: int,
    total_sources: int
) -> Dict[str, Any]:
    """
    Calculate ranking score and determine final category using Hybrid algorithm.
    
    Algorithm:
    1. Calculate consensus percentages
    2. Apply hierarchical rules:
       - >60% consensus → use that category
       - 40-60% consensus → weighted majority
       - <40% consensus → weighted majority with confidence check
    3. Apply confidence threshold (downgrade if <20% or <2 occurrences)
    4. Handle ties with category hierarchy
    
    Args:
        first_choice: Raw count in first_choice
        second_choice: Raw count in second_choice
        alternative: Raw count in alternative_antibiotic
        total_sources: Total number of sources
    
    Returns:
        Dict with final_category, confidence_score, ranking_reason, etc.
    """
    total_occurrences = first_choice + second_choice + alternative
    
    if total_occurrences == 0:
        return {
            'final_category': 'alternative_antibiotic',
            'confidence_score': 0.0,
            'ranking_reason': 'No occurrences found',
            'consensus_percentage': {},
            'weighted_score': 0.0
        }
    
    # Use raw counts
    w_first = float(first_choice)
    w_second = float(second_choice)
    w_alt = float(alternative)
    
    # Calculate percentages
    valid_total = w_first + w_second + w_alt
    if valid_total == 0:
        return {
            'final_category': 'alternative_antibiotic',
            'confidence_score': 0.0,
            'ranking_reason': 'No valid occurrences found',
            'consensus_percentage': {
                'first_choice': 0.0,
                'second_choice': 0.0,
                'alternative_antibiotic': 0.0
            },
            'weighted_score': 0.0
        }
    
    first_pct = w_first / valid_total if valid_total > 0 else 0.0
    second_pct = w_second / valid_total if valid_total > 0 else 0.0
    alt_pct = w_alt / valid_total if valid_total > 0 else 0.0
    
    # Calculate weighted scores (category hierarchy weights)
    category_weights = {
        'first_choice': 4.0,
        'second_choice': 3.0,
        'alternative_antibiotic': 2.0
    }
    
    weighted_scores = {
        'first_choice': w_first * category_weights['first_choice'],
        'second_choice': w_second * category_weights['second_choice'],
        'alternative_antibiotic': w_alt * category_weights['alternative_antibiotic']
    }
    
    # Confidence: how many sources mentioned this antibiotic
    confidence = min(total_occurrences / total_sources, 1.0) if total_sources > 0 else 0.0
    
    # Determine final category using hierarchical rules
    final_category = None
    ranking_reason = ""
    
    # Rule 1: Strong consensus (>60%)
    if first_pct > 0.6:
        final_category = 'first_choice'
        ranking_reason = f"Strong consensus: {first_pct:.0%} categorize as first_choice"
    elif second_pct > 0.6:
        final_category = 'second_choice'
        ranking_reason = f"Strong consensus: {second_pct:.0%} categorize as second_choice"
    elif alt_pct > 0.6:
        final_category = 'alternative_antibiotic'
        ranking_reason = f"Strong consensus: {alt_pct:.0%} categorize as alternative"
    
    # Rule 2: Moderate consensus (40-60%) → weighted majority
    elif first_pct >= 0.4 or second_pct >= 0.4 or alt_pct >= 0.4:
        max_weighted = max(weighted_scores.items(), key=lambda x: x[1])
        final_category = max_weighted[0]
        
        # Build reason
        parts = []
        if first_choice > 0:
            parts.append(f"{first_choice} first_choice")
        if second_choice > 0:
            parts.append(f"{second_choice} second_choice")
        if alternative > 0:
            parts.append(f"{alternative} alternative")
        
        ranking_reason = f"Moderate consensus ({max_weighted[0]}: {max_weighted[1]:.1f} score) - {', '.join(parts)}"
    
    # Rule 3: Low consensus (<40%) → weighted majority with confidence check
    else:
        max_weighted = max(weighted_scores.items(), key=lambda x: x[1])
        final_category = max_weighted[0]
        
        parts = []
        if first_choice > 0:
            parts.append(f"{first_choice} first_choice")
        if second_choice > 0:
            parts.append(f"{second_choice} second_choice")
        if alternative > 0:
            parts.append(f"{alternative} alternative")
        
        ranking_reason = f"Low consensus - weighted majority: {', '.join(parts)} (score: {max_weighted[1]:.1f})"
    
    # Rule 4: Low confidence threshold → downgrade one level
    min_confidence = 0.2  # Must appear in at least 20% of sources
    min_occurrences = 2  # Or at least 2 occurrences
    
    original_category = final_category
    
    if confidence < min_confidence or total_occurrences < min_occurrences:
        if final_category == 'first_choice':
            final_category = 'second_choice'
            ranking_reason += f" (downgraded from first_choice due to low confidence: {confidence:.0%}, {total_occurrences} occurrences)"
        elif final_category == 'second_choice':
            final_category = 'alternative_antibiotic'
            ranking_reason += f" (downgraded from second_choice due to low confidence: {confidence:.0%}, {total_occurrences} occurrences)"
        # alternative_antibiotic stays as is (lowest valid category)
    
    return {
        'final_category': final_category,
        'confidence_score': confidence,
        'ranking_reason': ranking_reason,
        'weighted_score': weighted_scores.get(final_category, 0.0),
            'consensus_percentage': {
                'first_choice': first_pct,
                'second_choice': second_pct,
                'alternative_antibiotic': alt_pct
            },
        'original_category_before_downgrade': original_category if original_category != final_category else None
    }


def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that groups antibiotics by name and reorganizes them into final ranked categories.
    Antibiotics are already filtered during extraction based on input parameters.
    Keeps same source_results structure but places antibiotics in their final categories.
    """
    try:
        source_results = state.get('source_results', [])
        
        if not source_results:
            logger.info("No source results to rank")
            return {'source_results': []}
        
        # Step 1: Collect all antibiotics from all sources, grouped by normalized name
        antibiotic_groups = defaultdict(lambda: {
            'first_choice': 0,
            'second_choice': 0,
            'alternative_antibiotic': 0,
            'original_names': set()
        })
        
        # Track original categories per antibiotic (for logging movements)
        antibiotic_original_categories = defaultdict(lambda: {
            'first_choice': 0,
            'second_choice': 0,
            'alternative_antibiotic': 0
        })
        
        total_sources = len(source_results)
        
        # Get progress callback from metadata if available
        metadata = state.get('metadata', {})
        progress_callback = metadata.get('progress_callback')
        
        # Emit progress for ranking start
        if progress_callback:
            progress_callback('rank', 0, 'Starting ranking...')
        
        # Step 1: Collect all antibiotics from all sources, grouped by normalized name
        for source_result in source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            # Count occurrences in each category
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict):
                            medical_name = ab_entry.get('medical_name', '').strip()
                            if medical_name:
                                normalized_name = normalize_antibiotic_name(medical_name)
                                antibiotic_groups[normalized_name][category] += 1
                                antibiotic_groups[normalized_name]['original_names'].add(medical_name)
                                antibiotic_original_categories[normalized_name][category] += 1
        
        # Emit progress after collection
        if progress_callback:
            progress_callback('rank', 20, f'Collected {len(antibiotic_groups)} unique antibiotics')
        
        # Step 2: Rank each antibiotic group and determine final category
        final_categories = {}  # normalized_name -> final_category
        total_antibiotics = len(antibiotic_groups)
        ranked_count = 0
        
        for normalized_name, group_data in antibiotic_groups.items():
            rank_result = _calculate_rank_score(
                first_choice=group_data['first_choice'],
                second_choice=group_data['second_choice'],
                alternative=group_data['alternative_antibiotic'],
                total_sources=total_sources
            )
            
            final_categories[normalized_name] = rank_result['final_category']
            ranked_count += 1
            
            # Emit progress during ranking
            if progress_callback and total_antibiotics > 0:
                sub_progress = 30 + (ranked_count / total_antibiotics * 20)  # 30-50% for ranking
                progress_callback('rank', sub_progress, f'Ranked {ranked_count}/{total_antibiotics} antibiotics')
        
        # Emit progress after ranking calculation
        if progress_callback:
            progress_callback('rank', 50, 'Reorganizing antibiotics into final categories...')
        
        # Step 3: Reorganize each source_result - place antibiotics in their final categories
        updated_source_results = []
        total_sources_to_reorganize = len(source_results)
        
        for idx, source_result in enumerate(source_results):
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            # Collect all antibiotics from this source
            all_antibiotics = []
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict) and ab_entry.get('medical_name'):
                            all_antibiotics.append(ab_entry)
            
            # Reorganize into final categories
            reorganized_plan = {
                'first_choice': [],
                'second_choice': [],
                'alternative_antibiotic': []
            }
            
            for ab_entry in all_antibiotics:
                medical_name = ab_entry.get('medical_name', '').strip()
                if medical_name:
                    normalized_name = normalize_antibiotic_name(medical_name)
                    final_category = final_categories.get(normalized_name, 'alternative_antibiotic')
                    if final_category in reorganized_plan:
                        reorganized_plan[final_category].append(ab_entry)
            
            # Create updated source result
            updated_source_result = source_result.copy()
            updated_source_result['antibiotic_therapy_plan'] = reorganized_plan
            updated_source_results.append(updated_source_result)
            
            # Emit progress during reorganization
            if progress_callback and total_sources_to_reorganize > 0:
                sub_progress = 50 + ((idx + 1) / total_sources_to_reorganize * 20)  # 50-70% for reorganization
                progress_callback('rank', sub_progress, f'Reorganized {idx + 1}/{total_sources_to_reorganize} sources')
        
        # Filter routes based on systemic flag before counting
        input_params = state.get('input_parameters', {})
        systemic = input_params.get('systemic')
        systemic_routes = {'IV', 'PO', 'IM'}
        
        antibiotics_to_remove = {
            'first_choice': [],
            'second_choice': [],
            'alternative_antibiotic': []
        }
        
        for source_result in updated_source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                antibiotics = therapy_plan.get(category, [])
                if not isinstance(antibiotics, list):
                    continue
                
                for idx, ab_entry in enumerate(antibiotics):
                    if not isinstance(ab_entry, dict):
                        continue
                    
                    route = ab_entry.get('route_of_administration')
                    if route is None:
                        route = ''
                    else:
                        route = str(route).strip()
                    
                    medical_name = ab_entry.get('medical_name', 'unknown')
                    
                    # Always filter out null/empty routes
                    if not route:
                        logger.warning(f"  [Route Filter] Filtered out {medical_name}: route is null/empty")
                        antibiotics_to_remove[category].append((source_result, idx))
                        continue
                    
                    # Filter based on systemic flag (only if systemic is not None)
                    if systemic is not None:
                        # Check if route is one of the systemic routes (IV, PO, IM)
                        route_upper = route.upper()
                        route_is_systemic = route_upper in systemic_routes
                        
                        if systemic is True:
                            # Only keep routes that are IV, PO, or IM
                            if not route_is_systemic:
                                logger.warning(f"  [Route Filter] Filtered out {medical_name}: route '{route}' is not IV, PO, or IM (systemic=True)")
                                antibiotics_to_remove[category].append((source_result, idx))
                        elif systemic is False:
                            # Exclude routes that are IV, PO, or IM
                            if route_is_systemic:
                                logger.warning(f"  [Route Filter] Filtered out {medical_name}: route '{route}' is IV, PO, or IM (systemic=False)")
                                antibiotics_to_remove[category].append((source_result, idx))
                        # If systemic is None, only null routes are filtered (already handled above)
            
            # Remove filtered antibiotics from source results
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                if antibiotics_to_remove[category]:
                    # Group by source_result
                    source_to_indices = {}
                    for source_result, idx in antibiotics_to_remove[category]:
                        source_id = id(source_result)
                        if source_id not in source_to_indices:
                            source_to_indices[source_id] = (source_result, [])
                        source_to_indices[source_id][1].append(idx)
                    
                    # Remove from each source in reverse order
                    for source_result, indices in source_to_indices.values():
                        therapy_plan = source_result.get('antibiotic_therapy_plan', {})
                        antibiotics = therapy_plan.get(category, [])
                        if isinstance(antibiotics, list):
                            for idx in sorted(set(indices), reverse=True):
                                if 0 <= idx < len(antibiotics):
                                    removed_ab = antibiotics.pop(idx)
                                    ab_name = removed_ab.get('medical_name', 'unknown') if isinstance(removed_ab, dict) else 'unknown'
                                    logger.info(f"  [Route Filter] Removed {ab_name} from {category} (route does not match systemic={systemic})")
        
        # Emit progress for ranking complete
        if progress_callback:
            progress_callback('rank', 100, 'Ranking complete')
        
        # Count antibiotics in each final category
        category_counts = {'first_choice': 0, 'second_choice': 0, 'alternative_antibiotic': 0}
        all_assignments = []
        
        # Use sets to track unique antibiotics per category
        category_antibiotics = {
            'first_choice': set(),
            'second_choice': set(),
            'alternative_antibiotic': set()
        }
        final_categories = {}
        
        for source_result in updated_source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict):
                            medical_name = ab_entry.get('medical_name', '').strip()
                            if medical_name:
                                normalized_name = normalize_antibiotic_name(medical_name)
                                category_antibiotics[category].add(normalized_name)
                                final_categories[normalized_name] = category
        
        # Count unique antibiotics per category
        for category, antibiotics_set in category_antibiotics.items():
            category_counts[category] = len(antibiotics_set)
        
        # Log assignments for all antibiotics
        for normalized_name, final_category in final_categories.items():
            # Get occurrence counts
            group_data = antibiotic_groups.get(normalized_name, {})
            original_cats = antibiotic_original_categories.get(normalized_name, {})
            original_name = sorted(group_data.get('original_names', set()))[0] if group_data.get('original_names') else normalized_name
            
            # Build count string
            counts = []
            if original_cats.get('first_choice', 0) > 0:
                counts.append(f"first={original_cats['first_choice']}")
            if original_cats.get('second_choice', 0) > 0:
                counts.append(f"second={original_cats['second_choice']}")
            if original_cats.get('alternative_antibiotic', 0) > 0:
                counts.append(f"alt={original_cats['alternative_antibiotic']}")
            
            count_str = f"({', '.join(counts)})" if counts else "(0)"
            total_occurrences = sum(original_cats.values())
            
            # Determine original category (most common occurrence)
            max_original = max(original_cats.items(), key=lambda x: x[1]) if original_cats else None
            
            if max_original and max_original[1] > 0:
                original_category = max_original[0]
                arrow = "→" if original_category != final_category else "="
                all_assignments.append({
                    'name': original_name,
                    'from': original_category,
                    'to': final_category,
                    'arrow': arrow,
                    'count_str': count_str,
                    'total': total_occurrences
                })
            else:
                all_assignments.append({
                    'name': original_name,
                    'from': 'alternative_antibiotic',
                    'to': final_category,
                    'arrow': '→',
                    'count_str': count_str,
                    'total': total_occurrences
                })
        
        logger.info(
            f"Ranked {len(final_categories)} antibiotics: "
            f"{category_counts['first_choice']} first_choice, "
            f"{category_counts['second_choice']} second_choice, "
            f"{category_counts['alternative_antibiotic']} alternative"
        )
        
        # Log all category assignments with counts
        for assignment in sorted(all_assignments, key=lambda x: (x['to'], x['name'])):
            logger.info(
                f"  {assignment['name']}: {assignment['from']} {assignment['arrow']} {assignment['to']} "
                f"{assignment['count_str']} (total: {assignment['total']})"
            )
        
        # Emit progress for counting complete
        if progress_callback:
            progress_callback('rank', 90, 'Finalizing ranking results...')
        
        # Save results
        _save_rank_results(input_params, updated_source_results)
        
        # Emit progress for ranking complete
        if progress_callback:
            progress_callback('rank', 100, 'Ranking complete')
        
        return {
            'source_results': updated_source_results  # Same structure, antibiotics reorganized
        }
        
    except Exception as e:
        logger.error(f"Error in rank_node: {e}", exc_info=True)
        # Record error in state
        errors = state.get('errors', [])
        errors.append(f"Rank node error: {str(e)}")
        return {'source_results': state.get('source_results', []), 'errors': errors}


def _save_rank_results(
    input_params: Dict, 
    source_results: List[Dict]
) -> None:
    """Save rank results to file."""
    try:
        from config import get_output_config
        output_config = get_output_config()
        
        # Check if saving is enabled
        if not output_config.get('save_enabled', True):
            logger.debug("Saving rank results disabled (production mode)")
            return
        
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "rank_result.json"
        
        result_data = {
            'input_parameters': input_params,
            'source_results': source_results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rank results saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save rank results: {e}")
