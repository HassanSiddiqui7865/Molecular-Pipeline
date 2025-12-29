"""
Rank node for LangGraph - Groups antibiotics by name and places them in final ranked categories.
Hybrid: Confidence-Interval + Hierarchical Fallback
Filters out unsafe/ineffective antibiotics using LLM before ranking.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from collections import defaultdict

from schemas import FilteredAntibioticsResult
from prompts import ANTIBIOTIC_FILTERING_PROMPT_TEMPLATE
from utils import (
    format_resistance_genes, get_icd_names_from_state,
    get_pathogens_from_input, format_pathogens,
    get_resistance_genes_from_input
)
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


def _filter_antibiotics_with_llm(
    unique_antibiotics: List[str],
    pathogen_display: str,
    resistant_gene: str,
    severity_codes: str,
    age: Optional[int],
    sample: Optional[str],
    systemic: Optional[bool],
    retry_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Filter antibiotics using LLM to remove unsafe, toxic, or ineffective ones.
    
    Args:
        unique_antibiotics: List of unique antibiotic names to filter
        pathogen_display: Pathogen name for context
        resistant_gene: Resistance gene for context
        severity_codes: ICD severity codes for context
        age: Patient age (optional)
        sample: Sample type (optional)
        systemic: Systemic flag (optional)
        retry_delay: Initial delay between retries in seconds
        
    Returns:
        Dict with:
        - 'antibiotics_to_keep': Set of antibiotic names that should be KEPT
        - 'filtered_out': List of dicts with 'medical_name' and 'filtering_reason'
    """
    if not unique_antibiotics:
        return {
            'antibiotics_to_keep': set(),
            'filtered_out': []
        }
    
    llm = _create_llm()
    if not llm:
        logger.warning("LlamaIndex LLM not available, skipping antibiotic filtering")
        return {
            'antibiotics_to_keep': set(unique_antibiotics),
            'filtered_out': []
        }
    
    # Format antibiotic list for prompt
    antibiotic_list = "\n".join([f"- {ab}" for ab in unique_antibiotics])
    
    # Format prompt
    prompt = ANTIBIOTIC_FILTERING_PROMPT_TEMPLATE.format(
        pathogen_display=pathogen_display,
        resistant_gene=resistant_gene,
        severity_codes=severity_codes,
        age=f"{age} years" if age else 'Not specified',
        sample=sample or 'Not specified',
        systemic='Yes' if systemic else 'No',
        antibiotic_list=antibiotic_list
    )
    
    attempt = 0
    while True:
        attempt += 1
        try:
            program = LLMTextCompletionProgram.from_defaults(
                output_cls=FilteredAntibioticsResult,
                llm=llm,
                prompt_template_str="{input_str}",
                verbose=False
            )
            
            result = program(input_str=prompt)
            
            if not result:
                logger.warning(f"Empty filtering result (attempt {attempt}), retrying...")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
            
            result_dict = result.model_dump()
            filtered_antibiotics = result_dict.get('filtered_antibiotics', [])
            
            # Build set of antibiotics to keep and list of filtered out ones
            antibiotics_to_keep = set()
            filtered_out = []
            
            for entry in filtered_antibiotics:
                medical_name = entry.get('medical_name', '').strip()
                should_keep = entry.get('should_keep', True)
                filtering_reason = entry.get('filtering_reason')
                
                if should_keep:
                    antibiotics_to_keep.add(medical_name)
                else:
                    filtered_out.append({
                        'medical_name': medical_name,
                        'filtering_reason': filtering_reason or 'No reason provided'
                    })
                    logger.info(f"Filtered out: {medical_name} - {filtering_reason}")
            
            logger.info(
                f"Antibiotic filtering: {len(antibiotics_to_keep)} kept, "
                f"{len(filtered_out)} filtered out from {len(unique_antibiotics)} total"
            )
            
            return {
                'antibiotics_to_keep': antibiotics_to_keep,
                'filtered_out': filtered_out
            }
            
        except Exception as e:
            logger.warning(f"Error in antibiotic filtering (attempt {attempt}): {e}")
            logger.info(f"Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)


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


def _calculate_rank_score(
    first_choice: int,
    second_choice: int,
    alternative: int,
    not_known: int,
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
        not_known: Raw count in not_known
        total_sources: Total number of sources
    
    Returns:
        Dict with final_category, confidence_score, ranking_reason, etc.
    """
    total_occurrences = first_choice + second_choice + alternative + not_known
    
    if total_occurrences == 0:
        return {
            'final_category': 'not_known',
            'confidence_score': 0.0,
            'ranking_reason': 'No occurrences found',
            'consensus_percentage': {},
            'weighted_score': 0.0
        }
    
    # Use raw counts
    w_first = float(first_choice)
    w_second = float(second_choice)
    w_alt = float(alternative)
    
    # Calculate percentages (exclude not_known from final ranking)
    valid_total = w_first + w_second + w_alt
    if valid_total == 0:
        return {
            'final_category': 'alternative_antibiotic',
            'confidence_score': 0.0,
            'ranking_reason': 'All occurrences in not_known category',
            'consensus_percentage': {
                'first_choice': 0.0,
                'second_choice': 0.0,
                'alternative_antibiotic': 0.0,
                'not_known': 1.0
            },
            'weighted_score': 0.0
        }
    
    first_pct = w_first / valid_total if valid_total > 0 else 0.0
    second_pct = w_second / valid_total if valid_total > 0 else 0.0
    alt_pct = w_alt / valid_total if valid_total > 0 else 0.0
    not_known_pct = not_known / total_occurrences if total_occurrences > 0 else 0.0
    
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
        if not_known > 0:
            parts.append(f"{not_known} not_known")
        
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
        if not_known > 0:
            parts.append(f"{not_known} not_known")
        
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
            'alternative_antibiotic': alt_pct,
            'not_known': not_known_pct
        },
        'original_category_before_downgrade': original_category if original_category != final_category else None
    }


def rank_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that groups antibiotics by name and reorganizes them into final ranked categories.
    First ranks antibiotics, then filters out unsafe/ineffective ones using LLM.
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
            'not_known': 0,
            'original_names': set()
        })
        
        # Track original categories per antibiotic (for logging movements)
        antibiotic_original_categories = defaultdict(lambda: {
            'first_choice': 0,
            'second_choice': 0,
            'alternative_antibiotic': 0,
            'not_known': 0
        })
        
        total_sources = len(source_results)
        
        # Step 1: Collect all antibiotics from all sources, grouped by normalized name
        for source_result in source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            # Count occurrences in each category
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict):
                            medical_name = ab_entry.get('medical_name', '').strip()
                            if medical_name:
                                normalized_name = _normalize_antibiotic_name(medical_name)
                                antibiotic_groups[normalized_name][category] += 1
                                antibiotic_groups[normalized_name]['original_names'].add(medical_name)
                                antibiotic_original_categories[normalized_name][category] += 1
        
        # Step 2: Rank each antibiotic group and determine final category
        final_categories = {}  # normalized_name -> final_category
        
        for normalized_name, group_data in antibiotic_groups.items():
            rank_result = _calculate_rank_score(
                first_choice=group_data['first_choice'],
                second_choice=group_data['second_choice'],
                alternative=group_data['alternative_antibiotic'],
                not_known=group_data['not_known'],
                total_sources=total_sources
            )
            
            final_categories[normalized_name] = rank_result['final_category']
        
        # Step 3: Reorganize each source_result - place antibiotics in their final categories
        updated_source_results = []
        
        for source_result in source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            
            # Collect all antibiotics from this source
            all_antibiotics = []
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict) and ab_entry.get('medical_name'):
                            all_antibiotics.append(ab_entry)
            
            # Reorganize into final categories
            reorganized_plan = {
                'first_choice': [],
                'second_choice': [],
                'alternative_antibiotic': [],
                'not_known': []
            }
            
            for ab_entry in all_antibiotics:
                medical_name = ab_entry.get('medical_name', '').strip()
                if medical_name:
                    normalized_name = _normalize_antibiotic_name(medical_name)
                    final_category = final_categories.get(normalized_name, 'not_known')
                    reorganized_plan[final_category].append(ab_entry)
            
            # Create updated source result
            updated_source_result = source_result.copy()
            updated_source_result['antibiotic_therapy_plan'] = reorganized_plan
            updated_source_results.append(updated_source_result)
        
        # Step 4: Filter antibiotics using LLM after ranking
        # Collect unique antibiotics from ranked results
        all_unique_antibiotics = set()
        for source_result in updated_source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict):
                            medical_name = ab_entry.get('medical_name', '').strip()
                            if medical_name:
                                all_unique_antibiotics.add(medical_name)
        
        # Filter antibiotics using LLM if we have any
        antibiotics_to_keep = set(all_unique_antibiotics)
        filtered_out_antibiotics = []
        
        if all_unique_antibiotics:
            # Get patient context from state
            input_params = state.get('input_parameters', {})
            pathogens = get_pathogens_from_input(input_params)
            pathogen_display = format_pathogens(pathogens) if pathogens else "unknown"
            
            resistant_genes = get_resistance_genes_from_input(input_params)
            resistant_gene = format_resistance_genes(resistant_genes) if resistant_genes else "unknown"
            
            severity_codes = get_icd_names_from_state(state)
            age = input_params.get('age')
            sample = input_params.get('sample')
            systemic = input_params.get('systemic')
            
            # Filter antibiotics
            filtering_result = _filter_antibiotics_with_llm(
                unique_antibiotics=list(all_unique_antibiotics),
                pathogen_display=pathogen_display,
                resistant_gene=resistant_gene,
                severity_codes=severity_codes,
                age=age,
                sample=sample,
                systemic=systemic
            )
            
            antibiotics_to_keep = filtering_result.get('antibiotics_to_keep', set(all_unique_antibiotics))
            filtered_out_antibiotics = filtering_result.get('filtered_out', [])
            
            # Remove filtered antibiotics from the ranked results
            for source_result in updated_source_results:
                therapy_plan = source_result.get('antibiotic_therapy_plan', {})
                for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                    antibiotics = therapy_plan.get(category, [])
                    if isinstance(antibiotics, list):
                        # Filter out antibiotics that should be removed
                        filtered_antibiotics = [
                            ab for ab in antibiotics
                            if isinstance(ab, dict) and ab.get('medical_name', '').strip() in antibiotics_to_keep
                        ]
                        therapy_plan[category] = filtered_antibiotics
                source_result['antibiotic_therapy_plan'] = therapy_plan
        
        # Count antibiotics in each final category after filtering
        category_counts = {'first_choice': 0, 'second_choice': 0, 'alternative_antibiotic': 0, 'not_known': 0}
        all_assignments = []
        
        # Recalculate final_categories based on what's left after filtering
        # Use sets to track unique antibiotics per category
        category_antibiotics = {
            'first_choice': set(),
            'second_choice': set(),
            'alternative_antibiotic': set(),
            'not_known': set()
        }
        filtered_final_categories = {}
        
        for source_result in updated_source_results:
            therapy_plan = source_result.get('antibiotic_therapy_plan', {})
            for category in ['first_choice', 'second_choice', 'alternative_antibiotic', 'not_known']:
                antibiotics = therapy_plan.get(category, [])
                if isinstance(antibiotics, list):
                    for ab_entry in antibiotics:
                        if isinstance(ab_entry, dict):
                            medical_name = ab_entry.get('medical_name', '').strip()
                            if medical_name:
                                normalized_name = _normalize_antibiotic_name(medical_name)
                                category_antibiotics[category].add(normalized_name)
                                filtered_final_categories[normalized_name] = category
        
        # Count unique antibiotics per category
        for category, antibiotics_set in category_antibiotics.items():
            category_counts[category] = len(antibiotics_set)
        
        # Log assignments only for antibiotics that passed filtering
        for normalized_name, final_category in filtered_final_categories.items():
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
            if original_cats.get('not_known', 0) > 0:
                counts.append(f"unknown={original_cats['not_known']}")
            
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
                    'from': 'not_known',
                    'to': final_category,
                    'arrow': '→',
                    'count_str': count_str,
                    'total': total_occurrences
                })
        
        logger.info(
            f"Ranked and filtered {len(filtered_final_categories)} antibiotics: "
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
        
        # Save results
        input_params = state.get('input_parameters', {})
        _save_rank_results(input_params, updated_source_results, filtered_out_antibiotics)
        
        return {
            'source_results': updated_source_results,  # Same structure, antibiotics reorganized
            'filtered_out_antibiotics': filtered_out_antibiotics  # Antibiotics filtered out with reasons
        }
        
    except Exception as e:
        logger.error(f"Error in rank_node: {e}", exc_info=True)
        return {'source_results': state.get('source_results', [])}


def _save_rank_results(
    input_params: Dict, 
    source_results: List[Dict],
    filtered_out_antibiotics: List[Dict] = None
) -> None:
    """Save rank results to file."""
    try:
        from config import get_output_config
        output_config = get_output_config()
        output_dir = Path(output_config.get('directory', 'output'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "rank_result.json"
        
        result_data = {
            'input_parameters': input_params,
            'source_results': source_results
        }
        
        # Include filtered out antibiotics if available
        if filtered_out_antibiotics:
            result_data['filtered_out_antibiotics'] = filtered_out_antibiotics
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Rank results saved to: {output_file}")
    except Exception as e:
        logger.warning(f"Failed to save rank results: {e}")
