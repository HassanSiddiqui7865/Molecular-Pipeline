"""
Extract node for LangGraph - Extracts information from search results using LangChain.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel

from schemas import SearchResult, AntibioticTherapyPlan, ResistanceGeneEntry

from utils import format_resistance_genes, get_icd_names_from_state

logger = logging.getLogger(__name__)


COMBINED_EXTRACTION_PROMPT = """Extract antibiotic information for {pathogen_display} with {resistant_gene} resistance.

PATIENT: Pathogen={pathogen_display} | Resistance={resistant_gene} | ICD={severity_codes} | Age={age} | Sample={sample} | Systemic={systemic}
{pathogens_context}
{resistance_genes_context}

SOURCE:
{content}

SELECTION:
- Extract antibiotics effective against {pathogen_display} with {resistant_gene}
- Match severity to ICD: {severity_codes}
- Consolidate duplicates unless clinically distinct

COMBINATIONS:
- Identify: "Drug1 and Drug2", "Drug1/Drug2", "Drug1-Drug2", hyphenated names
- Normalize to: "Drug1 plus Drug2" (lowercase "plus", title case)
- Examples: "Quinupristin-dalfopristin" → "Quinupristin plus Dalfopristin" | "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole"
- Set is_combined=True for ANY combination

CATEGORIES (only if explicitly stated):
- first_choice: "first-line", "preferred", "recommended", "primary"
- second_choice: "alternative", "second-line", "backup"
- alternative_antibiotic: "salvage", "last resort"
- not_known: Effective but category not stated

FIELDS:
- medical_name: Title case, no dosage/brand/route/salts. Normalize combinations to "Drug1 plus Drug2"
  Examples: "vancomycin 1g IV" → "Vancomycin" | "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole"
- is_combined: True if name contains "plus" after normalization, else False
- coverage_for: Specific indication using clinical terminology only (e.g., "MRSA bacteremia", "VRE bacteremia", "Staphylococcus aureus bacteremia"). Do NOT include ICD codes (e.g., A41.2) or ICD code names (e.g., "Sepsis due to..."). Use clinical terms like "bacteremia", "sepsis", "endocarditis".
- route_of_administration: "IV", "PO", "IM", "IV/PO", "Oral" (null if not mentioned)
- dose_duration: Natural text format for dosing information including ALL dosages (loading and maintenance) in concise way
  * Single: "600 mg PO q12h for 7 days"
  * With loading: "Loading: 1g IV, then 500 mg q12h for 7-14 days" or "1g IV once, then 500 mg q12h for 7-14 days"
  * Multiple phases: "450 mg q24h on Days 1 and 2, then 300 mg q24h for 7-14 days"
  * Combinations: "Trimethoprim 160 mg plus Sulfamethoxazole 800 mg PO q12h for 7 days"
  * EXCLUDE: monitoring details, target levels, infusion rates, administration notes (place in general_considerations)
- renal_adjustment: "Adjust dose for CrCl < X mL/min" or similar (null if not mentioned)
- general_considerations: Synthesize clinical notes concisely (null if none)

RESISTANCE GENES (for each from {resistant_gene}{resistance_genes_context}):
- detected_resistant_gene_name: Gene name (e.g., "mecA")
- potential_medication_class_affected: Affected classes
- general_considerations: Mechanism and impact (null if none)

VALIDATION:
- Each antibiotic in ONE category only
- medical_name: ALWAYS normalize combinations (never keep hyphen/slash)
- is_combined: True if hyphen (-), slash (/), "plus", or explicit "and/with" in original
- NEVER extract genes (mecA, vanA) as antibiotics
- dose_duration: Use natural text format, include all dosages for combinations
- Loading doses: Include loading doses if present - keep concise and natural (e.g., "Loading: 1g IV, then 500 mg q12h for 7-14 days")
- Synthesize intelligently (no verbatim copying)

EXCLUDE:
- Genes as antibiotics
- Ineffective/resistant antibiotics
- Drug classes without specific names
- Experimental drugs (unless recommended)
- Antibiotics in resistance/failure context only"""

class CombinedExtractionResult(BaseModel):
    """Combined schema for antibiotic therapy and resistance gene extraction."""
    antibiotic_therapy_plan: AntibioticTherapyPlan = Field(..., description="Extracted antibiotic therapy plan")
    pharmacist_analysis_on_resistant_gene: List[ResistanceGeneEntry] = Field(default_factory=list, description="Extracted resistance gene analysis")


def _extract_node_impl(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Internal implementation of extract node.
    """
    try:
        search_results = state.get('search_results', [])
        input_params = state.get('input_parameters', {})
        
        if not search_results:
            logger.warning("No search results to extract from")
            return {'source_results': []}
        
        # Get LangChain ChatModel
        from config import get_ollama_llm
        llm = get_ollama_llm()
        
        # Get pathogens
        from utils import get_pathogens_from_input, format_pathogens
        pathogens = get_pathogens_from_input(input_params)
        pathogen_display = format_pathogens(pathogens)
        # For extraction, use first pathogen as primary
        primary_pathogen = pathogens[0] if pathogens else {}
        pathogen_name = primary_pathogen.get('pathogen_name', '')
        pathogen_count = primary_pathogen.get('pathogen_count', '')
        
        # Get resistance genes
        from utils import get_resistance_genes_from_input, format_resistance_genes
        resistant_genes = get_resistance_genes_from_input(input_params)
        resistant_gene = format_resistance_genes(resistant_genes)
        
        # Get ICD names from state (transformed), fallback to codes
        severity_codes = get_icd_names_from_state(state)
        age = input_params.get('age')
        sample = input_params.get('sample', '')
        systemic = input_params.get('systemic', True)
        
        source_results = []
        
        # Process sources concurrently (2 at a time)
        def process_source(idx: int, result_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single source and return the result."""
            result = SearchResult(**result_data)
            logger.info(f"[{idx}] Processing: {result.title[:50]}...")
            
            # Prepare content with unique ID to prevent caching
            unique_id = str(uuid.uuid4())
            source_content = f"Title: {result.title}\nContent: {result.snippet}\n[Unique ID: {unique_id}]"
            
            # Extract both antibiotic therapy and resistance genes using LangChain
            # Use formatted display for prompts, but pass all pathogens for context
            extraction_result = _extract_combined(
                content=source_content,
                pathogen_name=pathogen_display,
                resistant_gene=resistant_gene,
                pathogen_count=pathogen_count,
                severity_codes=severity_codes,
                age=age,
                sample=sample,
                systemic=systemic,
                source_url=result.url,
                source_title=result.title,
                llm=llm,
                pathogens=pathogens,
                resistant_genes_list=resistant_genes
            )
            
            # Extract results
            therapy_result = extraction_result.get('antibiotic_therapy_plan', {})
            resistance_result = extraction_result.get('pharmacist_analysis_on_resistant_gene', [])
            
            # Create source result entry
            source_entry = {
                'source_url': result.url,
                'source_title': result.title,
                'source_index': idx,
                'antibiotic_therapy_plan': therapy_result or {},
                'pharmacist_analysis_on_resistant_gene': resistance_result or []
            }
            
            logger.info(f"[{idx}/{len(search_results)}] Completed")
            return source_entry
        
        # Use ThreadPoolExecutor to process 2 sources concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(process_source, idx + 1, result_data): idx + 1
                for idx, result_data in enumerate(search_results)
            }
            
            # Collect results as they complete (maintain order)
            results_dict = {}
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    result = future.result()
                    results_dict[idx] = result
                except Exception as e:
                    logger.error(f"[{idx}] Error processing source: {e}", exc_info=True)
                    raise
        
        # Sort by index to maintain original order
        source_results = [results_dict[i] for i in sorted(results_dict.keys())]
        
        # Save extraction result to file
        try:
            from config import get_output_config
            output_config = get_output_config()
            output_dir = Path(output_config.get('directory', 'output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            
            extraction_result_file = output_dir / "extraction_result.json"
            extraction_result_data = {
                'input_parameters': input_params,
                'source_results': source_results
            }
            
            with open(extraction_result_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_result_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Extraction result saved to: {extraction_result_file}")
        except Exception as e:
            logger.warning(f"Failed to save extraction result to file: {e}")
        
        return {'source_results': source_results}
        
    except Exception as e:
        logger.error(f"Error in extract_node: {e}", exc_info=True)
        raise


def extract_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node that extracts antibiotic therapy and resistance gene information from search results.
    Processes each source independently.
    """
    return _extract_node_impl(state)


def _extract_combined(
    content: str,
    pathogen_name: str,
    resistant_gene: str,
    pathogen_count: str,
    severity_codes: str,
    age: Optional[int],
    sample: str,
    systemic: bool,
    source_url: str,
    source_title: str,
    llm: BaseChatModel,
    pathogens: Optional[List[Dict[str, str]]] = None,
    resistant_genes_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract both antibiotic therapy plan and resistance genes using LangChain.
    Returns dictionary with both results.
    """
    # Build context for multiple pathogens if provided
    pathogens_context = ""
    if pathogens and len(pathogens) > 1:
        pathogens_list = []
        for p in pathogens:
            name = p.get('pathogen_name', '').strip()
            count = p.get('pathogen_count', '').strip()
            if name:
                if count:
                    pathogens_list.append(f"{name} ({count})")
                else:
                    pathogens_list.append(name)
        if pathogens_list:
            pathogens_context = f"\nAll Pathogens: {', '.join(pathogens_list)}"
    
    # Build context for multiple resistance genes if provided
    resistance_genes_context = ""
    if resistant_genes_list and len(resistant_genes_list) > 1:
        resistance_genes_context = f"\nAll Resistance Genes: {', '.join(resistant_genes_list)}"
    
    # Create prompt
    prompt = COMBINED_EXTRACTION_PROMPT.format(
        pathogen_display=pathogen_name,  # Already formatted
        resistant_gene=resistant_gene,  # Already formatted
        severity_codes=severity_codes,
        age=f"{age} years" if age else 'Not specified',
        sample=sample if sample else 'Not specified',
        systemic='Yes' if systemic else 'No',
        pathogens_context=pathogens_context,
        resistance_genes_context=resistance_genes_context,
        content=content
    )
    
    # Use LangChain structured output
    structured_llm = llm.with_structured_output(CombinedExtractionResult)
    
    try:
        result = structured_llm.invoke(prompt)
        
        if not result:
            logger.warning(f"LLM returned None for extraction from {source_title}")
            return {
                'antibiotic_therapy_plan': {
                    'first_choice': [],
                    'second_choice': [],
                    'alternative_antibiotic': []
                },
                'pharmacist_analysis_on_resistant_gene': []
            }
    except Exception as e:
        logger.error(f"Error during extraction from {source_title}: {e}")
        return {
            'antibiotic_therapy_plan': {
                'first_choice': [],
                'second_choice': [],
                'alternative_antibiotic': []
            },
            'pharmacist_analysis_on_resistant_gene': []
        }
    
    # Convert to dict
    result_dict = result.model_dump()
    
    # Log extraction summary
    therapy_plan_raw = result_dict.get('antibiotic_therapy_plan', {})
    first_count = len(therapy_plan_raw.get('first_choice', [])) if isinstance(therapy_plan_raw, dict) else 0
    second_count = len(therapy_plan_raw.get('second_choice', [])) if isinstance(therapy_plan_raw, dict) else 0
    alt_count = len(therapy_plan_raw.get('alternative_antibiotic', [])) if isinstance(therapy_plan_raw, dict) else 0
    logger.info(f"LLM extracted: first_choice={first_count}, second_choice={second_count}, alternative={alt_count}, resistance_genes={len(result_dict.get('pharmacist_analysis_on_resistant_gene', []))}")
    if first_count == 0 and second_count == 0 and alt_count == 0:
        logger.warning(f"No antibiotics extracted from source: {source_title}")
    
    # Extract therapy plan (remove confidence from nested structure if present)
    therapy_plan = therapy_plan_raw
    if isinstance(therapy_plan, dict) and 'confidence_score' in therapy_plan:
        therapy_plan.pop('confidence_score', None)
    
    return {
        'antibiotic_therapy_plan': therapy_plan,
        'pharmacist_analysis_on_resistant_gene': result_dict.get('pharmacist_analysis_on_resistant_gene', [])
    }

