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

from utils import format_resistance_genes, format_icd_codes

logger = logging.getLogger(__name__)


COMBINED_EXTRACTION_PROMPT = """Extract antibiotic information for {pathogen_name} with {resistant_gene} resistance.

PATIENT CONTEXT:
Pathogen: {pathogen_name} ({pathogen_count})
Resistance: {resistant_gene}
ICD Codes: {severity_codes}
Age: {age}

SOURCE:
{content}

---

RULES:

1. SELECTION:
   - Extract antibiotics explicitly effective against {pathogen_name} with {resistant_gene}
   - Match severity to ICD codes: {severity_codes}
   - Consolidate duplicates unless clinically distinct (different doses/indications)

2. COMBINATIONS:
   - Identify combinations from: "Drug1 and Drug2", "Drug1 with Drug2", "Drug1/Drug2", "Drug1-Drug2", hyphenated names (e.g., "Quinupristin-dalfopristin")
   - Format: ALWAYS normalize to "Drug1 plus Drug2" format (lowercase "plus", title case drug names)
   - Examples: "Quinupristin-dalfopristin" → "Quinupristin plus Dalfopristin" | "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole" | "Ampicillin and Gentamicin" → "Ampicillin plus Gentamicin"
   - Set is_combined = True for ANY combination (hyphenated, slash, "plus", or explicit "and/with")

3. CATEGORIES (only if explicitly stated):
   - first_choice: "first-line", "preferred", "recommended", "primary"
   - second_choice: "alternative", "second-line", "backup"
   - alternative_antibiotic: "salvage", "last resort"
   - not_known: Mentioned as effective but category not stated

4. FIELDS:

   medical_name: Title case, no dosage/brand/route/salts. ALWAYS normalize combinations to "Drug1 plus Drug2" format.
   Examples: "vancomycin 1g IV" → "Vancomycin" | "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole" | "Quinupristin-dalfopristin" → "Quinupristin plus Dalfopristin" | "Ampicillin and Gentamicin" → "Ampicillin plus Gentamicin"

   is_combined: True if therapies are combined (name contains "plus" after normalization), False otherwise

   coverage_for: Specific indication (e.g., "MRSA bacteremia")

   route_of_administration: "IV", "PO", "IM", "IV/PO", "Oral" (null if not mentioned)

   dose_duration: Always comma-separated "dose,route,frequency,duration"
   - Single: "600 mg,PO,q12h,7 days"
   - Combinations: Include all mentioned dosages. Format: "Drug1:dose1,route1,freq1,dur1|Drug2:dose2,route2,freq2,dur2" (pipe-separated with drug names)
   - If both combined and individual mentioned: "combined:dose,route,freq,dur|Drug1:dose1,route1,freq1,dur1|Drug2:dose2,route2,freq2,dur2"
   - Loading doses: Use maintenance only (ignore loading, use regular dose)
   - Components: dose (amount only), route (IV/PO/IM), frequency (q8h/q12h/BID/TID), duration (days/weeks)
   - Missing: Use "null" for any missing component
   - Examples: "600 mg,PO,q12h,7 days" | "Ampicillin:2g,IV,q4h,14 days|Gentamicin:1mg/kg,IV,q8h,14 days" | "null,null,null,null"

   renal_adjustment: "Adjust dose for CrCl < X mL/min" or similar (null if not mentioned)

   general_considerations: Synthesize clinical notes concisely (null if none)

5. RESISTANCE GENES (for each from {resistant_gene}):
   - detected_resistant_gene_name: Gene name (e.g., "mecA")
   - potential_medication_class_affected: Affected classes
   - general_considerations: Mechanism and impact (null if none)

VALIDATION:
- Each antibiotic in ONE category only
- medical_name: ALWAYS normalize combinations to "Drug1 plus Drug2" format (never keep hyphen/slash format)
- is_combined = True if name contains "plus" (after normalization) or original had: hyphen (-), slash (/), "plus", or explicit "and/with"
- NEVER extract genes (mecA, vanA) as antibiotics
- dose_duration: Always comma-separated, include all mentioned dosages for combinations
- Loading doses: Use maintenance only
- No verbatim copying - synthesize intelligently

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
        
        pathogen_name = input_params.get('pathogen_name', '')
        resistant_gene_raw = input_params.get('resistant_gene', '')
        # Format resistance genes (handle comma-separated)
        resistant_gene = format_resistance_genes(resistant_gene_raw)
        pathogen_count = input_params.get('pathogen_count', '')
        severity_codes_raw = input_params.get('severity_codes', '')
        # Format ICD codes (handle comma-separated)
        severity_codes = format_icd_codes(severity_codes_raw)
        age = input_params.get('age')
        
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
            extraction_result = _extract_combined(
                source_content, pathogen_name, resistant_gene,
                pathogen_count, severity_codes, age, result.url, result.title, llm
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
    source_url: str,
    source_title: str,
    llm: BaseChatModel
) -> Dict[str, Any]:
    """Extract both antibiotic therapy plan and resistance genes using LangChain.
    Returns dictionary with both results.
    """
    # Create prompt
    prompt = COMBINED_EXTRACTION_PROMPT.format(
        pathogen_name=pathogen_name,
        resistant_gene=resistant_gene,
        pathogen_count=pathogen_count,
        severity_codes=severity_codes,
        age=f"{age} years" if age else 'Not specified',
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

