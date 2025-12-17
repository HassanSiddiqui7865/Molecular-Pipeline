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


# Combined prompt for extracting both antibiotic therapy and resistance genes in one call
COMBINED_EXTRACTION_PROMPT = """Extract antibiotic stewardship information from the medical literature source provided below.

CONTEXT:
- Pathogen: {pathogen_name}
- Resistance Gene(s): {resistant_gene}
- Pathogen Count: {pathogen_count}
- ICD Code(s): {severity_codes}
- Patient Age: {age}

SOURCE CONTENT:
{content}

---

INSTRUCTIONS:

Extract ONLY antibiotics explicitly mentioned as effective against {pathogen_name} with {resistant_gene} resistance. If multiple resistance genes are specified, extract antibiotics effective against any of these resistance mechanisms.

Consider the patient's ICD codes ({severity_codes}) when extracting antibiotics - prioritize antibiotics appropriate for these conditions and severity levels. 

IMPORTANT: For combination therapies (e.g., "Daptomycin plus Ceftaroline", "Ampicillin plus Ceftriaxone"):
- ONLY extract if dose_duration is explicitly mentioned in the source
- If a combination therapy is mentioned but dose_duration is NOT specified, do NOT extract it
- Single antibiotics should always be extracted regardless of dose_duration availability

CATEGORIZATION:
- first_choice: Place here if source explicitly states 'first-line', 'preferred', 'recommended', 'guideline recommends', 'primary', or lists it as the primary/preferred option
- second_choice: Place here if source states 'alternative', 'second-line', 'if first-line unavailable', 'backup', or lists as secondary/backup option
- alternative_antibiotic: Place here if source mentions as 'other option', 'salvage therapy', 'last resort', 'consider if', or lists without clear preference/priority
- not_known: Place here ONLY when you are NOT confident about the category. Use this when the source mentions the antibiotic but does NOT clearly indicate its category (no explicit 'first-line', 'second-line', 'alternative', etc.). If you are uncertain or the category is ambiguous, use 'not_known'.

FIELD REQUIREMENTS FOR EACH ANTIBIOTIC:
- medical_name: GENERAL/NORMALIZED antibiotic name (NOT exact text from source). Extract the active ingredient name in normalized format:
  * Use title case: First letter capital, rest lowercase (e.g., "Linezolid", "Vancomycin", "Daptomycin")
  * Remove dosage information (e.g., "Linezolid 600mg" → "Linezolid", "Vancomycin 1g" → "Vancomycin")
  * Remove brand names in parentheses (e.g., "Linezolid (Zyvox)" → "Linezolid", "Vancomycin (Vancocin)" → "Vancomycin")
  * Remove route information from name (e.g., "Vancomycin IV" → "Vancomycin", "Linezolid PO" → "Linezolid")
  * Use generic name, not brand name (e.g., "Zyvox" → "Linezolid", "Vancocin" → "Vancomycin")
  * For combination therapies: Use format 'Drug1 plus Drug2' with normalized names (e.g., 'Daptomycin plus Ceftaroline')
  * Handle salt forms: Use base drug name (e.g., "Nitrofurantoin dihydrate" → "Nitrofurantoin", "Ampicillin sodium" → "Ampicillin")
  * Handle hyphens: Keep hyphenated names as-is (e.g., "Quinupristin-dalfopristin")
  
  PATTERN: Extract the core active ingredient name, normalized to title case, without dosage, brand names, routes, or salt forms.
  
  Examples:
  - Source says "linezolid 600mg twice daily" → medical_name: "Linezolid"
  - Source says "Zyvox (linezolid)" → medical_name: "Linezolid"
  - Source says "vancomycin IV 1g q12h" → medical_name: "Vancomycin"
  - Source says "daptomycin 10-12 mg/kg" → medical_name: "Daptomycin"
  - Source says "Nitrofurantoin dihydrate 100mg" → medical_name: "Nitrofurantoin"
  - Source says "quinupristin-dalfopristin" → medical_name: "Quinupristin-dalfopristin"
  
  REMEMBER: Only extract combinations if dose_duration is specified.
- coverage_for: Specific indication/condition it treats (e.g., 'VRE bacteremia', 'uncomplicated cystitis'). Be specific and concise. This is the medical indication, NOT the category label (NOT 'first-line/preferred' or 'alternative/second-line')
- route_of_administration: Route in standardized format: 'IV', 'PO', 'IM', 'IV/PO', 'IV or PO', 'Oral'. Extract from dose_duration if route is mentioned there but not separately. If not mentioned, use null (NOT 'Not specified')
- dose_duration: Dosing information in format: 'dose,route,frequency,duration'. Examples: '1000 mg,IV,q8h,10-14 days' or '15 mg/kg,IV,once daily,7 days' or '500 mg,PO,q12h,null'. For weight-based: '15 mg/kg,IV,once daily,7 days'. For fixed dose: '1000 mg,IV,q8h,10 days'. For combination therapies: '1000 mg (drug1),IV,q8h,10 days plus 600 mg (drug2),IV,q12h,10 days'. Use 'null' for missing components. Remove verbose phrases like 'during days +97-139', 'from first negative culture', 'high-dose', study details, averages, or case-specific information. CRITICAL: For combination therapies, if dose_duration is not mentioned, do NOT extract that combination at all. For single antibiotics, if not mentioned, use null (NOT 'Not specified' or 'Not mentioned')
- renal_adjustment: Renal adjustment information in format: 'Adjust dose in CrCl < X mL/min' or 'Dose adjust for renal dysfunction' or 'Avoid if CrCl < X mL/min'. Consider patient age if provided. If not mentioned, use null (NOT 'Not specified')
- general_considerations: Clinical notes, warnings, monitoring requirements, contraindications. Be comprehensive but concise. If nothing mentioned, use null (NOT 'Not specified')

RESISTANCE GENE EXTRACTION:
Extract information about the resistance mechanism(s) ({resistant_gene}). For each resistance gene mentioned in the source:
- detected_resistant_gene_name: Standard name of the detected resistance gene (must match one of: {resistant_gene})
- potential_medication_classes_affected: Medication classes affected by this resistance gene
- general_considerations: Resistance mechanism, mechanism of action, clinical impact, and treatment implications. If nothing mentioned, use null

If multiple resistance genes are specified in the context, extract information for each gene that is mentioned in the source.

DO NOT extract:
- Combination therapies where dose_duration is NOT specified (e.g., if "Daptomycin plus Ceftaroline" is mentioned but no dosing information is provided, do NOT extract it)
- Antibiotics mentioned as ineffective, resistant, or not recommended
- General drug classes without specific drug names (unless source explicitly recommends the class)
- Experimental or investigational drugs unless source explicitly recommends them
- Antibiotics mentioned only in context of resistance or failure"""


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

