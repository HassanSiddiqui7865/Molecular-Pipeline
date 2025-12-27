"""
Prompt templates and helper functions for LLM interactions.
"""
from typing import List, Optional, Dict


EXTRACTION_PROMPT_TEMPLATE = """Extract antibiotic therapy for {pathogen_display} with {resistant_gene} resistance.

PATIENT CONTEXT:
- Pathogen: {pathogen_display}
- Resistance: {resistant_gene}
- ICD Codes: {severity_codes}
- Age: {age} | Sample: {sample} | Systemic: {systemic}

SOURCE CONTENT:
The source content below is extracted from medical literature, guidelines, and research articles. It may contain:
- Plain text with markdown formatting (headers, lists, tables)
- Structured tables with pipe delimiters (|)
- Clinical guidelines, dosing recommendations, and treatment protocols
- References to studies, guidelines, and medical authorities
- Abbreviations and medical terminology

Extract information from this content:

{content}

EXTRACTION RULES:

1. ANTIBIOTICS:
   - Extract only antibiotics effective against {pathogen_display} with {resistant_gene}
   - Match clinical severity to ICD codes: {severity_codes}
   - Consolidate duplicates unless clinically distinct

2. COMBINATION DRUGS:
   - Detect: "Drug1 and Drug2", "Drug1/Drug2", "Drug1-Drug2", hyphenated names
   - Normalize: "Drug1 plus Drug2" (lowercase "plus", Title Case drugs)
   - Examples: "Quinupristin-dalfopristin" → "Quinupristin plus Dalfopristin"
   - Set is_combined=True for ANY combination

3. CATEGORIES:
   - first_choice: "first-line", "preferred", "recommended", "primary", or when listed first/primarily
   - second_choice: "alternative", "second-line", or when mentioned as backup option
   - alternative_antibiotic: "salvage", "last resort", or when mentioned as other option
   - not_known: Only use when category is truly unclear or cannot be determined
   
   Assign categories based on explicit statements OR contextual clues (order, emphasis, coverage description).
   For each categorized antibiotic, extract a complete, meaningful sentence from the source that explains 
   why it belongs in that category. The citation must provide sufficient context to justify the categorization.

4. REQUIRED FIELDS - STRICT FORMATTING (CRITICAL FOR CONSISTENCY):

   - medical_name: Title Case ONLY. No dosage, brand, route, or salts.
     Example: "Vancomycin" (NOT "vancomycin 1g IV" or "Vancomycin HCl")
   
   - coverage_for: Single primary indication ONLY. Format: "[Pathogen] [condition]"
     Examples: "MRSA bacteremia", "VRE bacteremia", "Staphylococcus aureus bacteremia"
     If multiple pathogens mentioned, use the PRIMARY one for this extraction context.
     DO NOT combine multiple pathogens unless source explicitly states combination therapy.
     DO NOT include ICD codes or ICD code names.
   
   - route_of_administration: EXACT format: "IV", "PO", "IM", or "IV/PO" (null if not stated)
     DO NOT use: "intravenous", "oral", "IV or PO", "intramuscular"
   
   - dose_duration: STANDARDIZED format. Follow these patterns EXACTLY:
     * Single dose: "[dose] [route] [frequency] for [duration]"
       Example: "15-20 mg/kg IV q12h for 14 days"
     * With loading: "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]"
       Example: "Loading: 1g IV, then 500 mg IV q12h for 7-14 days"
     * CRITICAL: DO NOT include renal adjustments in dose_duration
     * CRITICAL: If range given (e.g., "15-20 mg/kg"), KEEP the range - do not simplify
     * CRITICAL: ALWAYS include duration if mentioned (e.g., "for 14 days", "for 7-14 days")
     * Use abbreviations: "q12h", "q24h", "q8h" (NOT "every 12 hours", "twice daily", "BID")
     * DO NOT add "adjust for CrCl" or similar - that belongs in renal_adjustment
   
   - renal_adjustment: EXACT format: "Adjust dose for CrCl < [X] mL/min"
     Examples: "Adjust dose for CrCl < 30 mL/min", "Adjust dose for CrCl < 50 mL/min"
     DO NOT use: "Adjust dose if CrCl", "Adjust dose in CrCl", "Dose adjust for", "Adjust for renal dysfunction"
     If multiple thresholds, use most restrictive: "Adjust dose for CrCl < 30 mL/min"
     (null if not stated)
   
   - general_considerations: Brief clinical notes. Use semicolons to separate points.
     Examples: "Monitor trough levels; watch for nephrotoxicity"
     DO NOT include dosing information here - that belongs in dose_duration
     (null if none)
   
   - categorization_citation: Extract a complete, meaningful sentence or phrase from the source that 
     explains WHY this antibiotic is categorized as first_choice, second_choice, or alternative_antibiotic. 
     The citation must provide context that justifies the category (e.g., "Vancomycin is the first-line 
     treatment for MRSA bacteremia" or "Daptomycin is recommended as an alternative when vancomycin cannot 
     be used"). Include enough context to understand the reasoning. Do not use just the drug name or 
     incomplete phrases. If category is "not_known", use null.

5. RESISTANCE GENES (for each from {resistant_gene}):
   - detected_resistant_gene_name: Gene name (e.g., "mecA")
   - potential_medication_class_affected: Affected antibiotic classes
   - general_considerations: Resistance mechanism (null if none)

6. CONSISTENCY RULES (CRITICAL):
   - For the SAME antibiotic, extract the MOST COMPLETE and STANDARD information
   - If multiple dosages mentioned, use the STANDARD/MOST COMMON clinical dose
   - If range given (e.g., "15-20 mg/kg"), ALWAYS keep the range - do not simplify to single value
   - If duration mentioned, ALWAYS include it in dose_duration field
   - coverage_for should reflect PRIMARY indication for this specific pathogen/resistance
   - renal_adjustment MUST use format: "Adjust dose for CrCl < X mL/min" (not "if" or "in")
   - dose_duration and renal_adjustment are SEPARATE fields - do not mix them

7. VALIDATION:
   - Each antibiotic appears in ONE category only
   - medical_name: ALWAYS normalize combinations (no hyphens/slashes)
   - is_combined: True if original had hyphen/slash/"and"/"with"
   - NEVER extract resistance genes (mecA, vanA) as antibiotics
   - Extract verbatim when possible, but normalize format to match standards above

8. EXCLUDE:
   - Resistance genes listed as antibiotics
   - Ineffective/resistant antibiotics
   - Drug classes without specific names
   - Experimental drugs (unless recommended)"""

