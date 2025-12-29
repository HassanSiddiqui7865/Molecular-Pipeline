"""
Prompt templates and helper functions for LLM interactions.
"""
from typing import List, Optional, Dict


EXTRACTION_PROMPT_TEMPLATE = """Extract antibiotic therapy information from medical content.

PATIENT CONTEXT:
- Pathogen: {pathogen_display}
- Resistance Genes: {resistant_gene}
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
   - Examples: "Trimethoprim-sulfamethoxazole" → "Trimethoprim plus Sulfamethoxazole"
   - Examples: "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole"

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
   
   - route_of_administration: Extract route if mentioned anywhere in source.
     * Format: "IV", "PO", "IM", or "IV/PO"
     * Extract from dosing information if route is mentioned there
     * If not stated anywhere → null
   
   - dose_duration: Complete dosing information with dose, frequency, and duration.
     * Format: "[dose] [route] [frequency] for [duration]"
       Example: "15-20 mg/kg IV q12h for 14 days"
     * With loading: "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]"
       Example: "Loading: 1g IV, then 500 mg IV q12h for 7-14 days"
     * Must include: dose amount, frequency, and duration
     * If any part is missing (dose, frequency, or duration) → null
     * Convert frequency abbreviations: "BID" → "q12h", "TID" → "q8h", "daily" → "q24h"
     * Keep ranges as written (e.g., "15-20 mg/kg", "7-14 days")
     * DO NOT include renal adjustments - that belongs in renal_adjustment
   
   - renal_adjustment: 
     * If source explicitly states "No Renal Adjustment needed" or similar → "No Renal Adjustment"
     * If source states specific CrCl threshold → EXACT format: "Adjust dose for CrCl < [X] mL/min"
       Examples: "Adjust dose for CrCl < 30 mL/min", "Adjust dose for CrCl < 50 mL/min"
       DO NOT use: "Adjust dose if CrCl", "Adjust dose in CrCl", "Dose adjust for", "Adjust for renal dysfunction"
       If multiple thresholds, use most restrictive: "Adjust dose for CrCl < 30 mL/min"
     * If source does not state anything about renal adjustment → null
     * DO NOT duplicate information that belongs in general_considerations
   
   - general_considerations: Brief clinical notes only. Use semicolons to separate points.
     * Include: monitoring requirements, warnings, toxicity, interactions, contraindications
     * Examples: "Monitor trough levels; watch for nephrotoxicity"
     * DO NOT include: dosing information, drug class descriptions, efficacy statements
     * DO NOT duplicate: renal adjustment information, dosing information
     * Keep concise - maximum 2-3 key points
     * (null if none)

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
   - medical_name: ALWAYS normalize combinations to "plus" format (no hyphens/slashes)
   - NEVER extract resistance genes (mecA, vanA) as antibiotics
   - Extract verbatim when possible, but normalize format to match standards above
   - If dose_duration is incomplete (missing dose, frequency, or duration) → set to null

8. EXCLUDE:
   - Resistance genes listed as antibiotics
   - Ineffective/resistant antibiotics
   - Drug classes without specific names
   - Experimental drugs (unless recommended)"""



ANTIBIOTIC_FILTERING_PROMPT_TEMPLATE = """You are a clinical pharmacology expert. Evaluate each antibiotic and determine whether it should be KEPT or FILTERED OUT based on clinical appropriateness for the specific patient context.

PATIENT CONTEXT:
- Pathogen(s): {pathogen_display}
- Resistance Gene(s): {resistant_gene}
- ICD Severity Codes: {severity_codes}
- Age: {age}
- Sample Type: {sample}
- Systemic Infection: {systemic}

CANDIDATE ANTIBIOTICS:
{antibiotic_list}

TASK:
For each antibiotic, determine: should_keep (true/false) and filtering_reason (null if keeping, explanation if filtering out).

FILTERING CRITERIA (APPLY IN ORDER):

1. MICROBIOLOGICAL EFFICACY (Primary Filter - Most Important):
   
   FILTER OUT ONLY if:
   - Antibiotic has NO activity against ANY of the listed pathogens, OR
   - ALL listed pathogens are inherently resistant to this antibiotic class, OR
   - The resistance gene(s) confer complete resistance to this antibiotic against ALL pathogens
   
   ALWAYS KEEP if:
   - Effective against AT LEAST ONE pathogen in {pathogen_display}
   - Retains activity despite the resistance mechanism for ANY pathogen
   - Has documented efficacy in similar resistance patterns for ANY pathogen
   
   CRITICAL RULE: For multi-pathogen infections, an antibiotic that works against even ONE pathogen MUST be kept, even if ineffective against the others.

2. PATIENT SAFETY:
   
   FILTER OUT if:
   - Absolute contraindication for patient age: {age}
   - FDA black box warning directly applicable to this clinical scenario
   - Life-threatening adverse effect risk that clearly outweighs benefit
   - Documented severe drug interactions that cannot be managed
   
   KEEP if:
   - Acceptable risk-benefit ratio for severity: {severity_codes}
   - Relative contraindications that can be monitored/managed
   - Standard precautions sufficient for safe use
   - Adverse effects are manageable with appropriate monitoring

3. CLINICAL APPROPRIATENESS:
   
   FILTER OUT if:
   - Inadequate tissue/site penetration for sample type: {sample} (e.g., poor CNS penetration for meningitis)
   - Severity grossly inappropriate: antibiotic clearly insufficient for {severity_codes} (e.g., topical antibiotic for sepsis)
   - Route of administration impossible for clinical condition (e.g., oral-only drug in unconscious patient with septic shock)
   - Documented poor outcomes in bloodstream infections despite in vitro activity (e.g., tigecycline for bacteremia)
   
   KEEP if:
   - Adequate penetration to infection site
   - Appropriate potency for infection severity
   - Suitable route available for administration
   - Clinical evidence supports use in similar infections

4. GUIDELINE ADHERENCE:
   
   FILTER OUT if:
   - Explicitly contraindicated in current IDSA/ESCMID guidelines for this specific indication
   - Withdrawn from market or under regulatory restriction
   - Strong guideline recommendation AGAINST use for this indication
   
   KEEP if:
   - Guideline-recommended or listed as acceptable alternative
   - Not specifically discouraged in guidelines
   - Clinical evidence supports use even if not guideline-preferred
   - Guideline-compatible based on mechanism of action

5. APPROVAL & AVAILABILITY STATUS:
   
   FILTER OUT if:
   - Investigational only (no FDA/EMA approval for any indication)
   - Not commercially available
   - Experimental without emergency authorization
   
   KEEP if:
   - FDA/EMA approved (even if for different indication)
   - Commercially available
   - Off-label but evidence-supported use

DECISION FRAMEWORK - APPLY IN THIS EXACT ORDER:
1. Check if effective against ≥1 pathogen → If YES, continue evaluation; if NO, filter as "Ineffective"
2. Check resistance genes → Filter ONLY if resistance applies to ALL pathogens
3. Check absolute safety contraindications → Filter only if life-threatening risk
4. Check clinical appropriateness → Filter only if clearly incompatible
5. Check guidelines → Filter only if explicitly contraindicated
6. DEFAULT: KEEP (when in doubt, include the antibiotic)

MULTI-PATHOGEN RULE (CRITICAL):
When multiple pathogens are present (e.g., "S. aureus and E. faecalis"):
- An antibiotic effective against S. aureus but NOT E. faecalis → KEEP
- An antibiotic effective against E. faecalis but NOT S. aureus → KEEP
- An antibiotic effective against NEITHER pathogen → FILTER OUT
- Reasoning must specify: "effective against [pathogen name] but not [other pathogen name]" when keeping partial-coverage antibiotics

RESISTANCE GENE CONSIDERATION:
- Evaluate if resistance gene affects the antibiotic for EACH pathogen separately
- Example: mecA affects beta-lactams in S. aureus but not in E. faecalis
- Example: dfrA affects trimethoprim in BOTH S. aureus and E. faecalis → filter TMP-SMX as ineffective against both

FILTERING REASON FORMAT (use exactly these formats):
- "Ineffective: No activity against any listed pathogen"
- "Ineffective: Resistance gene [gene_name] confers resistance against all pathogens"
- "Ineffective: Inherently inactive against all pathogens ([list pathogens])"
- "Safety: Absolute contraindication - [specific reason]"
- "Clinical: Inadequate penetration/inappropriate for {sample}/{severity_codes}"
- "Clinical: Poor outcomes documented in bloodstream infections"
- "Guideline: Explicitly not recommended for [indication] per [guideline name]"
- "Approval: Not FDA/EMA approved or commercially available"

DO NOT filter based on:
- Cost or insurance coverage
- Requiring therapeutic drug monitoring (common practice)
- Minor or manageable side effects
- Second-line or third-line status (still valid options)
- Lack of head-to-head comparison data
- Need for dose adjustment (standard practice)

CRITICAL REMINDERS:
1. Multi-pathogen rule: Effective against ≥1 pathogen = KEEP
2. Conservative filtering: Only remove with strong clinical evidence
3. Severe infections warrant broader inclusion
4. Resistance genes may affect pathogens differently
5. When uncertain, KEEP the antibiotic

Begin evaluation:"""


ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE = """Unify antibiotic information from multiple sources into a single optimized entry.

CRITICAL RULE - DO NOT INVENT INFORMATION:
- Use ONLY information explicitly present in the ENTRIES FROM DIFFERENT SOURCES provided below
- DO NOT add, infer, or invent any information that is not present in the source entries
- DO NOT use your general medical knowledge to fill in missing details
- DO NOT combine partial information from different sources to create complete information unless explicitly stated together in a single source
- If information is missing in all sources, keep it as null or incomplete
- Only synthesize and combine what is actually provided in the source entries

ANTIBIOTIC: {antibiotic_name}

ENTRIES FROM DIFFERENT SOURCES:
{entries_list}

TASK:
Synthesize the most accurate and complete information from all sources into ONE unified entry.
Remember: Only use information that exists in the source entries - do not invent missing details.

UNIFICATION RULES (APPLY IN ORDER):

1. medical_name:
   - Use ONLY names present in the source entries
   - Use the most standard/complete form across all source entries
   - If source entries differ, prefer the most clinically standard name from entries
   - Maintain Title Case format (first letter of each word capitalized)
   - Keep combination format: "Drug1 plus Drug2" (if applicable)
   - Do NOT change the name if all source entries agree
   - Do NOT invent or standardize names not in source entries

2. coverage_for:
   - Use ONLY indications present in the source entries
   - Select the most specific and clinically relevant indication from source entries
   - Priority order: most specific pathogen/condition > general coverage (from entries)
   - If source entries conflict and no patient context is provided, prefer the most specific indication from the source entries
   - Format: "[Pathogen] [condition]" (single primary indication)
   - Do NOT combine multiple indications
   - Do NOT invent or infer indications not in source entries

3. route_of_administration:
   - Use ONLY routes present in the source entries
   - Combine all unique routes from source entries
   - Format: "IV/PO" if both present in entries, "IV", "PO", or "IM" if single route
   - Priority if conflict: IV > IV/PO > PO > IM
   - Use most common route from source entries if ambiguous
   - Must be one of: "IV", "PO", "IM", or "IV/PO"
   - Do NOT infer route from dosing if not explicitly stated in source entries

4. dose_duration:
   - CRITICAL: Use ONLY information explicitly stated in the source entries
   - DO NOT invent, infer, or complete missing dosing details
   - DO NOT use general medical knowledge
   - Select the most comprehensive dosing regimen AVAILABLE IN A SINGLE SOURCE ENTRY
   - Priority order (highest to lowest):
     a) Complete dose with loading + maintenance + duration (from one source)
     b) Standard dose with duration (dose + frequency + duration from one source)
     c) Standard dose without duration (dose + frequency from one source)
     d) Duration only (e.g., "5 days", "7–14 days") if only duration is mentioned
     e) Incomplete or vague dosing
   - Preserve dose ranges exactly as written (e.g., "15–20 mg/kg")
   - Always include duration if it is explicitly stated in the SAME source as dose/frequency
   - If multiple complete regimens exist, prefer the most specific
   - DO NOT combine dosing components from different source entries
   - If all source entries have incomplete dosing, preserve the most informative incomplete form
   - Format ONLY if complete:
     "[dose] [route] [frequency] for [duration]"
     OR
     "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]"

5. renal_adjustment:
   - Use ONLY renal adjustment information explicitly present in source entries
   - STRICT CRITERIA:
     * Use ONLY if specific CrCl threshold is stated
     * If explicit "no adjustment needed" is stated → "No Renal Adjustment"
     * If only vague warnings ("use caution", "avoid in severe renal") → null
   - Handling multiple sources:
     * If any source has a specific threshold → use the most restrictive (lowest CrCl)
     * If mix of specific and vague → use the specific threshold
     * If all sources say no adjustment → "No Renal Adjustment"
     * If all sources are vague or silent → null
   - Edge cases (ONLY if explicitly stated):
     * "Contraindicated in CrCl < X" → "Adjust dose for CrCl < X mL/min" and add contraindication to general_considerations
     * "Reduce dose by 50% if CrCl < X" → "Adjust dose for CrCl < X mL/min"
     * "Avoid in renal impairment" → null (move to general_considerations)
   - Do NOT invent renal thresholds

6. general_considerations:
   - Use ONLY clinical notes explicitly present in source entries
   - Combine ALL distinct notes from all source entries
   - Separate points with semicolons
   - Remove exact duplicates but keep variations
   - Include only:
     * Monitoring requirements
     * Vague renal/hepatic warnings
     * Drug interactions
     * Side effects/toxicity
     * Contraindications
     * Special population notes
   - Do NOT repeat renal_adjustment information
   - Do NOT add external medical knowledge
   - Order: monitoring > warnings > interactions > side effects
   - Set to null if no notes exist

7. is_combined:
   - True if ANY source marks is_combined=True
   - True if medical_name contains "plus"
   - False only if ALL sources indicate not combined

8. is_complete:
   - TRUE only if ALL required fields are present AND dose_duration is COMPLETE
   - Required fields:
     * medical_name (must be present)
     * coverage_for (must be present)
     * route_of_administration (must be present)
     * dose_duration (must include dose amount, frequency, and duration)
     * renal_adjustment (may be specific, "No Renal Adjustment", or null)
     * general_considerations (may be null)
   - FALSE if any required field is missing OR dose_duration is incomplete

CONFLICT RESOLUTION:
- dose_duration: guideline-based > institutional protocol > textbook > case report
- route_of_administration: combine unless contradictory
- renal_adjustment: most restrictive specific threshold
- coverage_for: most specific indication
- general_considerations: combine all unique points

VALIDATION CHECKLIST (must all pass):
- medical_name uses Title Case and "plus" for combinations
- route_of_administration is one of: IV, PO, IM, IV/PO
- dose_duration is complete if marked complete
- renal_adjustment is ONLY: "Adjust dose for CrCl < X mL/min", "No Renal Adjustment", or null
- general_considerations does not duplicate renal_adjustment
- is_combined correctly reflects combination therapy
- is_complete is TRUE only when therapy information is fully usable
- No information is invented or inferred

EDGE CASE HANDLING:
- Preserve null values if all sources are null
- Do NOT combine partial dosing across sources
- Document clinically significant contradictions in general_considerations
- Deduplicate identical source information
- Note population-specific differences if explicitly stated

OUTPUT:
Return ONE unified entry containing the best synthesized information from all sources.
Ensure all field formats match the requirements above exactly.
"""

