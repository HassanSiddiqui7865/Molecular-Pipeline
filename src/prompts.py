"""
Prompt templates and helper functions for LLM interactions.
"""
from typing import List, Optional, Dict


EXTRACTION_PROMPT_TEMPLATE = """Extract antibiotic therapy recommendations from medical content.

CONTEXT: Pathogen: {pathogen_display}{resistance_context} | Severity: {severity_codes} | Age: {age}, Sample: {sample}, Systemic: {systemic}

SOURCE: {content}

CRITICAL: Extract STRICTLY based on the input context above. Do NOT extract antibiotics or information for other pathogens, conditions, or patient scenarios.

TASK: Extract ONLY antibiotics effective against {pathogen_display}{resistance_task} matching the provided context. Extract ALL available information - do not leave fields null if data exists.

FIELDS:

medical_name: Title Case drug name only. Combinations: convert "Drug1/Drug2", "Drug1-Drug2", "Drug1 and Drug2" → "Drug1 plus Drug2". Examples: "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole", "Imipenem/cilastatin" → "Imipenem plus Cilastatin".

category: "first_choice" (first-line/preferred/primary), "second_choice" (alternatives/backup), "alternative_antibiotic" (salvage/last resort), or "not_known" (cannot determine). Use contextual clues (order, emphasis) if not explicit.

coverage_for: Format "[Pathogen] [condition]". Use ONLY pathogen matching {pathogen_display} from the input context. Condition based on sample type: Blood/systemic → "bacteremia" or "sepsis" (prefer bacteremia), Urine → "UTI" or "urinary tract infection", Sputum/Respiratory → "pneumonia" or "respiratory infection", CSF → "meningitis", Wound → "wound infection", Other → use appropriate condition from source. Example: "Staphylococcus aureus bacteremia" or "Escherichia coli UTI".

route_of_administration: Extract from explicit mentions ("IV", "intravenous", "PO", "oral", "IM") or infer from dosing. Values: "IV", "PO", "IM", or "IV/PO". Use null only if no route info exists.

dose_duration: Format "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". CRITICAL: Convert frequencies BEFORE formatting: BID/twice daily → q12h, TID/three times daily → q8h, QD/daily/once daily → q24h. Preserve ranges ("15-20 mg/kg", "7-14 days"). Extract partial info if available: "[dose] [route] [frequency]" if no duration, "[duration]" if only duration. Look for duration phrases: "for X days/weeks", "duration", "treatment/course length". Use null only if NO dosing info exists.

renal_adjustment: "No Renal Adjustment" if explicitly stated. If CrCl threshold mentioned, use "Adjust dose for CrCl < X mL/min" (use most restrictive if multiple). Look for: "CrCl < X", "creatinine clearance < X", "if/when CrCl < X". Use null if not mentioned. Do not duplicate general_considerations.

general_considerations: Extract monitoring ("monitor", "watch for"), warnings, toxicity, interactions, contraindications. Separate with semicolons. Exclude dosing, drug class descriptions, efficacy. Use null only if no safety/monitoring info exists.

{resistance_genes_section}CRITICAL RULES:
1. Formatting: BID→q12h, TID→q8h, QD/daily→q24h. Slashes→"plus" for combinations.
2. Extraction: Extract aggressively - use all available info. For duplicates, use most complete. Keep ranges intact. Include duration when mentioned anywhere.
{resistance_filtering_rule}
4. Validation: Each antibiotic in one category. Normalize combinations. Never extract resistance genes as antibiotics.

DO NOT EXTRACT: Resistance genes as antibiotics. Ineffective antibiotics (filtered by resistance). Drug classes without specific names. Experimental drugs (unless recommended)."""

ANTIBIOTIC_FILTERING_PROMPT_TEMPLATE = """Evaluate each antibiotic: KEEP or FILTER OUT based on clinical appropriateness.

CONTEXT: Pathogen(s): {pathogen_display}{resistance_context} | Severity: {severity_codes} | Age: {age} | Sample: {sample} | Systemic: {systemic}

CANDIDATES: {antibiotic_list}

TASK: For each antibiotic, determine should_keep (true/false) and filtering_reason (null if keeping, explanation if filtering).

FILTERING CRITERIA (apply in order):

1. MICROBIOLOGICAL EFFICACY (Primary - Most Important):
   FILTER OUT if: No activity against ANY pathogen OR all pathogens inherently resistant OR resistance gene confers complete resistance against ALL pathogens.
   KEEP if: Effective against ≥1 pathogen OR retains activity despite resistance for ANY pathogen OR documented efficacy in similar resistance patterns.
   CRITICAL: Multi-pathogen rule - effective against ≥1 pathogen = KEEP (even if ineffective against others).

2. PATIENT SAFETY:
   FILTER OUT if: Absolute contraindication for age {age} OR FDA black box warning applicable OR life-threatening risk outweighs benefit OR severe unmanageable drug interactions.
   KEEP if: Acceptable risk-benefit for severity {severity_codes} OR relative contraindications manageable OR standard precautions sufficient OR manageable adverse effects.

3. CLINICAL APPROPRIATENESS:
   FILTER OUT if: Inadequate penetration for {sample} (e.g., poor CNS for meningitis) OR severity inappropriate (e.g., topical for sepsis) OR route impossible (e.g., PO-only in unconscious) OR documented poor outcomes (e.g., tigecycline for bacteremia).
   KEEP if: Adequate penetration OR appropriate potency OR suitable route available OR clinical evidence supports use.

4. GUIDELINE ADHERENCE:
   FILTER OUT if: Explicitly contraindicated (guidelines say "do not use" or "avoid") OR withdrawn/restricted OR strong recommendation AGAINST.
   KEEP if: Guideline-recommended OR listed as acceptable alternative/second-line/third-line OR not discouraged OR evidence-supported OR guideline-compatible mechanism. CRITICAL: "Not first-line" or "alternative option" does NOT mean contraindicated - these should be KEPT.

5. APPROVAL & AVAILABILITY:
   FILTER OUT if: Investigational only (no FDA/EMA approval) OR not commercially available OR experimental without authorization.
   KEEP if: FDA/EMA approved (even different indication) OR commercially available OR off-label but evidence-supported.

DECISION FRAMEWORK (exact order):
1. Effective against ≥1 pathogen? YES→continue, NO→filter "Ineffective"
{resistance_decision_step}3. Absolute safety contraindication? YES→filter, NO→continue
4. Clearly clinically inappropriate? YES→filter, NO→continue
5. Explicitly contraindicated in guidelines (says "do not use" or "avoid")? YES→filter, NO→continue. Note: "alternative" or "not first-line" = KEEP.
6. DEFAULT: KEEP (when uncertain, include)

MULTI-PATHOGEN RULE: Effective against S. aureus but NOT E. faecalis → KEEP. Effective against E. faecalis but NOT S. aureus → KEEP. Effective against NEITHER → FILTER OUT. Reasoning: "effective against [pathogen] but not [other]" for partial coverage.

{resistance_genes_evaluation}

FILTERING REASON FORMAT:
- "Ineffective: No activity against any listed pathogen"
- "Ineffective: Resistance gene [gene_name] confers resistance against all pathogens"
- "Ineffective: Inherently inactive against all pathogens ([list])"
- "Safety: Absolute contraindication - [reason]"
- "Clinical: Inadequate penetration/inappropriate for {sample}/{severity_codes}"
- "Guideline: Explicitly contraindicated (do not use) for [indication] per [guideline]"

DO NOT FILTER based on: Cost/insurance, TDM requirements, minor/manageable side effects, second/third-line status, lack of head-to-head data, dose adjustment needs.

CRITICAL: Effective against ≥1 pathogen = KEEP. Conservative filtering - only remove with strong evidence. "Alternative" or "second-line" options are VALID and should be KEPT. Only filter if guidelines explicitly say "do not use" or "avoid" or "contraindicated". When uncertain, KEEP."""


ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE = """Unify antibiotic information from multiple sources into ONE optimized entry.

CRITICAL: Use ONLY information explicitly present in source entries. DO NOT invent, infer, or use medical knowledge. DO NOT combine partial info across sources unless stated together in one source. If missing in all sources, keep null.

ANTIBIOTIC: {antibiotic_name}

SOURCES: {entries_list}

TASK: Synthesize most accurate/complete info from all sources into ONE unified entry.

UNIFICATION RULES:

1. medical_name: Use only names from entries. Most standard/complete form. If differ, prefer most clinically standard. Title Case. CRITICAL: For combinations, use lowercase "plus" (e.g., "Trimethoprim plus Sulfamethoxazole", NOT "Plus"). Keep "Drug1 plus Drug2" format. Don't change if all agree. Don't invent names.

2. coverage_for: Use only indications from entries. Most specific pathogen/condition. Priority: specific > general. Format "[Pathogen] [condition]" (single primary). Don't combine multiple. Don't invent.

3. route_of_administration: Use only routes from entries. Combine unique routes: "IV/PO" if both, else "IV"/"PO"/"IM". Priority if conflict: IV > IV/PO > PO > IM. Must be IV/PO/IM/IV/PO. Don't infer from dosing.

4. dose_duration: CRITICAL - Use ONLY from entries. Don't invent/infer/complete. Select most comprehensive regimen from SINGLE source. Priority: a) Loading+maintenance+duration, b) Dose+freq+duration, c) Dose+freq, d) Duration only, e) Incomplete. Preserve ranges ("15-20 mg/kg"). Include duration if in SAME source as dose/freq. If multiple complete, prefer most specific. Don't combine components across sources. For combination drugs, use total dose if shown as single value, or format as "[dose] [route] [frequency]" if components shown separately. Format if complete: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]".

5. renal_adjustment: Use only from entries. STRICT: Only if specific CrCl threshold stated. "No adjustment needed" → "No Renal Adjustment". Vague warnings → null. Multiple sources: use most restrictive threshold (lowest CrCl). Mix specific+vague → use specific. All say no adjustment → "No Renal Adjustment". All vague/silent → null. Edge cases: "Contraindicated CrCl < X" → "Adjust dose for CrCl < X mL/min" + add to general_considerations. "Reduce 50% if CrCl < X" → "Adjust dose for CrCl < X mL/min". "Avoid in renal" → null (move to general_considerations). Don't invent thresholds.

6. general_considerations: Use only notes from entries. Combine ALL distinct notes. Separate with semicolons. Remove exact duplicates and near-duplicates (same meaning, different wording). Keep concise - 2-3 key points maximum. Include: monitoring, vague renal/hepatic warnings, interactions, side effects/toxicity, contraindications, special populations. Exclude: renal_adjustment info, external knowledge, redundant statements. Order: monitoring > warnings > interactions > side effects. Null if no notes.

7. is_combined: True if ANY source has is_combined=True OR medical_name contains "plus" (case-insensitive). False only if ALL indicate not combined.

8. is_complete: TRUE only if ALL fields are present AND complete. ALL fields are REQUIRED: medical_name, coverage_for, route_of_administration, dose_duration (must be COMPLETE with dose+freq+duration), renal_adjustment, general_considerations. FALSE if ANY field is missing, null, or incomplete. CRITICAL: If dose_duration is incomplete (missing dose, frequency, or duration), set dose_duration to null (so enrichment can extract it) and set is_complete to FALSE.

CONFLICT RESOLUTION: dose_duration: guideline > protocol > textbook > case report. route: combine unless contradictory. renal_adjustment: most restrictive threshold. coverage_for: most specific. general_considerations: combine unique points, remove duplicates.

VALIDATION: Title Case + lowercase "plus" for combinations. Route: IV/PO/IM/IV/PO. dose_duration complete if marked. renal_adjustment: "Adjust dose for CrCl < X mL/min"/"No Renal Adjustment"/null. No duplication with general_considerations. is_combined reflects combinations. is_complete TRUE only when fully usable. No invented info.

EDGE CASES: Preserve null if all null. Don't combine partial dosing across sources. Document contradictions in general_considerations. Deduplicate identical/near-identical info. Note population differences if stated.

OUTPUT: ONE unified entry with best synthesized info. All formats match requirements."""

RESISTANCE_GENE_UNIFICATION_PROMPT_TEMPLATE = """Unify resistance gene information from multiple sources into unified entries.

CRITICAL: Use ONLY information explicitly present in source entries. DO NOT invent, infer, or use medical knowledge. DO NOT combine partial info across sources unless stated together in one source. If missing in all sources, keep null.

RESISTANCE GENES FROM SOURCES:
{genes_list}

TASK: For each unique resistance gene, synthesize most accurate/complete info from all sources into ONE unified entry per gene.

UNIFICATION RULES (per gene):

1. detected_resistant_gene_name: Use only names from entries. Most standard/complete form. If differ, prefer most clinically standard. Don't change if all agree. Don't invent names.

2. potential_medication_class_affected: Use only classes from entries. Combine all unique classes mentioned. Remove duplicates (e.g., "beta-lactams" and "beta-lactam antibiotics" → "beta-lactams"). Format: combine with commas. Examples: "beta-lactams, penicillins, cephalosporins, monobactams". Don't invent classes.

3. general_considerations: Use only notes from entries. Combine ALL distinct notes. Separate with semicolons. CRITICAL: Remove exact duplicates AND near-duplicates (same meaning, different wording). Keep concise - 2-3 key points maximum. Include: resistance mechanisms, how gene confers resistance, clinical implications, inhibition by beta-lactamase inhibitors. Order: mechanism > clinical implications > inhibitors. Remove redundant statements (e.g., "hydrolyzes X" and "confers resistance to X" if they mean the same). Null if no notes.

CONFLICT RESOLUTION: detected_resistant_gene_name: most standard form. potential_medication_class_affected: combine all unique classes, remove duplicates. general_considerations: combine unique points, remove duplicates and near-duplicates.

VALIDATION: detected_resistant_gene_name matches gene name. potential_medication_class_affected lists all affected classes without duplicates. general_considerations combines unique mechanisms/considerations without redundancy. No invented info.

OUTPUT: List of unified entries, one per unique resistance gene. All formats match requirements."""


ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE = """Validate if drugs.com page matches the antibiotic we're searching for.

SEARCHING FOR: {antibiotic_name}
PAGE TITLE: {page_title}

TASK: Determine if page title indicates the same drug (same active ingredient/medically equivalent).

Return is_match=True if same drug, is_match=False if different drug."""


DOSAGE_EXTRACTION_PROMPT_TEMPLATE = """Extract ONLY missing fields for {medical_name} from drugs.com content.

PATIENT: Age={patient_age} | ICD={icd_codes}{gene_context}

CRITICAL: Extract STRICTLY based on the input context above. Do NOT extract information for other conditions, ICD codes, ages, or scenarios.

MISSING FIELDS (extract ONLY these): {missing_fields}

EXISTING DATA (context only, preserve if not missing):
{existing_data}

{cross_chunk_context}PAGE CONTENT (chunk {chunk_num} of {total_chunks}):
{chunk_content}

FIELDS (extract ONLY if in missing_fields):

dose_duration: Format "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". Match to ICD: {icd_codes}{gene_matching} and Age: {patient_age}. Frequency MUST include "q" prefix (q8h, q12h, q24h). Include loading doses if present. Choose ONE most appropriate. Use null if no dosing info.

route_of_administration: Extract from explicit mentions or infer from dosing. Values: "IV", "PO", "IM", or "IV/PO". Use null if no route info.

coverage_for: Format "[Pathogen] [condition]" using clinical terminology (e.g., "MRSA bacteremia", "VRE bacteremia"). Match to patient's clinical condition (ICD: {icd_codes}{gene_context}). Use null if no info.

renal_adjustment: "No Renal Adjustment" if explicitly stated. If CrCl threshold mentioned, use "Adjust dose for CrCl < X mL/min" (most restrictive if multiple). Use null if not mentioned. Do not duplicate general_considerations.

general_considerations: Extract monitoring, warnings, toxicity, interactions, contraindications. Separate with semicolons. Exclude dosing, drug class descriptions. Use null if no info.

CRITICAL RULES:
1. Extract ONLY fields in missing_fields - preserve existing for others
2. Match to input context: ICD: {icd_codes}{gene_matching}, Age: {patient_age}
3. Frequency MUST include "q" prefix (q8h, q12h, q24h)
4. DO NOT invent information - use only what's in content
5. Maintain consistency with existing data AND previous chunks context
6. If previous chunks extracted fields, maintain consistency (e.g., same route, compatible dosing)
7. Accuracy > completeness"""

