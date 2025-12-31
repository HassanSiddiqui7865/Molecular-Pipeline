"""
Prompt templates and helper functions for LLM interactions.
"""
from typing import List, Optional, Dict


EXTRACTION_PROMPT_TEMPLATE = """Extract antibiotic therapy recommendations from medical content.

CONTEXT: Pathogen: {pathogen_display}{resistance_context}{allergy_context} | Severity: {severity_codes} | Age: {age}, Sample: {sample}, Systemic: {systemic}

SOURCE: {content}

PARSING INSTRUCTIONS: The source content may contain unformatted guidelines, research papers, or medical literature with markdown formatting, citation markers (e.g., **(A-III)**, [1, 2]), reference numbers, table structures, incomplete sentences, and mixed formatting. Parse through this content carefully to extract relevant antibiotic information, ignoring formatting artifacts and focusing on the actual medical recommendations and dosing information.

CRITICAL CONSTRAINT: Extract STRICTLY based on the input context above. Do NOT extract antibiotics or information for other pathogens, conditions, or patient scenarios.

TASK: Extract ONLY antibiotics effective against {pathogen_display}{resistance_task} matching the provided context. Extract ALL available information - do not leave fields null if data exists.

FIELDS:

medical_name: Title Case drug name only. Combinations: convert "Drug1/Drug2", "Drug1-Drug2", "Drug1 and Drug2" → "Drug1 plus Drug2". Examples: "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole", "Imipenem/cilastatin" → "Imipenem plus Cilastatin". Never extract resistance genes as antibiotics.

category: "first_choice" (first-line/preferred/primary), "second_choice" (alternatives/backup), "alternative_antibiotic" (salvage/last resort), or "not_known" (cannot determine). Use contextual clues (order, emphasis) if not explicit. Each antibiotic in one category only.

coverage_for: Format "[Pathogen] [condition]". Use ONLY pathogen matching {pathogen_display} from the input context. Condition based on sample type: Blood/systemic → "bacteremia" or "sepsis" (prefer bacteremia), Urine → "UTI" or "urinary tract infection", Sputum/Respiratory → "pneumonia" or "respiratory infection", CSF → "meningitis", Wound → "wound infection", Other → use appropriate condition from source. Example: "Staphylococcus aureus bacteremia" or "Escherichia coli UTI".

route_of_administration: Extract the route of administration from explicit mentions or infer from dosing context. Acceptable values include systemic routes: "IV", "PO", "IM", "IV/PO", "IV/IM", "PO/IM", and non-systemic routes: "Vaginal", "Intravaginal", "Topical", "Ophthalmic", "Otic", "Nasal", "Rectal", "Inhalation", "Sublingual", "Buccal", or combinations like "PO/Vaginal", "Topical/Vaginal", "IV/PO", etc. Use the exact route mentioned in the source. For non-systemic conditions (e.g., vaginal infections, skin infections), extract routes like "Vaginal", "Intravaginal", "Topical" as appropriate. Use null only if no route info exists.

dose_duration: Format "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". CRITICAL: Convert frequencies BEFORE formatting: BID/twice daily → q12h, TID/three times daily → q8h, QD/daily/once daily → q24h. Preserve ranges ("15-20 mg/kg", "7-14 days"). CRITICAL: Preserve percentage concentrations and formulations when present in source (e.g., "0.75% gel", "2% cream", "1.3% gel") - include them in dose_duration exactly as mentioned. Examples: "0.75% gel 5 g Vaginal q24h for 5 days" (correct - preserves percentage), "2% cream 5 g Vaginal q24h for 7 days" (correct - preserves percentage), "500 mg IV q8h for 7 days" (correct). Extract partial info if available: "[dose] [route] [frequency]" if no duration, "[duration]" if only duration. Look for duration phrases: "for X days/weeks", "duration", "treatment/course length". NEVER include the drug name (medical_name) in dose_duration - only include dose amount (with percentage/formulation if present), route, frequency, and duration. Examples: "500 mg IV q8h for 7 days" (correct), "vancomycin 500 mg IV q8h for 7 days" (WRONG - drug name included). Use null only if NO dosing info exists.

renal_adjustment: Extract ONLY what is explicitly mentioned in the source. For systemic medications (IV, PO, IM): "No Renal Adjustment" if explicitly stated, or "Adjust dose for CrCl < X mL/min" if CrCl threshold mentioned (use most restrictive if multiple). Look for: "CrCl < X", "creatinine clearance < X", "if/when CrCl < X". For non-systemic medications (topical, vaginal, ophthalmic, otic, etc.): Use null if not mentioned in source. DO NOT add "Not Applicable" or "N/A" during extraction - only extract what is explicitly stated in the source. Use null if not mentioned. Do not duplicate general_considerations.

general_considerations: Extract monitoring ("monitor", "watch for"), warnings, toxicity, interactions, contraindications. Separate with semicolons. Exclude dosing, drug class descriptions, efficacy. Use null only if no safety/monitoring info exists.

{resistance_genes_section}{allergy_filtering_rule}CRITICAL RULES:
Formatting: BID→q12h, TID→q8h, QD/daily→q24h. Slashes→"plus" for combinations. Extraction: Extract aggressively - use all available info. For duplicates, use most complete. Keep ranges intact. Include duration when mentioned anywhere.
{resistance_filtering_rule}
Validation: Each antibiotic in one category. Normalize combinations. Never extract resistance genes as antibiotics.

DO NOT EXTRACT: Resistance genes as antibiotics. Ineffective antibiotics (filtered by resistance). Allergenic antibiotics (filtered by allergies). Drug classes without specific names. Experimental drugs (unless recommended)."""

ANTIBIOTIC_FILTERING_PROMPT_TEMPLATE = """Evaluate each antibiotic: KEEP or FILTER OUT based on clinical appropriateness.

CONTEXT: Pathogen(s): {pathogen_display}{resistance_context}{allergy_context} | Severity: {severity_codes} | Age: {age} | Sample: {sample} | Systemic: {systemic}

CANDIDATES: {antibiotic_list}

TASK: For each antibiotic, determine should_keep (true/false) and filtering_reason (null if keeping, explanation if filtering).

FILTERING CRITERIA (apply in order):

MICROBIOLOGICAL EFFICACY (Primary - Most Important): FILTER OUT if: No activity against ANY pathogen OR all pathogens inherently resistant OR resistance gene confers complete resistance against ALL pathogens. KEEP if: Effective against ≥1 pathogen OR retains activity despite resistance for ANY pathogen OR documented efficacy in similar resistance patterns. CRITICAL: Multi-pathogen rule - effective against ≥1 pathogen = KEEP (even if ineffective against others).

PATIENT SAFETY: FILTER OUT if: Absolute contraindication for age {age} OR FDA black box warning applicable OR life-threatening risk outweighs benefit OR severe unmanageable drug interactions{allergy_filtering_criteria}. KEEP if: Acceptable risk-benefit for severity {severity_codes} OR relative contraindications manageable OR standard precautions sufficient OR manageable adverse effects.

CLINICAL APPROPRIATENESS: FILTER OUT if: Inadequate penetration for {sample} (e.g., poor CNS for meningitis) OR severity inappropriate (e.g., topical for sepsis) OR route impossible (e.g., PO-only in unconscious) OR documented poor outcomes (e.g., tigecycline for bacteremia). KEEP if: Adequate penetration OR appropriate potency OR suitable route available OR clinical evidence supports use.

GUIDELINE ADHERENCE: FILTER OUT if: Explicitly contraindicated (guidelines say "do not use" or "avoid") OR withdrawn/restricted OR strong recommendation AGAINST. KEEP if: Guideline-recommended OR listed as acceptable alternative/second-line/third-line OR not discouraged OR evidence-supported OR guideline-compatible mechanism. CRITICAL: "Not first-line" or "alternative option" does NOT mean contraindicated - these should be KEPT.

APPROVAL & AVAILABILITY: FILTER OUT if: Investigational only (no FDA/EMA approval) OR not commercially available OR experimental without authorization. KEEP if: FDA/EMA approved (even different indication) OR commercially available OR off-label but evidence-supported.

DECISION FRAMEWORK (exact order):
Effective against ≥1 pathogen? YES→continue, NO→filter "Ineffective"
{resistance_decision_step}{allergy_decision_step}Absolute safety contraindication? YES→filter, NO→continue
Clearly clinically inappropriate? YES→filter, NO→continue
Explicitly contraindicated in guidelines (says "do not use" or "avoid")? YES→filter, NO→continue. Note: "alternative" or "not first-line" = KEEP.
DEFAULT: KEEP (when uncertain, include)

MULTI-PATHOGEN RULE: Effective against S. aureus but NOT E. faecalis → KEEP. Effective against E. faecalis but NOT S. aureus → KEEP. Effective against NEITHER → FILTER OUT. Reasoning: "effective against [pathogen] but not [other]" for partial coverage.

{resistance_genes_evaluation}{allergy_evaluation}

FILTERING REASON FORMAT:
"Ineffective: No activity against any listed pathogen"
"Ineffective: Resistance gene [gene_name] confers resistance against all pathogens"
"Ineffective: Inherently inactive against all pathogens ([list])"
"Safety: Patient allergic to [allergy] - contraindicated"
"Safety: Absolute contraindication - [reason]"
"Clinical: Inadequate penetration/inappropriate for {sample}/{severity_codes}"
"Guideline: Explicitly contraindicated (do not use) for [indication] per [guideline]"

DO NOT FILTER based on: Cost/insurance, TDM requirements, minor/manageable side effects, second/third-line status, lack of head-to-head data, dose adjustment needs.

CRITICAL: Effective against ≥1 pathogen = KEEP. Conservative filtering - only remove with strong evidence. "Alternative" or "second-line" options are VALID and should be KEPT. Only filter if guidelines explicitly say "do not use" or "avoid" or "contraindicated". When uncertain, KEEP."""


ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE = """Unify antibiotic information from multiple sources into ONE optimized entry.

CRITICAL: Use ONLY information explicitly present in source entries. DO NOT invent, infer, or use medical knowledge. DO NOT combine partial info across sources unless stated together in one source. If missing in all sources, keep null.

ANTIBIOTIC: {antibiotic_name}

SOURCES: {entries_list}

TASK: Synthesize most accurate/complete info from all sources into ONE unified entry.

SOURCE PRIORITY (use highest priority source available):
1. GUIDELINE SOURCES (Highest Priority): CDC, WHO, IDSA, clinical practice guidelines, treatment guidelines
2. PROTOCOL SOURCES: Hospital protocols, treatment protocols
3. TEXTBOOK/RESEARCH PAPERS: Medical textbooks, research papers
4. CASE REPORTS/OTHER: Case reports, other sources

UNIFICATION RULES:

medical_name: Use only names from entries. Most standard/complete form. If differ, prefer most clinically standard. Title Case. For combinations, use lowercase "plus" (e.g., "Trimethoprim plus Sulfamethoxazole"). Keep "Drug1 plus Drug2" format. Don't change if all agree. Don't invent names.

coverage_for: Use only indications from entries. If guideline sources exist, use coverage_for from guideline sources first. Most specific pathogen/condition. Priority: specific > general. Format "[Pathogen] [condition]" (single primary). Don't combine multiple. Don't invent.

route_of_administration: If the same antibiotic appears with DIFFERENT routes (e.g., "PO" and "Vaginal"), combine ALL unique routes separated by "/" (e.g., "PO/Vaginal"). If all entries have the same route, use that route. If guideline sources exist, prioritize routes from guideline sources. If multiple guidelines mention different routes, combine all guideline routes. For systemic routes, priority if conflict: IV > IV/PO > PO > IM. For non-systemic routes, preserve the exact route mentioned (Vaginal, Intravaginal, Topical, Ophthalmic, Otic, Nasal, Rectal, etc.). Accept any valid route of administration. Don't infer from dosing.

dose_duration: Use ONLY from entries. Don't invent/infer/complete. COMPLETE DOSAGE must include dose, frequency (q8h/q12h/q24h), AND duration. Format complete: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". CRITICAL: Preserve percentage concentrations and formulations exactly as they appear in source entries (e.g., "0.75% gel", "2% cream", "1.3% gel", "6.5% ointment"). These are part of the dose specification and must be included. Examples: "0.75% gel 5 g Vaginal q24h for 5 days" (correct - preserves percentage), "2% cream 5 g Vaginal q24h for 7 days" (correct - preserves percentage), "500 mg IV q8h for 7 days" (correct). INCOMPLETE DOSAGE: If dosage is missing dose, frequency, or duration, set dose_duration to NULL. Enrichment will extract the missing information later. PRIORITY FOR COMPLETE DOSAGES: If guideline sources exist with complete dosage, use guideline dosage (prefer shorter durations when multiple guideline options exist). If no guideline complete dosage, use protocol complete dosage. If no guideline/protocol complete dosage, use textbook/research complete dosage. If no complete dosage exists in any source, set to NULL. DIFFERENT ROUTES: If the same antibiotic has DIFFERENT routes with DIFFERENT dosages (e.g., "500 mg PO q12h for 7 days" and "0.75% gel 5 g Vaginal q24h for 5 days"), combine them with " or " separator: "500 mg PO q12h for 7 days or 0.75% gel 5 g Vaginal q24h for 5 days". Each dosage must be complete (dose + frequency + duration). Preserve percentage concentrations when present. If any dosage is incomplete, set dose_duration to NULL. SAME ROUTE: If all entries have the same route, prefer guideline sources, then prefer shorter durations when multiple complete options exist (e.g., "7 days" over "14 days", "5 days" over "7 days"). If only incomplete dosages exist, set to NULL. CRITICAL: NEVER include the drug name (medical_name) in dose_duration. If a source entry contains the drug name in dose_duration, remove it. Only include: dose amount (with percentage/formulation if present), route, frequency, and duration. Preserve ranges ("15-20 mg/kg", "4–6 weeks"). Don't combine components across sources. For combination drugs, use total dose if shown as single value.

renal_adjustment: Extract only from entries. If guideline sources exist, use renal adjustment information from guideline sources first. Use "No Renal Adjustment" if explicitly stated or if all sources indicate no adjustment needed. Use "Adjust dose for CrCl < X mL/min" if specific CrCl threshold is mentioned (use most restrictive threshold if multiple sources, prioritizing guideline thresholds). For non-systemic medications (topical, vaginal, ophthalmic, otic, etc.), if route_of_administration is non-systemic and no renal adjustment is mentioned in any source, use "Not Applicable" or "N/A". For systemic medications, set to null if all sources are vague or silent about renal adjustment (vague warnings should be moved to general_considerations). Don't invent thresholds or adjustments not mentioned in sources.

general_considerations: Combine distinct notes from all entries. Separate with semicolons. Remove duplicates and near-duplicates. Keep concise (2-3 key points). If guideline sources exist, prioritize guideline information. Include: monitoring requirements, vague renal/hepatic warnings, drug interactions, side effects, contraindications, special population considerations. Exclude: specific renal adjustment thresholds (those go in renal_adjustment), dosing details, external knowledge. Order by priority: monitoring > warnings > interactions > side effects. Use null if no notes found.

is_combined: True if any source has is_combined=True OR medical_name contains "plus" (case-insensitive). False only if all sources indicate not combined.

is_complete: Set to TRUE only when ALL required fields are present and non-null: medical_name, coverage_for, route_of_administration, dose_duration (must be complete with dose, frequency, AND duration), renal_adjustment, general_considerations. Set to FALSE if any field is missing, null, or incomplete. CRITICAL: If dose_duration is incomplete (missing dose, frequency, or duration), set dose_duration to NULL and is_complete to FALSE (enrichment will extract it). If renal_adjustment is null, set is_complete to FALSE (enrichment will extract it).

VALIDATION: medical_name in Title Case with lowercase "plus" for combinations. route_of_administration: Any valid route (IV, PO, IM, Vaginal, Intravaginal, Topical, Ophthalmic, Otic, Nasal, Rectal, Inhalation, or combinations like IV/PO, PO/Vaginal, etc.). dose_duration must include dose, frequency (q8h/q12h/q24h), and duration when complete, or NULL if incomplete. renal_adjustment: For systemic routes: "Adjust dose for CrCl < X mL/min", "No Renal Adjustment", or null. For non-systemic routes: "Not Applicable", "N/A", or null. general_considerations should not duplicate renal_adjustment content. is_combined reflects whether drug is a combination. is_complete is TRUE only when all fields are present, non-null, and complete. Do not invent information not found in sources.

OUTPUT: ONE unified entry with best synthesized info. All formats match requirements."""

RESISTANCE_GENE_UNIFICATION_PROMPT_TEMPLATE = """Unify resistance gene information from multiple sources into unified entries.

CRITICAL: Use ONLY information explicitly present in source entries. DO NOT invent, infer, or use medical knowledge. DO NOT combine partial info across sources unless stated together in one source. If missing in all sources, keep null.

RESISTANCE GENES FROM SOURCES:
{genes_list}

TASK: For each unique resistance gene, synthesize most accurate/complete info from all sources into ONE unified entry per gene.

UNIFICATION RULES (per gene):

detected_resistant_gene_name: Use only names from entries. Most standard/complete form. If differ, prefer most clinically standard. Don't change if all agree. Don't invent names. Conflict resolution: most standard form.

potential_medication_class_affected: Use only classes from entries. Combine all unique classes mentioned. Remove duplicates (e.g., "beta-lactams" and "beta-lactam antibiotics" → "beta-lactams"). Format: combine with commas. Examples: "beta-lactams, penicillins, cephalosporins, monobactams". Don't invent classes. Conflict resolution: combine all unique classes, remove duplicates.

general_considerations: Use only notes from entries. Combine ALL distinct notes. Separate with semicolons. CRITICAL: Remove exact duplicates AND near-duplicates (same meaning, different wording). Keep concise - 2-3 key points maximum. Include: resistance mechanisms, how gene confers resistance, clinical implications, inhibition by beta-lactamase inhibitors. Order: mechanism > clinical implications > inhibitors. Remove redundant statements (e.g., "hydrolyzes X" and "confers resistance to X" if they mean the same). Null if no notes. Conflict resolution: combine unique points, remove duplicates and near-duplicates.

VALIDATION: detected_resistant_gene_name matches gene name. potential_medication_class_affected lists all affected classes without duplicates. general_considerations combines unique mechanisms/considerations without redundancy. No invented info.

OUTPUT: List of unified entries, one per unique resistance gene. All formats match requirements."""


ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE = """Validate if drugs.com page matches the antibiotic we're searching for.

SEARCHING FOR: {antibiotic_name}
PAGE TITLE: {page_title}

TASK: Determine if page title indicates the same drug (same active ingredient/medically equivalent).

Return is_match=True if same drug, is_match=False if different drug."""


DOSAGE_EXTRACTION_PROMPT_TEMPLATE = """Extract ONLY missing fields for {medical_name} from drugs.com content.

PATIENT: Age={patient_age} | ICD={icd_codes}{gene_context}{allergy_context}

CRITICAL: Extract STRICTLY based on the input context above. Do NOT extract information for other conditions, ICD codes, ages, or scenarios.

MISSING FIELDS (extract ONLY these): {missing_fields}

EXISTING DATA (context only, preserve if not missing):
{existing_data}

{cross_chunk_context}PAGE CONTENT (chunk {chunk_num} of {total_chunks}):
{chunk_content}

FIELDS (extract ONLY if in missing_fields):

dose_duration: Extract dosing information matching ICD: {icd_codes}{gene_matching} and Age: {patient_age}. PRIORITY: 1) Complete guideline recommendations (with dose, frequency, AND duration) - prioritize these first, 2) If no complete guidelines, use any available information (even if incomplete). Format complete: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". Format incomplete: "[dose] [route] [frequency]" if missing duration, or "[duration]" if only duration available. Frequency MUST include "q" prefix (q8h, q12h, q24h). Include loading doses if present. CRITICAL: Preserve percentage concentrations and formulations when present in content (e.g., "0.75% gel", "2% cream", "1.3% gel", "6.5% ointment") - these are part of the dose specification and must be included. Examples: "0.75% gel 5 g Vaginal q24h for 5 days" (correct - preserves percentage), "2% cream 5 g Vaginal q24h for 7 days" (correct - preserves percentage), "500 mg IV q8h for 7 days" (correct). NEVER include the drug name ({medical_name}) in dose_duration - only extract dose amount (with percentage/formulation if present), route, frequency, and duration. If the content mentions the drug name with dosing, extract only the dosing part. Examples: "500 mg IV q8h for 7 days" (correct), "{medical_name} 500 mg IV q8h for 7 days" (WRONG - remove drug name). Use null only if NO dosing information exists in content.

route_of_administration: Extract the route of administration from explicit mentions or infer from dosing context. Acceptable values include systemic routes: "IV", "PO", "IM", "IV/PO", "IV/IM", "PO/IM", and non-systemic routes: "Vaginal", "Intravaginal", "Topical", "Ophthalmic", "Otic", "Nasal", "Rectal", "Inhalation", "Sublingual", "Buccal", or combinations like "PO/Vaginal", "Topical/Vaginal", "IV/PO", etc. Use the exact route mentioned in the content. For non-systemic conditions, extract routes like "Vaginal", "Intravaginal", "Topical" as appropriate. Use null if no route info.

coverage_for: Format "[Pathogen] [condition]" using clinical terminology (e.g., "MRSA bacteremia", "VRE bacteremia"). Match to patient's clinical condition (ICD: {icd_codes}{gene_context}). Use null if no info.

renal_adjustment: Extract ONLY what is explicitly mentioned in the content. For systemic medications (IV, PO, IM): "No Renal Adjustment" if explicitly stated OR if not found/not mentioned. If CrCl threshold mentioned, use "Adjust dose for CrCl < X mL/min" (most restrictive if multiple). Default to "No Renal Adjustment" if not mentioned. For non-systemic medications (topical, vaginal, ophthalmic, otic, etc.): Use null if not mentioned in content. DO NOT add "Not Applicable" or "N/A" during extraction - only extract what is explicitly stated in the content. Do not duplicate general_considerations.

general_considerations: Extract monitoring, warnings, toxicity, interactions, contraindications. Separate with semicolons. Exclude dosing, drug class descriptions. Use null if no info.

CRITICAL RULES: Extract ONLY fields in missing_fields - preserve existing for others. Match to input context: ICD: {icd_codes}{gene_matching}, Age: {patient_age}. Frequency MUST include "q" prefix (q8h, q12h, q24h). DO NOT invent information - use only what's in content. Maintain consistency with existing data AND previous chunks context. If previous chunks extracted fields, maintain consistency (e.g., same route, compatible dosing). Accuracy > completeness."""

