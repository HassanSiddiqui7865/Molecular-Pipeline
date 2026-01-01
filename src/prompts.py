"""
Prompt templates and helper functions for LLM interactions.
"""


EXTRACTION_PROMPT_TEMPLATE = """Extract antibiotic therapy recommendations from medical content.

CONTEXT: Pathogen: {pathogen_display}{resistance_context}{allergy_context} | Severity: {severity_codes} | Age: {age}, Panel: {panel}

SOURCE: {content}

PARSING INSTRUCTIONS: The source content may contain unformatted guidelines, research papers, or medical literature with markdown formatting, citation markers (e.g., **(A-III)**, [1, 2]), reference numbers, table structures, incomplete sentences, and mixed formatting. Parse through this content carefully to extract relevant antibiotic information, ignoring formatting artifacts and focusing on the actual medical recommendations and dosing information.

CRITICAL CONSTRAINT: Extract STRICTLY based on the input context above. Do NOT extract antibiotics or information for other pathogens, conditions, or patient scenarios.

TASK: Extract ONLY antibiotics effective against {pathogen_display}{resistance_task} matching the provided context. Extract ALL available information - do not leave fields null if data exists.

FIELDS:

medical_name: Title Case drug name only. Remove formulation types (Gel, Cream, Ointment, Solution, Suspension, Tablet, Capsule, Injection, etc.) from the name. Extract only the base drug name. Examples: "Metronidazole Gel" → "Metronidazole", "Clindamycin Cream" → "Clindamycin", "Vancomycin Injection" → "Vancomycin". Combinations: convert "Drug1/Drug2", "Drug1-Drug2", "Drug1 and Drug2" → "Drug1 plus Drug2". Examples: "TMP/SMX" → "Trimethoprim plus Sulfamethoxazole", "Imipenem/cilastatin" → "Imipenem plus Cilastatin". Never extract resistance genes as antibiotics.

category: "first_choice" (first-line/preferred/primary), "second_choice" (alternatives/backup), "alternative_antibiotic" (salvage/last resort), or "not_known" (cannot determine). Use contextual clues (order, emphasis) if not explicit. Each antibiotic in one category only.

coverage_for: Format "[Pathogen] [condition]". Use ONLY pathogen matching {pathogen_display} from the input context. Condition based on panel type: Blood/systemic → "bacteremia" or "sepsis" (prefer bacteremia), Urine → "UTI" or "urinary tract infection", Sputum/Respiratory → "pneumonia" or "respiratory infection", CSF → "meningitis", Wound → "wound infection", Other → use appropriate condition from source. Example: "Staphylococcus aureus bacteremia" or "Escherichia coli UTI".

route_of_administration: Extract the route of administration from explicit mentions or infer from dosing context. Use the EXACT route mentioned in the source - do not create combinations. Valid single routes: "IV", "PO", "IM", "Vaginal", "Intravaginal", "Topical", "Ophthalmic", "Otic", "Nasal", "Rectal", "Inhalation", "Sublingual", "Buccal". Route combinations like "IV/PO", "PO/Vaginal" are ONLY valid if the source explicitly states them as a single combined route (e.g., "IV/PO" meaning the drug can be given via either IV or PO interchangeably). DO NOT create route combinations from separate mentions. If source says "IV or PO" or mentions routes separately, extract the most appropriate single route based on context. If the same drug appears with different routes in the source, extract separate entries (one per route). For non-systemic conditions (e.g., vaginal infections, skin infections), extract routes like "Vaginal", "Intravaginal", "Topical" as appropriate. Use null only if no route info exists.

dose_duration: COMPLETE DOSAGE must include dose, frequency (q8h/q12h/q24h), AND duration. Format complete: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". CRITICAL: Convert frequencies BEFORE formatting: BID/twice daily → q12h, TID/three times daily → q8h, QD/daily/once daily → q24h. Preserve ranges ("15-20 mg/kg", "7-14 days"). 

FOR SYSTEMIC ROUTES (IV, PO, IM): Extract dose in mg/kg or mg format. Examples: "500 mg IV q8h for 7 days", "15-20 mg/kg IV q12h for 10 days", "Loading: 2 g IV, then 1 g IV q12h for 7 days". Look for: dose amounts (mg, mg/kg), frequency (q8h/q12h/q24h or BID/TID/QD), duration ("for X days/weeks", "duration", "treatment/course length").

FOR NON-SYSTEMIC ROUTES (Topical, Vaginal, Intravaginal, Ophthalmic, Otic, etc.): Extract dose with percentage concentrations and formulations when present. CRITICAL: Preserve percentage concentrations and formulations exactly as mentioned in source (e.g., "0.75% gel", "2% cream", "1.3% gel", "6.5% ointment") - these are part of the dose specification and must be included. Examples: "0.75% gel 5 g Vaginal q24h for 5 days" (correct - preserves percentage), "2% cream 5 g Vaginal q24h for 7 days" (correct - preserves percentage), "1% cream Topical q12h for 14 days" (correct). Look for: percentage/formulation (X% gel/cream/ointment), amount (g, mL), frequency (q8h/q12h/q24h or BID/TID/QD), duration ("for X days/weeks").

INCOMPLETE DOSAGE: If dosage is missing dose, frequency, OR duration, set dose_duration to NULL. Do NOT extract partial information. The enrichment node will extract missing information later. CRITICAL: A complete dosage requires ALL three components: dose (with percentage/formulation for non-systemic), frequency (q8h/q12h/q24h), AND duration. If any component is missing, set to NULL.

NEVER include the drug name (medical_name) in dose_duration - only include dose amount (with percentage/formulation if present for non-systemic), route, frequency, and duration. Examples: "500 mg IV q8h for 7 days" (correct), "vancomycin 500 mg IV q8h for 7 days" (WRONG - drug name included). Use null if dosage is incomplete (missing dose, frequency, or duration) or if NO dosing info exists.

renal_adjustment: Extract ONLY what is explicitly mentioned in the source. For systemic medications (IV, PO, IM): "No Renal Adjustment" if explicitly stated, or "Adjust dose for CrCl < X mL/min" if CrCl threshold mentioned (use most restrictive if multiple). Look for: "CrCl < X", "creatinine clearance < X", "if/when CrCl < X". Use null if not mentioned in source. For non-systemic medications (topical, vaginal,intravaginal, ophthalmic, otic, etc.): "No Renal Adjustment" if explicitly stated, otherwise use null if not mentioned in source. DO NOT add "Not Applicable" or "N/A" during extraction - only extract what is explicitly stated in the source. Use null if not mentioned. The enrichment node will set "No Renal Adjustment" as default later if needed. Do not duplicate general_considerations.

general_considerations: Extract monitoring ("monitor", "watch for"), warnings, toxicity, interactions, contraindications. Separate with semicolons. Exclude dosing, drug class descriptions, efficacy. Use null only if no safety/monitoring info exists. Do not add default text.

{resistance_genes_section}{allergy_filtering_rule}CRITICAL RULES:
Formatting: BID→q12h, TID→q8h, QD/daily→q24h. Slashes→"plus" for combinations. Extraction: Extract aggressively - use all available info. IMPORTANT: If the same drug (same medical_name) appears with DIFFERENT routes (e.g., "PO" and "Vaginal"), extract them as SEPARATE entries - one entry per route. Do NOT combine or merge entries with different routes. For duplicates with the SAME route, use most complete. Keep ranges intact. Include duration when mentioned anywhere.
{resistance_filtering_rule}
Validation: Each antibiotic in one category. Normalize combinations. Never extract resistance genes as antibiotics. Same drug with different routes = separate entries.

DO NOT EXTRACT: Resistance genes as antibiotics. Ineffective antibiotics (filtered by resistance). Allergenic antibiotics (filtered by allergies). Drug classes without specific names. Experimental drugs (unless recommended)."""


ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE = """Unify antibiotic information from multiple sources into ONE optimized entry.

CRITICAL: Use ONLY information explicitly present in source entries. DO NOT invent, infer, or use medical knowledge. DO NOT combine partial info across sources unless stated together in one source. If missing in all sources, keep null.

ANTIBIOTIC: {antibiotic_name}
ROUTE: {route_of_administration}

IMPORTANT: All entries in this group have the SAME route_of_administration ({route_of_administration}). This route is fixed and cannot be changed. DO NOT combine or unify entries with different routes - they are processed separately. If you see entries with different routes, this indicates a grouping error - use the route specified above ({route_of_administration}).

SOURCES: {entries_list}

TASK: Synthesize most accurate/complete info from all sources into ONE unified entry for this specific route.

SOURCE PRIORITY (use highest priority source available):
1. GUIDELINE SOURCES (Highest Priority): CDC, WHO, IDSA, clinical practice guidelines, treatment guidelines
2. PROTOCOL SOURCES: Hospital protocols, treatment protocols
3. TEXTBOOK/RESEARCH PAPERS: Medical textbooks, research papers
4. CASE REPORTS/OTHER: Case reports, other sources

UNIFICATION RULES:

medical_name: Use only names from entries. Remove formulation types (Gel, Cream, Ointment, Solution, Suspension, Tablet, Capsule, Injection, etc.) from the name. Extract only the base drug name. Most standard/complete form. If differ, prefer most clinically standard. Title Case. Examples: "Metronidazole Gel" → "Metronidazole", "Clindamycin Cream" → "Clindamycin", "Vancomycin Injection" → "Vancomycin". For combinations, use lowercase "plus" (e.g., "Trimethoprim plus Sulfamethoxazole"). Keep "Drug1 plus Drug2" format. Don't change if all agree. Don't invent names.

coverage_for: Use only indications from entries. If guideline sources exist, use coverage_for from guideline sources first. Most specific pathogen/condition. Priority: specific > general. Format "[Pathogen] [condition]" (single primary). Don't combine multiple. Don't invent.

route_of_administration: All entries in this group have the SAME route ({route_of_administration}). Use this exact route value. DO NOT change it. DO NOT create route combinations. DO NOT split route combinations into separate routes. If the route is "IV/PO", keep it as "IV/PO" (it's a single route value meaning the drug can be given via either route). If the route is "PO", keep it as "PO". If the route is "Vaginal", keep it as "Vaginal". Use the route exactly as it appears in the entries. If guideline sources exist and all entries have the same route, use that route. Don't infer routes from dosing - use only what's explicitly in route_of_administration field of entries.

dose_duration: Use ONLY from entries. Don't invent/infer/complete. COMPLETE DOSAGE must include dose, frequency (q8h/q12h/q24h), AND duration. Format complete: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". CRITICAL: The route in dose_duration must match route_of_administration ({route_of_administration}). If an entry's dose_duration mentions a different route, exclude that entry's dose_duration or extract only the part matching the route. Preserve ranges ("15-20 mg/kg", "4–6 weeks", "7-14 days"). 

FOR SYSTEMIC ROUTES (IV, PO, IM): Use dose in mg/kg or mg format from entries. Examples: "500 mg IV q8h for 7 days", "15-20 mg/kg IV q12h for 10 days", "Loading: 2 g IV, then 1 g IV q12h for 7 days". Look for: dose amounts (mg, mg/kg), frequency (q8h/q12h/q24h or BID/TID/QD), duration ("for X days/weeks", "duration", "treatment/course length").

FOR NON-SYSTEMIC ROUTES (Topical, Vaginal, Intravaginal, Ophthalmic, Otic, etc.): Preserve percentage concentrations and formulations exactly as they appear in source entries (e.g., "0.75% gel", "2% cream", "1.3% gel", "6.5% ointment"). These are part of the dose specification and must be included. Examples: "0.75% gel 5 g Vaginal q24h for 5 days" (correct - preserves percentage), "2% cream 5 g Vaginal q24h for 7 days" (correct - preserves percentage), "1% cream Topical q12h for 14 days" (correct). Look for: percentage/formulation (X% gel/cream/ointment), amount (g, mL), frequency (q8h/q12h/q24h or BID/TID/QD), duration ("for X days/weeks").

INCOMPLETE DOSAGE: If dosage is missing dose, frequency, OR duration, set dose_duration to NULL. Do NOT use partial information. Enrichment will extract the missing information later. CRITICAL: A complete dosage requires ALL three components: dose (with percentage/formulation for non-systemic), frequency (q8h/q12h/q24h), AND duration. If any component is missing, set to NULL.

PRIORITY FOR COMPLETE DOSAGES: Since all entries have the same route ({route_of_administration}), select the best dosage based on priority: 1) Guideline sources (highest priority), 2) Shorter duration (e.g., "5 days" over "7 days", "single dose" over multi-day), 3) Simpler regimen (single dose > once daily > multiple times daily), 4) More complete information (all components present), 5) Standard formulations. If guideline sources exist with complete dosage, use guideline dosage (prefer shorter durations when multiple guideline options exist). If no guideline complete dosage, use protocol complete dosage. If no guideline/protocol complete dosage, use textbook/research complete dosage. If no complete dosage exists in any source, set to NULL. CRITICAL: Include ONLY ONE dosage - select the best one based on the priority above. Each dosage must be complete (dose + frequency + duration). Preserve percentage concentrations when present. If any dosage is incomplete, set dose_duration to NULL.

NEVER include the drug name (medical_name) in dose_duration. If a source entry contains the drug name in dose_duration, remove it. Only include: dose amount (with percentage/formulation if present for non-systemic), route, frequency, and duration. Don't combine components across sources. For combination drugs, use total dose if shown as single value.

renal_adjustment: Extract only from entries. If guideline sources exist, use renal adjustment information from guideline sources first. Use "No Renal Adjustment" if explicitly stated or if all sources indicate no adjustment needed. Use "Adjust dose for CrCl < X mL/min" if specific CrCl threshold is mentioned (use most restrictive threshold if multiple sources, prioritizing guideline thresholds). For non-systemic medications (topical, vaginal, ophthalmic, otic, etc.), if route_of_administration is non-systemic and no renal adjustment is mentioned in any source, use null. For systemic medications, if all sources are vague or silent about renal adjustment, use null (vague warnings should be moved to general_considerations). Don't invent thresholds or adjustments not mentioned in sources. Use null if not found in any source. Do not add default text.

general_considerations: Combine distinct notes from all entries. Separate with semicolons. Remove duplicates and near-duplicates. Keep concise (2-3 key points). If guideline sources exist, prioritize guideline information. Include: monitoring requirements, vague renal/hepatic warnings, drug interactions, side effects, contraindications, special population considerations. Exclude: specific renal adjustment thresholds (those go in renal_adjustment), dosing details, external knowledge. Order by priority: monitoring > warnings > interactions > side effects. Use null if no notes found in any source. Do not invent or add default text.

is_combined: True if any source has is_combined=True OR medical_name contains "plus" (case-insensitive). False only if all sources indicate not combined.

is_complete: Set to TRUE only when ALL required fields are present and non-null: medical_name, coverage_for, route_of_administration, dose_duration (must be complete with dose, frequency, AND duration), renal_adjustment, general_considerations. Set to FALSE if any field is missing, null, or incomplete. CRITICAL: If dose_duration is incomplete (missing dose, frequency, or duration), set dose_duration to NULL and is_complete to FALSE (enrichment will extract it). If renal_adjustment is null, set is_complete to FALSE (enrichment will extract it). Note: renal_adjustment and general_considerations can be null - they do not need to be filled for is_complete to be TRUE if they are intentionally null (not missing).

VALIDATION: medical_name in Title Case with lowercase "plus" for combinations. route_of_administration: Must be exactly ({route_of_administration}) - the route all entries share. DO NOT modify it. DO NOT create route combinations. DO NOT split route combinations. If route is "IV/PO", output "IV/PO". If route is "PO", output "PO". If route is "Vaginal", output "Vaginal". If entries somehow have different routes, this indicates an error - use the route specified in the group ({route_of_administration}). dose_duration must include dose, frequency (q8h/q12h/q24h), and duration when complete, or NULL if incomplete. The route in dose_duration must match route_of_administration. If dose_duration mentions a different route than route_of_administration, extract only the dosage matching route_of_administration. renal_adjustment: "Adjust dose for CrCl < X mL/min" or "No Renal Adjustment" if explicitly stated in sources, or null if not found. general_considerations should not duplicate renal_adjustment content. Use null if no notes found. is_combined reflects whether drug is a combination. is_complete is TRUE only when all required fields are present, non-null, and complete. Do not invent information not found in sources.

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

ROUTE CONSISTENCY (applies to ALL fields): If existing_data contains route_of_administration, ALL extracted fields must be consistent with that route. Extract information ONLY for the route specified in existing_data. If content shows different routes, extract ONLY information matching the route in existing_data. If no information exists for that route, set the field to NULL. DO NOT extract information for different routes. DO NOT change the route in existing_data. Maintain route consistency across all chunks.

FIELDS (extract ONLY if in missing_fields):

dose_duration: Extract dosing information matching ICD: {icd_codes}{gene_matching} and Age: {patient_age}. COMPLETE DOSAGE must include dose, frequency (q8h/q12h/q24h), AND duration. Format: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". CRITICAL: Convert frequencies BEFORE formatting: BID/twice daily → q12h, TID/three times daily → q8h, QD/daily/once daily → q24h. Preserve ranges ("15-20 mg/kg", "7-14 days"). Frequency MUST include "q" prefix (q8h, q12h, q24h). Include loading doses if present. The route in dose_duration MUST match the route from existing_data if present.

FOR SYSTEMIC ROUTES (IV, PO, IM): Extract dose in mg/kg or mg format. Examples: "500 mg IV q8h for 7 days", "15-20 mg/kg IV q12h for 10 days", "Loading: 2 g IV, then 1 g IV q12h for 7 days". When multiple dosage options exist, ALWAYS prioritize SHORTER durations (e.g., "5 days" over "7 days", "single dose" over multi-day courses).

FOR NON-SYSTEMIC ROUTES (Topical, Vaginal, Intravaginal, Ophthalmic, Otic, etc.): Preserve percentage concentrations and formulations (e.g., "0.75% gel", "2% cream", "1.3% gel", "6.5% ointment"). Examples: "0.75% gel 5 g Vaginal q24h for 5 days", "2% cream 5 g Vaginal q24h for 7 days", "1% cream Topical q12h for 14 days". When multiple dosage options exist, ALWAYS prioritize SHORTER durations.

INCOMPLETE DOSAGE: If dosage is missing dose, frequency, OR duration, set to NULL. Do NOT extract partial information. A complete dosage requires ALL three components: dose (with percentage/formulation for non-systemic), frequency (q8h/q12h/q24h), AND duration.

PRIORITY: 1) Complete guideline recommendations (with dose, frequency, AND duration), 2) If no complete guidelines exist, set to NULL. Do NOT extract incomplete dosages.

NEVER include the drug name ({medical_name}) in dose_duration. Only include: dose amount (with percentage/formulation if present for non-systemic), route, frequency, and duration. Examples: "500 mg IV q8h for 7 days" (correct), "{medical_name} 500 mg IV q8h for 7 days" (WRONG - remove drug name). Use null if dosage is incomplete or if NO dosing information exists.

route_of_administration: Extract ONLY if missing in existing_data. If existing_data already has a route, preserve it - do not change. If extracting: Use the EXACT route mentioned in the content - do not create combinations. Valid single routes: "IV", "PO", "IM", "Vaginal", "Intravaginal", "Topical", "Ophthalmic", "Otic", "Nasal", "Rectal", "Inhalation", "Sublingual", "Buccal". Route combinations are ONLY valid if the content explicitly states them as a single combined route. DO NOT create route combinations. If content mentions multiple routes separately, extract the most appropriate single route based on context. Use null if no route info.

coverage_for: Format "[Pathogen] [condition]" using clinical terminology (e.g., "MRSA bacteremia", "VRE bacteremia"). Match to patient's clinical condition (ICD: {icd_codes}{gene_context}). Use null if no info.

renal_adjustment: Extract what is mentioned in the content. For systemic medications (IV, PO, IM): "No Renal Adjustment" if explicitly stated, or "Adjust dose for CrCl < X mL/min" if CrCl threshold mentioned (most restrictive if multiple). For non-systemic medications: "No Renal Adjustment" if explicitly stated, otherwise use null. Use null if not mentioned. Do not add default text. Do not duplicate general_considerations.

general_considerations: Extract monitoring, warnings, toxicity, interactions, contraindications. Separate with semicolons. Exclude dosing, drug class descriptions. Use null if no info found. Do not add default text.

CRITICAL RULES: Extract ONLY fields in missing_fields - preserve existing for others. Match to input context: ICD: {icd_codes}{gene_matching}, Age: {patient_age}. Frequency MUST include "q" prefix (q8h, q12h, q24h). DO NOT invent information - use only what's in content. Maintain consistency with existing data AND previous chunks context. If previous chunks extracted fields, maintain consistency (e.g., same route, compatible dosing). Accuracy > completeness."""

SEARCH_PROMPT_TEMPLATE = """Evidence-based antibiotic dosing regimens for {pathogen_name}{resistance_phrase}{condition_text}, specifying drug names, dosing, dosing frequency, route of administration, and treatment duration, with brief antimicrobial stewardship considerations{severity_codes_text}"""

