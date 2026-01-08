"""
Prompt templates and helper functions for LLM interactions.
"""


EXTRACTION_PROMPT_TEMPLATE = """Extract antibiotic therapy recommendations from medical content.

CONTEXT: Pathogen: {pathogen_display}{resistance_context}{allergy_context} | Severity: {severity_codes} | Age: {age}, Panel: {panel}

{cross_chunk_context}SOURCE CONTENT{chunk_info}: {content}

PARSING INSTRUCTIONS: The source may contain unformatted guidelines, research papers, or medical literature with citations, tables, incomplete sentences, mixed formatting, or fragmented text. Focus strictly on extracting relevant antibiotic recommendations and dosing information. Ignore formatting artifacts, summaries, headings, or unrelated text.

CRITICAL CONSTRAINT: Extract ONLY antibiotics relevant to {pathogen_display}{resistance_task} matching the input context. Do NOT include information for other pathogens, conditions, severities, ages, or patient scenarios. The coverage_for field MUST reference ONLY pathogens from {pathogen_display}. Ignore all other pathogens mentioned in the source content.

FIELDS:

medical_name
Extract the base drug name in Title Case. Remove all formulation details including: Liposomal, Conventional, Standard, Generic, Gel, Cream, Ointment, Solution, Suspension, Tablet, Capsule, Injection, Patch, Syrup, and any other formulation descriptors. For combination drugs, convert slashes, hyphens, or the word "and" into "plus". Examples: "Liposomal amphotericin B" becomes "Amphotericin B". "TMP/SMX" becomes "Trimethoprim plus Sulfamethoxazole". "Imipenem/cilastatin" becomes "Imipenem plus Cilastatin". Never extract resistance genes as antibiotics. Never extract drug classes without a specific drug name.

category
Default category is first_choice.
Assign second_choice ONLY if the source explicitly states phrases such as if first fails, if not suitable, backup option, alternative if failure, use if intolerance, or equivalent wording that clearly indicates use after failure or contraindication.
Assign alternative_antibiotic ONLY if the source explicitly indicates the same active ingredient with a different brand name or formulation.
Do NOT infer categories from order, placement, emphasis, or assumptions.
If no explicit language indicates second_choice or alternative_antibiotic, keep the category as first_choice.
Each antibiotic must appear in only one category.

coverage_for
Extract pathogen names only using ONLY pathogens listed in {pathogen_display}. Do NOT include pathogens not present in the input context. If the drug covers multiple pathogens, list them comma-separated. Examples: "MRSA", "E. coli", "MRSA, E. coli, Klebsiella".

route_of_administration
Extract explicit routes or infer from dosing context. Valid routes are IV, PO, IM, Vaginal, Intravaginal, Topical, Ophthalmic, Otic, Nasal, Rectal, Inhalation, Sublingual, Buccal. Extract combined routes such as IV/PO ONLY if explicitly stated as interchangeable. If the same drug appears with multiple routes, extract separate entries per route. Use null if route information is missing.

dose_duration
Extract complete dosage including dose amount in mg or mg/kg for systemic routes, or percentage concentration and formulation for non-systemic routes, plus route, frequency, and duration. Systemic format is dose route frequency for duration, or Loading dose route, then dose route frequency for duration. Non-systemic format must preserve percentage and formulation exactly, for example 0.75% gel 5 g Vaginal q24h for 5 days. Normalize frequencies strictly as follows. BID or twice daily becomes q12h. TID or three times daily becomes q8h. QD, daily, or once daily becomes q24h. Preserve ranges exactly as written. If multiple complete dosages exist, prioritize shorter duration. Set dose_duration to null if dose, frequency, or duration is missing. Do NOT extract partial dosage. Do NOT invent missing information. Do NOT include the drug name in dose_duration.

renal_adjustment
Extract renal dosing adjustment information using standardized format only. For systemic routes: use "No Renal Adjustment" if explicitly stated that no adjustment is needed, or use "Adjust dose for CrCl < X mL/min" format if a specific creatinine clearance threshold is mentioned (use the most restrictive threshold if multiple are mentioned). For non-systemic routes: use "No Renal Adjustment" only if explicitly stated, otherwise use null. Standardize all variations: "No adjustment recommended", "No adjustment needed", "CrCl 50 mL/min or less" → convert to standard format. If information states "data not available" or similar, use null. Do not invent defaults. Keep detailed renal adjustment explanations in general_considerations field, but use only the standardized value here.

general_considerations
Extract monitoring requirements, warnings, toxicity, interactions, contraindications, special population considerations, and clinically relevant precautions. Include detailed renal adjustment information here (e.g., specific CrCl thresholds, monitoring requirements, dialysis considerations) even though the standardized value appears in renal_adjustment field. Keep concise: maximum 200 characters, 2-3 key points separated by semicolons. Prioritize the most critical clinical considerations. Exclude dosing instructions, drug class descriptions, resistance mechanisms, or efficacy statements. Use null if no information exists.

{resistance_genes_section}{allergy_filtering_rule}

CRITICAL RULES:

Normalize all drug combinations using the word plus.
Create separate entries per route and never merge routes.
Include all available complete dosage information and preserve ranges exactly.
Do not extract resistance genes, ineffective antibiotics, allergenic antibiotics, drug classes without specific drug names, or experimental drugs unless explicitly recommended.
Do not infer category from order, formatting, or emphasis. Use explicit language only.
Set dose_duration to null if any component is missing. Never estimate or guess.
Maintain consistency across chunks using cross_chunk_context.
Aggressively extract all relevant information without adding new facts.
Preserve percentage concentrations and formulations exactly for non-systemic routes.

{resistance_filtering_rule}
"""




ANTIBIOTIC_UNIFICATION_PROMPT_TEMPLATE_OPT = """Unify antibiotic information from multiple sources into ONE optimized entry.

CRITICAL: Use ONLY information explicitly present in source entries. Do NOT invent, infer, or use medical knowledge. Do NOT combine partial info across sources unless stated together in a single source. If missing in all sources, keep null.

ANTIBIOTIC: {antibiotic_name}
ROUTE: {route_of_administration}

IMPORTANT: All entries in this group have the SAME route ({route_of_administration}). This route is fixed and cannot be changed. Process each route separately. If entries have different routes, this indicates a grouping error — use the route specified above.

SOURCES: {entries_list}

TASK: Synthesize the most accurate and complete information from all sources into ONE unified entry for this specific route.

SOURCE PRIORITY (use highest priority source available):
1. GUIDELINE SOURCES: CDC, WHO, IDSA, clinical practice guidelines, treatment guidelines
2. PROTOCOL SOURCES: Hospital protocols, treatment protocols
3. TEXTBOOK/RESEARCH PAPERS: Medical textbooks, research papers
4. CASE REPORTS/OTHER: Case reports, other sources

UNIFICATION RULES:

medical_name: Use only names from entries. Remove all formulation details including: Liposomal, Conventional, Standard, Generic, Gel, Cream, Ointment, Solution, Suspension, Tablet, Capsule, Injection, Patch, Syrup, and any other formulation descriptors. Use the most standard base drug name. Title Case. For combinations, use lowercase "plus" (e.g., "Trimethoprim plus Sulfamethoxazole"). Keep "Drug1 plus Drug2" format. Examples: "Liposomal amphotericin B" → "Amphotericin B". Do not invent names.

coverage_for: Extract pathogen names only from entries. If multiple pathogens are mentioned, combine them comma-separated. Prioritize guideline sources first. Use only pathogen names from entries. Do not invent.

route_of_administration: Must match the fixed route ({route_of_administration}). Do not change or combine routes. If the route in entries differs, ignore it.

dose_duration: Use only complete dosages from entries. Complete means dose + frequency + duration. Format: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". Route in dose_duration must match {route_of_administration}. Preserve ranges ("15-20 mg/kg", "7-14 days"). Incomplete dosages → NULL. Select the best dosage based on source priority and completeness. Only ONE dosage per route.

renal_adjustment: Use only standardized format from entries. Prioritize guideline sources. For systemic routes: use "No Renal Adjustment" if explicitly stated, or "Adjust dose for CrCl < X mL/min" format if a threshold is mentioned (use most restrictive if multiple). For non-systemic routes: use "No Renal Adjustment" only if explicitly stated, otherwise null. Standardize all variations to these formats. Keep detailed renal adjustment explanations (specific thresholds, monitoring, dialysis info) in general_considerations field, but use only the standardized value here. Do not invent.

general_considerations: Combine distinct notes from all entries. Remove duplicates/near-duplicates. Keep concise: maximum 200 characters, 2-3 key points separated by semicolons. Prioritize guideline sources. Include monitoring, warnings, interactions, side effects, contraindications, special population considerations, and detailed renal adjustment information (specific CrCl thresholds, monitoring requirements, dialysis considerations). The standardized renal adjustment value appears in renal_adjustment field, but detailed explanations belong here. Exclude dosing instructions. Null if no info.

is_combined: True if any source has is_combined=True or medical_name contains "plus" (case-insensitive). False only if all sources indicate not combined.

is_complete: True only if all required fields are present and non-null: medical_name, coverage_for, route_of_administration, dose_duration (complete), renal_adjustment, general_considerations. False if any required field is missing, null, or incomplete. Incomplete dose_duration → is_complete = False.

VALIDATION: 
- medical_name: Title Case with lowercase "plus" for combinations.  
- route_of_administration: Must exactly match {route_of_administration}.  
- dose_duration: Complete if dose, frequency, and duration are present; otherwise NULL. Route must match route_of_administration.  
- renal_adjustment: Standardized format only: "No Renal Adjustment" or "Adjust dose for CrCl < X mL/min". Detailed renal info belongs in general_considerations.
- general_considerations: Maximum 200 characters, 2-3 key points. Includes detailed renal adjustment explanations. Null if no notes.
- is_combined and is_complete flags reflect source information only.  

OUTPUT: ONE unified entry with the most accurate, complete info. All formats must match requirements."""


RESISTANCE_GENE_UNIFICATION_PROMPT_TEMPLATE_OPT = """Unify resistance gene information from multiple sources into unified entries.

CRITICAL: Use ONLY information explicitly present in source entries. Do NOT invent, infer, or use external medical knowledge. Do NOT combine partial info across sources unless stated together in a single source. If missing in all sources, keep null.

RESISTANCE GENES FROM SOURCES:
{genes_list}

TASK: For each unique resistance gene, synthesize the most accurate and complete information from all sources into ONE unified entry per gene.

UNIFICATION RULES (per gene):

detected_resistant_gene_name: Use only names from entries. Choose the most standard and clinically recognized form if multiple variants exist. Do not invent or alter names if all sources agree. Resolve conflicts by selecting the standard form.

potential_medication_class_affected: Use only classes explicitly mentioned in entries. Combine all unique classes, remove duplicates, and standardize naming (e.g., "beta-lactam antibiotics" → "beta-lactams"). Separate with commas. Do not invent additional classes.

general_considerations: Use only notes from entries. Combine all distinct points. Separate with semicolons. Remove exact duplicates and near-duplicates (different wording, same meaning). Keep concise (2-3 key points). Include relevant items such as: resistance mechanisms, clinical implications, inhibition by beta-lactamase inhibitors. Follow order: mechanism > clinical implications > inhibitors. Avoid redundant statements (e.g., "hydrolyzes X" vs "confers resistance to X" if meaning is identical). Null if no notes.

VALIDATION: 
- detected_resistant_gene_name matches the gene name exactly.  
- potential_medication_class_affected lists all affected classes without duplicates.  
- general_considerations combines unique mechanisms/considerations without redundancy.  
- No information should be invented.

OUTPUT: Provide a list of unified entries, one per unique resistance gene, matching the rules above."""



ANTIBIOTIC_MATCH_VALIDATION_PROMPT_TEMPLATE_OPT = """Validate whether a drugs.com page matches the antibiotic we are searching for.

SEARCHING FOR: {antibiotic_name}
PAGE TITLE: {page_title}

TASK: Determine if the page title corresponds to the exact same drug, meaning the same active ingredient and medically equivalent form.  

- Return is_match=True if the page title indicates the same drug (same active ingredient).  
- Return is_match=False if the page title indicates a different drug, combination, or unrelated substance.  

CRITICAL: Do not guess or infer — base your answer strictly on the page title text."""

DOSAGE_EXTRACTION_PROMPT_TEMPLATE_OPT = """Extract ONLY missing fields for {medical_name} from drugs.com content.

PATIENT: Age={patient_age} | ICD={icd_codes}{gene_context}{allergy_context}

CRITICAL: Extract STRICTLY based on the input context above. Do NOT extract information for other conditions, ICD codes, ages, or scenarios.

MISSING FIELDS (extract ONLY these): {missing_fields}

EXISTING DATA (preserve if not missing):
{existing_data}

{cross_chunk_context}PAGE CONTENT (chunk {chunk_num} of {total_chunks}):
{chunk_content}

ROUTE CONSISTENCY: If existing_data contains route_of_administration, all extracted fields MUST be consistent with that route. Extract only for that route. If content shows other routes, ignore them. Do not change existing route. Maintain consistency across chunks.

FIELDS (extract ONLY if in missing_fields):

dose_duration: Extract dosing information matching ICD: {icd_codes}{gene_matching} and Age: {patient_age}. COMPLETE DOSAGE requires dose, frequency (q8h/q12h/q24h), AND duration. Format: "[dose] [route] [frequency] for [duration]" or "Loading: [dose] [route], then [dose] [route] [frequency] for [duration]". Convert frequencies: BID/twice daily → q12h, TID/three times daily → q8h, QD/daily/once daily → q24h. Preserve ranges ("15-20 mg/kg", "7-14 days"). Include loading doses if present. Route must match route in existing_data if present.

Systemic routes (IV, PO, IM): Use mg/kg or mg. Prioritize shorter durations if multiple options exist (e.g., 5 days > 7 days, single dose > multi-day).

Non-systemic routes (Topical, Vaginal, Intravaginal, Ophthalmic, Otic, etc.): Preserve exact % concentrations and formulations (e.g., "0.75% gel", "2% cream"). Prioritize shorter durations if multiple options exist.

Incomplete dosage: If dose, frequency, OR duration is missing, set dose_duration = NULL. Do NOT extract partial info. Only complete dosages count.

Do not include drug name in dose_duration. Examples: "500 mg IV q8h for 7 days" (correct), "{medical_name} 500 mg IV q8h for 7 days" (WRONG). Null if incomplete or missing.

route_of_administration: Extract only if missing in existing_data. Preserve existing route if present. Use exact route in content; do not combine routes unless explicitly a single combined route (e.g., IV/PO). Null if no info.

coverage_for: Extract pathogen names only. If multiple pathogens apply, list them comma-separated (e.g., "MRSA", "VRE", "MRSA, E. coli"). Match patient context (ICD: {icd_codes}{gene_context}). Null if no info.

renal_adjustment: Extract using standardized format only. For systemic routes: use "No Renal Adjustment" if explicitly stated, or "Adjust dose for CrCl < X mL/min" format if a threshold is mentioned (use most restrictive if multiple). For non-systemic routes: use "No Renal Adjustment" only if explicitly stated, otherwise null. Standardize all variations to these formats. Keep detailed renal adjustment information (specific thresholds, monitoring, dialysis) in general_considerations field, but use only the standardized value here. Do not invent.

general_considerations: Extract monitoring, warnings, toxicity, interactions, contraindications, special population considerations, and detailed renal adjustment information (specific CrCl thresholds, monitoring requirements, dialysis considerations). Keep concise: maximum 200 characters, 2-3 key points separated by semicolons. The standardized renal adjustment value appears in renal_adjustment field, but detailed explanations belong here. Exclude dosing instructions or drug class descriptions. Null if no info. Do not invent.

CRITICAL RULES:
- Extract ONLY fields in missing_fields; preserve others.  
- Match input context: ICD, Age, gene info.  
- Maintain route consistency across chunks and with existing_data.  
- Accuracy > completeness.  
- Use only content; do not invent missing values."""


SEARCH_PROMPT_TEMPLATE = """Evidence-based antibiotic dosing regimens for {pathogen_name}{resistance_phrase}{condition_text}, specifying drug names, dosing, dosing frequency, route of administration, and treatment duration, with brief antimicrobial stewardship considerations{severity_codes_text}"""


GUIDELINE_CLEANING_PROMPT_TEMPLATE = """Clean and convert unstructured guideline content into plain natural language while preserving ALL context.

CRITICAL REQUIREMENTS:
1. PRESERVE EVERY PIECE OF CONTEXT - This is extremely critical. Do not lose any information, details, numbers, dosages, conditions, warnings, or any medical content.
2. Convert to plain natural language that reads like a clinical guideline itself.
3. Remove ALL formatting: no tables, no bullet points, no markdown, no HTML, no special characters used for formatting.
4. Remove styling artifacts: no bold, italic, headers, footers, citations in brackets, reference numbers.
5. Convert structured data (tables, lists) into flowing natural language sentences.
6. Maintain medical accuracy and preserve all dosages, frequencies, durations, conditions, and clinical information exactly.
7. Write in a clear, professional medical guideline style using complete sentences.

INPUT CONTENT:
{content}

TASK: Transform this content into clean, plain natural language that:
- Reads like a clinical practice guideline
- Preserves ALL medical information, dosages, conditions, warnings, and context
- Uses complete sentences in natural flowing prose
- Removes all formatting, tables, lists, and styling
- Maintains professional medical writing style
- Is ready for further processing without any formatting artifacts

OUTPUT: Provide ONLY the cleaned plain text guideline. No explanations, no metadata, just the cleaned content."""

