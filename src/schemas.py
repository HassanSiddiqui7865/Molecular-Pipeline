"""
Pydantic schemas for data structures in the Molecular Pipeline.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AntibioticEntry(BaseModel):
    """Schema for a single antibiotic entry."""
    medical_name: str = Field(..., description="Exact drug name as written in source. For combination therapies, use format 'Drug1 plus Drug2' (e.g., 'Ampicillin plus Ceftriaxone')")
    coverage_for: str = Field(..., description="Specific indication/condition it treats (e.g., 'VRE bacteremia', 'uncomplicated cystitis'). Be specific and concise.")
    route_of_administration: Optional[str] = Field(None, description="Route in standardized format: 'IV', 'PO', 'IM', 'IV/PO', 'IV or PO', 'Oral'. Extract from dose_duration if route is mentioned there but not separately. If not mentioned, use null (NOT 'Not specified').")
    dose_duration: Optional[str] = Field(None, description="Dosing information in format: 'dose,route,frequency,duration'. Examples: '1000 mg,IV,q8h,10-14 days' or '15 mg/kg,IV,once daily,7 days' or '500 mg,PO,q12h,null'. For weight-based: '15 mg/kg,IV,once daily,7 days'. For fixed dose: '1000 mg,IV,q8h,10 days'. For combination therapies: '1000 mg (drug1),IV,q8h,10 days plus 600 mg (drug2),IV,q12h,10 days'. Use 'null' for missing components. Remove verbose phrases like 'during days +97-139', 'from first negative culture', 'high-dose', study details, averages, or case-specific information. If truly not mentioned, use null (NOT 'Not specified' or 'Not mentioned').")
    renal_adjustment: Optional[str] = Field(None, description="Renal adjustment information in format: 'Adjust dose in CrCl < X mL/min' or 'Dose adjust for renal dysfunction' or 'Avoid if CrCl < X mL/min'. Consider patient age if provided. If not mentioned, use null (NOT 'Not specified').")
    general_considerations: Optional[str] = Field(None, description="Clinical notes, warnings, monitoring requirements, contraindications. Be comprehensive but concise. If nothing mentioned, use null (NOT 'Not specified').")


class AntibioticTherapyPlan(BaseModel):
    """Schema for antibiotic therapy plan with categorized antibiotics."""
    first_choice: List[AntibioticEntry] = Field(default_factory=list, description="First-line/preferred antibiotics. Place here if source explicitly states 'first-line', 'preferred', 'recommended', 'guideline recommends', 'primary', or lists it as the primary/preferred option.")
    second_choice: List[AntibioticEntry] = Field(default_factory=list, description="Second-line alternatives. Place here if source states 'alternative', 'second-line', 'if first-line unavailable', 'backup', or lists as secondary/backup option.")
    alternative_antibiotic: List[AntibioticEntry] = Field(default_factory=list, description="Alternative options. Place here if source mentions as 'other option', 'salvage therapy', 'last resort', 'consider if', or lists without clear preference/priority.")
    not_known: List[AntibioticEntry] = Field(default_factory=list, description="Antibiotics mentioned in source but category is unclear/not specified. Use this ONLY when you are NOT confident about the category - when the source mentions the antibiotic but does NOT clearly indicate if it's first-line, second-line, or alternative.")


class ResistanceGeneEntry(BaseModel):
    """Schema for a single resistance gene analysis entry."""
    detected_resistant_gene_name: str = Field(..., description="Standard name of the detected resistance gene")
    potential_medication_class_affected: str = Field(..., description="Medication classes affected by this resistance gene")
    general_considerations: Optional[str] = Field(None, description="Resistance mechanism, mechanism of action, clinical impact, and treatment implications. If nothing mentioned, use null.")


class SourceResult(BaseModel):
    """Schema for a single source extraction result."""
    source_url: str = Field(..., description="URL of the source")
    source_title: str = Field(..., description="Title of the source")
    source_index: int = Field(..., description="Index of the source (1-based)")
    antibiotic_therapy_plan: AntibioticTherapyPlan = Field(default_factory=AntibioticTherapyPlan, description="Extracted antibiotic therapy plan")
    pharmacist_analysis_on_resistant_gene: List[ResistanceGeneEntry] = Field(default_factory=list, description="Resistance gene analysis")


class SearchResult(BaseModel):
    """Schema for a Perplexity search result."""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Content snippet from the source")


class InputParameters(BaseModel):
    """Schema for input parameters to the pipeline."""
    pathogen_name: str = Field(..., description="Name of the pathogen")
    resistant_gene: str = Field(..., description="Detected resistance gene")
    pathogen_count: str = Field(..., description="Pathogen count (e.g., '10^5 CFU/ML')")
    severity_codes: str = Field(..., description="Severity codes (e.g., 'A41.9, B95.62')")
    age: Optional[int] = Field(None, description="Patient age in years (e.g., 45, 30, 65)")


# Note: PipelineState is defined in graph.py as TypedDict for LangGraph compatibility


class UnifiedAntibioticEntry(BaseModel):
    """Schema for unified antibiotic entry."""
    medical_name: str = Field(..., description="Antibiotic name")
    coverage_for: Optional[str] = Field(None, description="Specific indication/condition it treats")
    route_of_administration: Optional[str] = Field(None, description="Synthesized route of administration")
    dose_duration: Optional[str] = Field(None, description="Synthesized dosing and duration information")
    renal_adjustment: Optional[str] = Field(None, description="Synthesized renal adjustment information")
    general_considerations: Optional[str] = Field(None, description="Synthesized general considerations")
    final_category: str = Field(..., description="Final category: first_choice, second_choice, or alternative_antibiotic")
    mentioned_in_sources: List[int] = Field(..., description="Source indices where this antibiotic was mentioned")


class UnifiedResistanceGene(BaseModel):
    """Schema for unified resistance gene entry."""
    detected_resistant_gene_name: str = Field(..., description="Synthesized gene name")
    potential_medication_class_affected: str = Field(..., description="Synthesized medication classes affected")
    general_considerations: Optional[str] = Field(None, description="Synthesized general considerations")


class UnifiedAntibioticsResult(BaseModel):
    """Schema for unified antibiotics result from LLM."""
    first_choice: List[UnifiedAntibioticEntry] = Field(default_factory=list, description="Unified first-choice antibiotics")
    second_choice: List[UnifiedAntibioticEntry] = Field(default_factory=list, description="Unified second-choice antibiotics")
    alternative_antibiotic: List[UnifiedAntibioticEntry] = Field(default_factory=list, description="Unified alternative antibiotics")


class UnifiedResistanceGenesResult(BaseModel):
    """Schema for unified resistance genes result from LLM."""
    resistance_genes: List[UnifiedResistanceGene] = Field(default_factory=list, description="Unified resistance genes")


class OutputData(BaseModel):
    """Schema for the final output JSON."""
    input_parameters: InputParameters
    extraction_date: str
    result: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Synthesized result from all sources")


class RankedAntibioticEntry(BaseModel):
    """Schema for a ranked antibiotic entry."""
    medical_name: str
    ranked_category: str = Field(..., description="Category after ranking: 'first_choice', 'second_choice', or 'alternative_antibiotic'")
    ranking_reason: str = Field(..., description="Brief explanation (1-2 sentences) of why it's ranked in this category")


class RankedAntibioticsResult(BaseModel):
    """Schema for ranked antibiotics result."""
    ranked_antibiotics: List[RankedAntibioticEntry] = Field(default_factory=list, description="List of ranked antibiotics with their new categories")

