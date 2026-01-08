"""
Pydantic schemas for data structures in the Molecular Pipeline.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AntibioticEntry(BaseModel):
    """Schema for a single antibiotic entry."""
    medical_name: str
    coverage_for: Optional[str] = None
    route_of_administration: Optional[str] = None
    dose_duration: Optional[str] = None
    renal_adjustment: Optional[str] = None
    general_considerations: Optional[str] = None
    is_combined: bool = False


class AntibioticTherapyPlan(BaseModel):
    """Schema for antibiotic therapy plan with categorized antibiotics."""
    first_choice: List[AntibioticEntry] = Field(default_factory=list)
    second_choice: List[AntibioticEntry] = Field(default_factory=list)
    alternative_antibiotic: List[AntibioticEntry] = Field(default_factory=list)


class ResistanceGeneEntry(BaseModel):
    """Schema for a single resistance gene analysis entry."""
    detected_resistant_gene_name: str
    potential_medication_class_affected: Optional[str] = None
    general_considerations: Optional[str] = None


class CombinedExtractionResult(BaseModel):
    """Combined schema for antibiotic therapy and resistance gene extraction."""
    antibiotic_therapy_plan: AntibioticTherapyPlan
    pharmacist_analysis_on_resistant_gene: List[ResistanceGeneEntry] = Field(default_factory=list)


class SearchResult(BaseModel):
    """Schema for a Perplexity search result."""
    title: str
    url: str
    snippet: str


class PathogenEntry(BaseModel):
    """Schema for a single pathogen entry."""
    pathogen_name: str
    pathogen_count: str


class InputParameters(BaseModel):
    """Schema for input parameters to the pipeline."""
    pathogens: List[PathogenEntry]
    resistant_genes: List[str]
    severity_codes: List[str]
    age: Optional[int] = None
    panel: Optional[str] = None
    systemic: Optional[bool] = None
    allergy: Optional[List[str]] = None


class UnifiedResistanceGene(BaseModel):
    """Schema for unified resistance gene entry."""
    detected_resistant_gene_name: str
    potential_medication_class_affected: Optional[str] = None
    general_considerations: Optional[str] = None


class UnifiedResistanceGenesResult(BaseModel):
    """Schema for unified resistance genes result from LLM."""
    resistance_genes: List[UnifiedResistanceGene] = Field(default_factory=list)


class OutputData(BaseModel):
    """Schema for the final output JSON."""
    input_parameters: InputParameters
    extraction_date: str
    result: Optional[Dict[str, Any]] = Field(default_factory=dict)


class UnifiedAntibioticEntryForSynthesis(BaseModel):
    """Schema for unified antibiotic entry in synthesis node."""
    medical_name: str
    coverage_for: Optional[str] = None
    route_of_administration: Optional[str] = None
    dose_duration: Optional[str] = None
    renal_adjustment: Optional[str] = None
    general_considerations: Optional[str] = None
    is_combined: bool = False
    is_complete: bool = False


class AntibioticMatchResult(BaseModel):
    """Schema for antibiotic match validation."""
    is_match: bool
    reason: str


class DosageExtractionResult(BaseModel):
    """Schema for extracted dosage information from drugs.com."""
    dose_duration: Optional[str] = None
    route_of_administration: Optional[str] = None
    general_considerations: Optional[str] = None
    coverage_for: Optional[str] = None
    renal_adjustment: Optional[str] = None

