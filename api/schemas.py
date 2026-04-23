"""
EHR Temporal Validator — API Schemas
Pydantic models for request validation and response serialization.
"""
from typing import Optional
from pydantic import BaseModel, Field


# ─── Input Models ────────────────────────────────────────────────────────────

class ClinicalNote(BaseModel):
    note_id: str = Field(..., example="note_1")
    date: str = Field(..., example="2023-03-15", description="ISO date YYYY-MM-DD")
    category: str = Field(..., example="Clinic Note")
    text: str = Field(..., example="Patient presents with...")

class StructuredMedication(BaseModel):
    name: str = Field(..., example="metformin")
    dose: Optional[str] = Field(None, example="1000mg BID")
    status: Optional[str] = Field(None, example="active")

class StructuredDiagnosis(BaseModel):
    description: str = Field(..., example="Type 2 diabetes mellitus")
    icd_code: Optional[str] = Field(None, example="E11.9")

class StructuredData(BaseModel):
    medications: Optional[list[StructuredMedication]] = []
    diagnoses: Optional[list[StructuredDiagnosis]] = []

class ValidateRequest(BaseModel):
    patient_id: str = Field(..., example="SYNTH_001")
    notes: list[ClinicalNote] = Field(..., min_items=1)
    icd_codes: Optional[list[str]] = Field(default=[], example=["E11.9", "I10"])
    structured_data: Optional[StructuredData] = None

    class Config:
        json_schema_extra = {
            "example": {
                "patient_id": "SYNTH_001",
                "icd_codes": ["E11.9", "I10", "C50.912"],
                "notes": [
                    {
                        "note_id": "note_1",
                        "date": "2023-03-15",
                        "category": "Operative Note",
                        "text": "Patient with diabetes mellitus type 2 on metformin underwent left mastectomy."
                    },
                    {
                        "note_id": "note_2",
                        "date": "2023-05-10",
                        "category": "Clinic Note",
                        "text": "Patient has a history of diabetes mellitus type 2 on metformin. Patient reported stopping metformin two weeks ago due to GI side effects."
                    }
                ]
            }
        }


# ─── Output Models ───────────────────────────────────────────────────────────

class Top3Issue(BaseModel):
    rank: int
    conflict_type: str
    severity: str
    problem: str
    evidence: list[str]
    action: str
    impact: Optional[str] = None
    confidence: str
    source: Optional[str] = None

class ICDDivergenceSummary(BaseModel):
    summary: str
    free_text_only: list[str]
    code_only: list[str]
    matched: list[str]

class ValidateResponse(BaseModel):
    patient_id: str
    trust_score: float = Field(..., ge=0.0, le=1.0)
    record_freshness: Optional[float] = Field(None, ge=0.0, le=1.0)
    top3_issues: list[Top3Issue]
    icd_divergence: ICDDivergenceSummary
    temporal_contradictions: list[dict]
    named_flags: list[dict]
    processing_time_ms: int
    disclaimer: str = "Synthetic-data prototype. Not validated for clinical use."


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    description: str = "Preflight — Pre-Generation Chart Consistency Engine"


class SchemaResponse(BaseModel):
    input_schema: dict
    output_schema: dict
    notes: str
