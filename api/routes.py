"""
Preflight — EHR Temporal Validator — API Routes
Three endpoints:
  GET  /health   — liveness check
  POST /validate — full pipeline run
  GET  /schema   — input/output reference
"""
from fastapi import APIRouter, HTTPException, Request
from .schemas import ValidateRequest, ValidateResponse, HealthResponse, SchemaResponse
from .services import run_validation, get_pipeline_resources

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness check. Returns ok if the service is running."""
    return HealthResponse()


@router.post("/validate", response_model=ValidateResponse, tags=["Validation"])
def validate(request: ValidateRequest, http_request: Request):
    """
    Run the full EHR validation pipeline on a patient record.

    Accepts longitudinal clinical notes plus optional structured data
    (ICD codes, medications, diagnoses). Returns:
    - Chart trust score (0–1, relative signal — not a validated clinical accuracy claim)
    - Top 3 clinical issues detected
    - ICD divergence summary
    - Temporal contradictions
    - Named flags (medication conflicts, copy-forward suspicion)
    - Record freshness score

    **Prototype only. Not validated for clinical use.**
    """
    try:
        resources = get_pipeline_resources(http_request)
        return run_validation(
            request,
            entity_extractor=resources["entity_extractor"],
            ollama_resolver=resources["ollama_resolver"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@router.get("/schema", response_model=SchemaResponse, tags=["System"])
def schema():
    """Returns the input and output JSON schema for /validate."""
    return SchemaResponse(
        input_schema=ValidateRequest.schema(),
        output_schema=ValidateResponse.schema(),
        notes=(
            "Pass at least 2 clinical notes per patient for meaningful results. "
            "ICD codes and structured_data are optional but improve divergence analysis. "
            "All dates must be ISO format: YYYY-MM-DD. "
            "Trust score is a relative signal within a cohort — see methodology note in API description."
        ),
    )
