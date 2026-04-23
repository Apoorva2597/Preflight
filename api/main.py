"""
Preflight — EHR Temporal Validator — FastAPI Application
Apoorva Kolhatkar | MHI, University of Michigan

A prototype REST API for validating longitudinal clinical EHR records.
Detects copy-forward propagation, temporal contradictions, and ICD
divergence between free-text notes and structured coded fields.

DISCLAIMER: Prototype only. Not validated for clinical use.
Run locally: uvicorn api.main:app --reload
"""
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src/ to path so pipeline modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from entity_extractor import EntityExtractor
from ollama_resolver import OllamaResolver
from .routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — runs once at startup and once at shutdown.

    Heavy model initialization (GLiNER, Ollama) happens here so that
    the /validate endpoint never pays model-loading cost per request.
    Instances are stored on app.state and injected into services via
    get_pipeline_resources() in services.py.

    Without this, each POST /validate call re-downloads/re-initializes
    GLiNER (~30–60 s) and creates a fresh Ollama connection — making the
    API effectively unusable in any latency-sensitive context.
    """
    print("[Startup] Initializing pipeline resources…")

    # GLiNER / BioClinicalBERT / rule-based (whichever tier is available)
    # This is the expensive call — model weights are loaded once here.
    app.state.entity_extractor = EntityExtractor()
    print("[Startup] EntityExtractor ready.")

    # Ollama connection check + resolver init
    # If Ollama is not running, OllamaResolver degrades gracefully.
    app.state.ollama_resolver = OllamaResolver()
    print("[Startup] OllamaResolver ready.")

    print("[Startup] Pipeline ready — API accepting requests.")
    print("Docs: http://localhost:8000/docs")

    yield  # --- application is running ---

    # Shutdown cleanup (add any teardown here)
    print("[Shutdown] Preflight shutting down.")


app = FastAPI(
    title="Preflight",
    description=(
        "A prototype API service for validating longitudinal clinical EHR records. "
        "Detects copy-forward propagation, temporal contradictions, structured/note "
        "conflicts, and ICD divergence between free-text notes and coded fields. "
        "Built on Michigan Medicine NLP research methodology.\n\n"
        "**Prototype only. Not validated for clinical use.**\n\n"
        "**Trust score methodology:** Penalty-based heuristic. Weights calibrated to "
        "synthetic ground truth. Not yet validated against clinician-annotated charts. "
        "Score is a relative signal within a cohort, not an absolute accuracy claim."
    ),
    version="0.1.0",
    contact={
        "name": "Apoorva Kolhatkar",
        "email": "apokol@umich.edu",
        "url": "https://github.com/Apoorva2597/ehr-temporal-validator",
    },
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
