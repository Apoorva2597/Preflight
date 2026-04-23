"""
llm_resolver.py — Preflight — EHR Temporal Validator
Apoorva Kolhatkar | Michigan Medicine NLP Research

Abstract base class for longitudinal entity status resolvers.

PURPOSE:
    Defines the interface that any LLM-backed resolver must implement.
    OllamaResolver is the reference implementation (local Ollama inference).
    This abstraction makes the resolver layer swap-able without touching
    any pipeline logic:

        OllamaResolver    — local Ollama (llama3.2, mistral, etc.)
        ClaudeResolver    — Anthropic API (production, HIPAA BAA required)
        AzureOpenAIResolver — Azure OpenAI in-VPC deployment
        MockResolver      — deterministic stub for unit tests

ARCHITECTURE POSITION:
    Stage 2b in the pipeline — after GLiNER extracts entities, before
    confidence scoring and contradiction detection.

        Entity extraction (GLiNER)
                ↓
        [LLMResolver.resolve_all()] — any implementation
                ↓
        Confidence scoring, contradiction detection, ICD divergence

SWAP-IN PATTERN:
    To replace Ollama with a different LLM backend:

        class MyResolver(LLMResolver):
            def resolve_all(self, extracted_entities, all_notes, candidate_profiles=None):
                # call your LLM endpoint
                ...
            def resolve_single(self, entity_text, entity_type, current_note, all_notes):
                # single-entity resolution for debugging
                ...

    Then in main.py lifespan:
        app.state.ollama_resolver = MyResolver()   # same interface, drop-in

PRODUCTION NOTE:
    For a stack running 50,000 encounters/night, the resolver would run against
    an internal LLM deployment (fine-tuned model, Claude API, Azure OpenAI in VPC)
    rather than local Ollama. The interface is identical — only the implementation
    class changes. Prompt templates live in the concrete implementation, not here.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMResolver(ABC):
    """
    Abstract base class for longitudinal entity status resolvers.

    Concrete implementations accept a list of note-level entity dicts
    (as produced by EntityExtractor.extract_all) and return the same
    structure enriched with a "resolved_statuses" key per note.

    The resolver is responsible for:
      - Determining whether each medication/diagnosis is currently active,
        discontinued, held, or uncertain, given the longitudinal note history.
      - Flagging copy-forward risk (entity unchanged across notes with no
        documented clinical re-assessment).
      - Returning structured decisions that downstream modules can consume
        without knowing which LLM produced them.

    Implementations MUST be safe to call when the backend is unavailable —
    graceful degradation (return input unchanged, log a warning) is required.
    """

    @abstractmethod
    def resolve_all(
        self,
        extracted_entities: list[dict],
        all_notes: list[dict],
        candidate_profiles: Optional[dict] = None,
    ) -> list[dict]:
        """
        Enrich extracted entities with LLM-resolved longitudinal status.

        Parameters
        ----------
        extracted_entities : list[dict]
            Output of EntityExtractor.extract_all — one dict per note,
            each containing {"note_id": ..., "entities": {...}}.
        all_notes : list[dict]
            Raw note dicts with {"note_id", "date", "text", ...}.
        candidate_profiles : dict, optional
            Output of LongitudinalStateBuilder — pre-filters which entities
            need resolution. If None, all entities are resolved.

        Returns
        -------
        list[dict]
            Same structure as extracted_entities, with a
            "resolved_statuses" key added to each note's entity dict.
            If the backend is unavailable, returns extracted_entities unchanged.
        """
        ...

    @abstractmethod
    def resolve_single(
        self,
        entity_text: str,
        entity_type: str,
        current_note: dict,
        all_notes: list[dict],
    ) -> Optional[dict]:
        """
        Resolve a single entity. Used for targeted re-evaluation and debugging.

        Parameters
        ----------
        entity_text : str
            The entity surface form, e.g. "metformin 500mg".
        entity_type : str
            "medication" or "diagnosis".
        current_note : dict
            The note where the entity appears.
        all_notes : list[dict]
            All notes in the patient record for context retrieval.

        Returns
        -------
        dict or None
            Structured status decision, or None if backend unavailable.
        """
        ...

    @property
    def enabled(self) -> bool:
        """
        Whether this resolver's backend is reachable and ready.
        Concrete implementations should expose this as a public attribute
        (set in __init__) rather than overriding this property.
        """
        return getattr(self, "_enabled", False)
