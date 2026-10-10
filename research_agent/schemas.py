"""Typed data models.

Every structured LLM output is validated against one of these pydantic models.
If validation fails, the LLM client feeds the error back to the model and asks it
to repair its answer (see ``llm.LLMClient.structured``).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# --------------------------------------------------------------------------- papers
class Paper(_Model):
    paper_id: str                      # e.g. "arxiv:2005.11401", "s2:abc123", "demo:0001"
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    venue: str = ""
    citation_count: Optional[int] = None
    source: str = ""                   # arxiv | semantic_scholar | library | demo
    arxiv_id: str = ""

    def short_ref(self) -> str:
        first = self.authors[0].split()[-1] if self.authors else "Unknown"
        etal = " et al." if len(self.authors) > 1 else ""
        return f"{first}{etal} ({self.year or 'n.d.'})"


class Chunk(_Model):
    chunk_id: str                      # "<paper_id>#<n>"
    paper_id: str
    section: str = "Body"
    text: str
    position: int = 0


# --------------------------------------------------------------------------- planning
class SubQuestion(_Model):
    id: str = Field(description="Short id such as SQ1")
    question: str
    rationale: str = ""


class ResearchPlan(_Model):
    sub_questions: list[SubQuestion] = Field(min_length=1, max_length=6)
    search_queries: list[str] = Field(min_length=1, max_length=8)
    key_concepts: list[str] = Field(default_factory=list)
    inclusion_criteria: str = ""

    @field_validator("sub_questions")
    @classmethod
    def _ids(cls, v: list[SubQuestion]) -> list[SubQuestion]:
        for i, sq in enumerate(v, 1):
            if not sq.id:
                sq.id = f"SQ{i}"
        return v


# --------------------------------------------------------------------------- screening
class ScreeningDecision(_Model):
    paper_id: str
    relevance: int = Field(ge=0, le=10)
    reason: str = ""
    sub_question_ids: list[str] = Field(default_factory=list)


class ScreeningResult(_Model):
    decisions: list[ScreeningDecision]


# --------------------------------------------------------------------------- extraction
FindingType = Literal["result", "method", "limitation", "claim", "dataset"]
Verdict = Literal["supported", "partially_supported", "unsupported"]


class ExtractedFinding(_Model):
    claim: str = Field(description="One self-contained factual statement, in your own words")
    chunk_id: str = Field(description="The passage id the claim comes from")
    evidence: str = Field(description="A short verbatim quote (<= 40 words) from that passage")
    finding_type: FindingType = "claim"


class ExtractionResult(_Model):
    findings: list[ExtractedFinding] = Field(default_factory=list)


class Finding(_Model):
    """A claim tied to an exact passage of a paper, plus its verification status."""
    finding_id: str
    sub_question_id: str
    claim: str
    paper_id: str
    chunk_id: str
    evidence: str
    finding_type: str = "claim"
    quote_match: Optional[bool] = None      # did the evidence quote appear in the passage?
    verdict: Optional[str] = None           # supported | partially_supported | unsupported
    verdict_rationale: str = ""
    iteration: int = 0

    @property
    def is_verified(self) -> bool:
        return bool(self.quote_match) and self.verdict in ("supported", "partially_supported")


# --------------------------------------------------------------------------- verification
class ClaimVerdict(_Model):
    finding_id: str
    verdict: Verdict
    rationale: str = ""


class VerificationResult(_Model):
    verdicts: list[ClaimVerdict]


# --------------------------------------------------------------------------- reflection
class Reflection(_Model):
    sufficient: bool = Field(description="True if the evidence answers every sub-question well enough")
    coverage_notes: str = ""
    missing_aspects: list[str] = Field(default_factory=list)
    new_queries: list[str] = Field(default_factory=list, max_length=5)


# --------------------------------------------------------------------------- synthesis
class ThemeStatement(_Model):
    statement: str
    finding_ids: list[str] = Field(default_factory=list)


class MethodRow(_Model):
    paper_id: str
    approach: str
    data_or_setting: str = ""
    key_result: str = ""


class Gap(_Model):
    gap: str
    why_it_matters: str = ""
    finding_ids: list[str] = Field(default_factory=list)
    suggested_direction: str = ""


class Synthesis(_Model):
    agreements: list[ThemeStatement] = Field(default_factory=list)
    disagreements: list[ThemeStatement] = Field(default_factory=list)
    method_comparison: list[MethodRow] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)


# --------------------------------------------------------------------------- reranking / memory
class PassageScore(_Model):
    index: int
    score: float = Field(ge=0, le=10)


class RerankResult(_Model):
    scores: list[PassageScore]


class MemorySummary(_Model):
    summary: str


# --------------------------------------------------------------------------- evaluation
class JudgeScores(_Model):
    faithfulness: int = Field(ge=1, le=5, description="Are claims supported by the cited evidence?")
    coverage: int = Field(ge=1, le=5, description="Does the report answer all parts of the question?")
    synthesis: int = Field(ge=1, le=5, description="Does it compare/contrast rather than list papers?")
    gap_quality: int = Field(ge=1, le=5, description="Are the identified gaps specific and grounded?")
    citation_quality: int = Field(ge=1, le=5, description="Are citations placed on the right claims?")
    justification: str = ""
