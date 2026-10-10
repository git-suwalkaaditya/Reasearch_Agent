"""Runtime configuration, read from environment variables (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value[:1] in ("'", '"'):
            value = value[1:].split(value[0], 1)[0]
        else:
            value = value.split(" #", 1)[0].strip()   # allow inline comments
        if value:
            os.environ.setdefault(key.strip(), value)


_load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    # --- LLM (any OpenAI-compatible endpoint: OpenAI, Groq, Gemini, Ollama, OpenRouter, vLLM ...)
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", "https://api.openai.com/v1"))
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.1")))
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT", 120))
    use_json_mode: bool = field(default_factory=lambda: _env_bool("LLM_JSON_MODE", True))
    use_tool_calling: bool = field(default_factory=lambda: _env_bool("LLM_TOOL_CALLING", True))

    # --- Embeddings / reranking
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))  # auto|sentence-transformers|openai|lsa
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    reranker_backend: str = field(default_factory=lambda: _env("RERANKER_BACKEND", "auto"))  # auto|cross-encoder|llm|none
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # --- Search sources
    semantic_scholar_api_key: str = field(default_factory=lambda: _env("SEMANTIC_SCHOLAR_API_KEY"))
    sources: tuple = field(default_factory=lambda: tuple(s.strip() for s in _env("SEARCH_SOURCES", "arxiv,semantic_scholar").split(",") if s.strip()))

    # --- Agent budget
    max_iterations: int = field(default_factory=lambda: _env_int("MAX_ITERATIONS", 2))
    max_papers: int = field(default_factory=lambda: _env_int("MAX_PAPERS", 8))
    papers_per_iteration: int = field(default_factory=lambda: _env_int("PAPERS_PER_ITERATION", 5))
    results_per_query: int = field(default_factory=lambda: _env_int("RESULTS_PER_QUERY", 8))
    max_tool_steps: int = field(default_factory=lambda: _env_int("MAX_TOOL_STEPS", 8))
    passages_per_question: int = field(default_factory=lambda: _env_int("PASSAGES_PER_QUESTION", 6))
    max_pdf_pages: int = field(default_factory=lambda: _env_int("MAX_PDF_PAGES", 30))
    chunk_chars: int = field(default_factory=lambda: _env_int("CHUNK_CHARS", 1000))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 150))

    # --- Storage
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", str(ROOT / "workspace"))))

    # --- Offline demo mode (mock LLM + bundled demo papers, no network, no keys)
    offline: bool = field(default_factory=lambda: _env_bool("OFFLINE", False))

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "reports").mkdir(exist_ok=True)
        (self.data_dir / "pdfs").mkdir(exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "research.db"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def pdf_dir(self) -> Path:
        return self.data_dir / "pdfs"

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems."""
        problems = []
        if not self.offline:
            if not self.llm_model:
                problems.append("LLM_MODEL is not set (or run with --offline).")
            if not self.llm_api_key and "localhost" not in self.llm_base_url and "127.0.0.1" not in self.llm_base_url:
                problems.append("LLM_API_KEY is not set (not needed for a local Ollama endpoint).")
        return problems
