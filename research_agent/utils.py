"""Small shared helpers: logging, retries, JSON extraction, text normalisation, fuzzy matching."""
from __future__ import annotations

import difflib
import json
import logging
import re
import time
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

logger = logging.getLogger("research_agent")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------- retries
def retry(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    what: str = "operation",
) -> T:
    """Call ``fn`` with exponential backoff. Re-raises the last error."""
    last: BaseException | None = None
    for i in range(attempts):
        try:
            return fn()
        except exceptions as exc:  # noqa: PERF203
            last = exc
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            logger.warning("%s failed (%s: %s) - retrying in %.1fs", what, type(exc).__name__, str(exc)[:160], delay)
            time.sleep(delay)
    assert last is not None
    raise last


# --------------------------------------------------------------------------- JSON
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Parse JSON from an LLM reply that may contain prose or ``` fences."""
    if text is None:
        raise ValueError("empty response")
    text = text.strip()
    candidates = [text]
    candidates += [m.group(1).strip() for m in _FENCE.finditer(text)]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            candidates.append(text[start : end + 1])
    last_err: Exception | None = None
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError as exc:
            last_err = exc
    raise ValueError(f"could not parse JSON: {last_err}")


# --------------------------------------------------------------------------- text
_STOP = set(
    "a an the of and or in on for to with by from is are was were be been being this that these those it its as at "
    "we our us they their which who whom what when where how than then also can could may might will would should "
    "into over under between about via using use used based such not no but if more most less least very".split()
)
_WORD = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = _WORD.findall(text.lower())
    return toks if keep_stopwords else [t for t in toks if t not in _STOP and len(t) > 1]


def normalize_text(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-").replace("ﬁ", "fi").replace("ﬂ", "fl")
    text = re.sub(r"-\s*\n\s*", "", text)          # de-hyphenate line breaks from PDFs
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def fuzzy_contains(needle: str, haystack: str, threshold: float = 0.85) -> tuple[bool, float]:
    """Does ``needle`` (a quote) appear in ``haystack`` (a passage), allowing small differences?

    Returns (match, best_ratio). Exact normalised substring -> ratio 1.0.
    """
    n, h = normalize_text(needle).strip(" .\"'"), normalize_text(haystack)
    if not n:
        return False, 0.0
    if n in h:
        return True, 1.0
    n_toks, h_toks = n.split(), h.split()
    if not h_toks:
        return False, 0.0
    width = len(n_toks)
    best = 0.0
    step = 1 if len(h_toks) < 600 else 2
    for start in range(0, max(1, len(h_toks) - width + 1), step):
        window = " ".join(h_toks[start : start + width])
        ratio = difflib.SequenceMatcher(None, n, window, autojunk=False).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.98:
                break
    return best >= threshold, best


def token_overlap(a: str, b: str) -> float:
    """Fraction of content words of ``a`` that also appear in ``b``."""
    ta = set(tokenize(a))
    if not ta:
        return 0.0
    return len(ta & set(tokenize(b))) / len(ta)


def truncate(text: str, n: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def chunked(items: list[T], size: int) -> Iterable[list[T]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
