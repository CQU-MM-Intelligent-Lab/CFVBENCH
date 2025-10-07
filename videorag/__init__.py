"""videorag package init.

Use lazy imports to avoid pulling heavy deps (imagebind/torchaudio) unless needed.
This prevents environment-level binary mismatches from breaking unrelated modules
like `_llm` or `iterative_refinement`.
"""

__all__ = ["VideoRAG", "QueryParam"]

def __getattr__(name: str):
    if name in {"VideoRAG", "QueryParam"}:
        from .videorag import VideoRAG, QueryParam
        return {"VideoRAG": VideoRAG, "QueryParam": QueryParam}[name]
    raise AttributeError(name)