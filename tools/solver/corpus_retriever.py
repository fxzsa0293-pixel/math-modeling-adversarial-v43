from __future__ import annotations

from pathlib import Path
from typing import List

from .schemas import CorpusHit, ProblemProfile, read_text_safe

METHOD_HINTS = {
    "forecasting": ["naive baseline", "rolling mean", "ridge/lasso lag model", "tree ensemble", "residual correction"],
    "evaluation_ranking": ["equal weight", "entropy weight", "TOPSIS", "PCA/factor analysis", "stability weighted ranking"],
    "constrained_optimization": ["greedy baseline", "linear programming relaxation", "integer programming", "local search", "robust optimization"],
    "medical_environment_statistics": ["mean baseline", "regularized regression", "interaction terms", "cross validation", "sensitivity analysis"],
    "spatial_or_geostat": ["nearest neighbor baseline", "spatial interpolation", "multi-objective scoring", "Pareto selection"],
    "mechanism_or_physics": ["constant baseline", "first order mechanism", "parameter sweep", "sensitivity analysis"],
    "retail_forecast_operation": ["mean baseline", "weekday profile", "rolling forecast", "pricing policy", "forecast-then-optimize"],
}


def retrieve_corpus(profile: ProblemProfile, corpus_root: Path | None = None, max_hits: int = 8) -> List[CorpusHit]:
    root = Path(corpus_root or profile.data_root)
    query_terms = set(profile.keywords + profile.archetypes)
    candidates = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".txt", ".py", ".m"}]
    hits: List[CorpusHit] = []
    for path in candidates[:800]:
        text = read_text_safe(path, max_chars=8000).lower()
        matched = [term for term in query_terms if term.lower() in text]
        if not matched:
            continue
        score = len(matched) / max(1, len(query_terms))
        hints = []
        for archetype in profile.archetypes:
            hints.extend(METHOD_HINTS.get(archetype, []))
        hits.append(CorpusHit(str(path.stem), str(path), round(score, 4), matched[:12], list(dict.fromkeys(hints))[:8]))
    if not hits:
        hints = []
        for archetype in profile.archetypes:
            hints.extend(METHOD_HINTS.get(archetype, []))
        hits.append(CorpusHit("builtin_method_prior", str(root), 0.1, profile.archetypes, list(dict.fromkeys(hints))[:8]))
    return sorted(hits, key=lambda hit: hit.score, reverse=True)[:max_hits]
