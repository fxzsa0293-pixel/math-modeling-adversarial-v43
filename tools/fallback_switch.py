"""Small auditable primary/fallback execution primitive.

The callbacks are deliberately injected by the caller so domain solvers remain
responsible for their own contracts and verification.  This module only
records the switching semantics and never hides a primary failure.
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_with_fallback(
    primary_route: str,
    fallback_route: str,
    primary: Callable[[], Any],
    fallback: Callable[[], Any],
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    if not primary_route.strip() or not fallback_route.strip() or primary_route == fallback_route:
        raise ValueError("distinct_primary_and_fallback_routes_required")
    events: list[dict[str, Any]] = []
    started = _now()
    try:
        value = primary()
        events.append({"route": primary_route, "status": "succeeded", "at": _now()})
        result = {"status": "primary_succeeded", "selected_route": primary_route, "value": value}
    except Exception as exc:  # noqa: BLE001 - failure is the evidence being rehearsed
        events.append({
            "route": primary_route, "status": "failed", "at": _now(),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(limit=3),
        })
        try:
            value = fallback()
            events.append({"route": fallback_route, "status": "succeeded", "at": _now()})
            result = {"status": "fallback_succeeded", "selected_route": fallback_route, "value": value}
        except Exception as fallback_exc:  # noqa: BLE001
            events.append({
                "route": fallback_route, "status": "failed", "at": _now(),
                "error_type": type(fallback_exc).__name__, "error": str(fallback_exc),
                "traceback": traceback.format_exc(limit=3),
            })
            result = {"status": "all_routes_failed", "selected_route": "", "value": None}
    payload = {
        "version": "v44-fallback-switch-1", "kind": "fallback_switch_rehearsal",
        "started_at": started, "completed_at": _now(),
        "primary_route": primary_route, "fallback_route": fallback_route,
        "events": events, "passed": result["status"] in {"primary_succeeded", "fallback_succeeded"},
        **result,
    }
    if evidence_path is not None:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        payload["evidence_path"] = str(evidence_path.resolve())
    return payload
