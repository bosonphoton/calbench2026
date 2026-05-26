"""Langfuse -> local trace cache.

Pulls Langfuse traces (via the official `langfuse` SDK when available, else
HTTP fallback against `/api/public`) and best-effort converts each into a
`GameTraceBase` so that the local `GameDataset` analysis pipeline can ingest
them.

NOTE: Langfuse cache is for cross-session aggregation; local JSON traces
remain authoritative for `final_state` / `metrics`. Conversion is lossy.
"""

from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from typing import Any, Iterable

import requests

from a2a_engine.schemas import GameConfigBase, GameEvent, GameTraceBase

try:  # optional dep
    from langfuse import Langfuse  # type: ignore

    _HAS_SDK = True
except Exception:  # pragma: no cover
    Langfuse = None  # type: ignore
    _HAS_SDK = False


def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v is not None else default


def _basic_auth() -> tuple[str, str]:
    pk = _env("LANGFUSE_PUBLIC_KEY") or ""
    sk = _env("LANGFUSE_SECRET_KEY") or ""
    return pk, sk


def _base_url() -> str:
    return _env("LANGFUSE_BASE_URL", "https://cloud.langfuse.com") or "https://cloud.langfuse.com"


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def _fetch_traces_http(
    *,
    session_ids: list[str] | None,
    tags: list[str] | None,
    from_timestamp: str | datetime | None,
    limit: int | None,
) -> list[dict]:
    auth = _basic_auth()
    base = _base_url().rstrip("/")
    url = f"{base}/api/public/traces"
    params: dict[str, Any] = {}
    if from_timestamp:
        params["fromTimestamp"] = (
            from_timestamp if isinstance(from_timestamp, str) else from_timestamp.isoformat()
        )
    if tags:
        params["tags"] = tags
    if session_ids:
        # API filters one session at a time; iterate.
        out: list[dict] = []
        for sid in session_ids:
            p = dict(params)
            p["sessionId"] = sid
            out.extend(_paginated_get(url, p, auth, limit))
        return out
    return _paginated_get(url, params, auth, limit)


def _paginated_get(url: str, params: dict, auth: tuple[str, str], limit: int | None) -> list[dict]:
    out: list[dict] = []
    page = 1
    while True:
        p = dict(params)
        p["page"] = page
        p["limit"] = min(limit or 100, 100)
        r = requests.get(url, params=p, auth=auth, timeout=60)
        r.raise_for_status()
        body = r.json()
        data = body.get("data") or []
        out.extend(data)
        if not data or (limit and len(out) >= limit):
            break
        meta = body.get("meta") or {}
        if page >= int(meta.get("totalPages", page)):
            break
        page += 1
    return out[: limit] if limit else out


def _fetch_observations_http(trace_id: str) -> list[dict]:
    auth = _basic_auth()
    base = _base_url().rstrip("/")
    url = f"{base}/api/public/observations"
    return _paginated_get(url, {"traceId": trace_id}, auth, None)


def fetch_traces(
    *,
    session_ids: list[str] | None = None,
    tags: list[str] | None = None,
    from_timestamp: str | datetime | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Return raw Langfuse trace dicts. Uses SDK if importable, else HTTP."""
    if _HAS_SDK:
        try:
            client = Langfuse(
                public_key=_env("LANGFUSE_PUBLIC_KEY"),
                secret_key=_env("LANGFUSE_SECRET_KEY"),
                host=_base_url(),
            )
            kwargs: dict[str, Any] = {}
            if tags:
                kwargs["tags"] = tags
            if from_timestamp:
                kwargs["from_timestamp"] = from_timestamp
            if limit:
                kwargs["limit"] = limit
            if session_ids:
                out: list[dict] = []
                for sid in session_ids:
                    resp = client.api.trace.list(session_id=sid, **kwargs)
                    out.extend(_normalize_list(resp))
                return out
            resp = client.api.trace.list(**kwargs)
            return _normalize_list(resp)
        except Exception:
            # Fall through to HTTP.
            pass
    return _fetch_traces_http(
        session_ids=session_ids, tags=tags, from_timestamp=from_timestamp, limit=limit
    )


def fetch_observations(trace_id: str) -> list[dict]:
    if _HAS_SDK:
        try:
            client = Langfuse(
                public_key=_env("LANGFUSE_PUBLIC_KEY"),
                secret_key=_env("LANGFUSE_SECRET_KEY"),
                host=_base_url(),
            )
            resp = client.api.observations.get_many(trace_id=trace_id)
            return _normalize_list(resp)
        except Exception:
            pass
    return _fetch_observations_http(trace_id)


def _normalize_list(resp: Any) -> list[dict]:
    if resp is None:
        return []
    if isinstance(resp, list):
        return [_to_dict(x) for x in resp]
    data = getattr(resp, "data", None)
    if data is not None:
        return [_to_dict(x) for x in data]
    if isinstance(resp, dict):
        return [_to_dict(x) for x in resp.get("data", [])]
    return []


def _to_dict(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "dict"):
        return x.dict()
    return dict(x.__dict__)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


_AGENT_NAME_RE = re.compile(r"invoke_agent\s+(\S+)")


def langfuse_trace_to_game_trace(lf_trace: dict, observations: list[dict]) -> GameTraceBase:
    """Best-effort reconstruction of a GameTraceBase from a Langfuse trace.

    Lossy: `final_state` and `metrics` are left empty — the local JSON traces
    remain authoritative for those.
    """
    session_id = lf_trace.get("sessionId") or lf_trace.get("session_id")
    game_id = session_id or lf_trace.get("id") or str(uuid.uuid4())

    tags = lf_trace.get("tags") or []
    experiment_name: str | None = None
    for t in tags:
        if isinstance(t, str) and t.startswith("experiment="):
            experiment_name = t.split("=", 1)[1]
            break
    if experiment_name is None and tags:
        experiment_name = str(tags[0])

    started = _parse_dt(lf_trace.get("timestamp") or lf_trace.get("createdAt"))
    ended = _parse_dt(lf_trace.get("updatedAt") or lf_trace.get("endTime"))

    # Sort observations by start time so events are ordered.
    obs_sorted = sorted(
        observations,
        key=lambda o: _parse_dt(o.get("startTime") or o.get("start_time") or o.get("timestamp"))
        or datetime.min,
    )

    events: list[GameEvent] = []
    # Map invoke_agent observation id -> agent name.
    agent_by_obs: dict[str, str] = {}
    for obs in obs_sorted:
        name = obs.get("name") or ""
        m = _AGENT_NAME_RE.match(name)
        if m:
            agent_by_obs[obs["id"]] = m.group(1)

    for obs in obs_sorted:
        ts = _parse_dt(obs.get("startTime") or obs.get("start_time")) or started or datetime.utcnow()
        name = obs.get("name") or ""
        otype = (obs.get("type") or "").lower()

        m = _AGENT_NAME_RE.match(name)
        if m:
            events.append(
                GameEvent(type="invoke_agent", timestamp=ts, data={"agent": m.group(1)})
            )
            continue

        if "chat" in name.lower() or otype in ("generation", "llm"):
            parent = obs.get("parentObservationId") or obs.get("parent_observation_id")
            speaker = agent_by_obs.get(parent) if parent else None
            output = obs.get("output")
            text = _extract_text(output)
            if text is not None:
                events.append(
                    GameEvent(
                        type="message",
                        timestamp=ts,
                        data={"speaker": speaker or "unknown", "text": text},
                    )
                )

    config = GameConfigBase(
        game_name=str(lf_trace.get("name") or "unknown"),
        num_agents=len(set(agent_by_obs.values())) or 0,
        experiment_name=experiment_name,
        experiment_run_id=lf_trace.get("id"),
    )

    return GameTraceBase(
        game_id=str(game_id),
        config=config,
        events=events,
        final_state={},
        metrics={},
        started_at=started or datetime.utcnow(),
        ended_at=ended,
        stopped=False,
    )


def _parse_dt(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        s = str(v).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _extract_text(output: Any) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for k in ("content", "text", "output", "message"):
            v = output.get(k)
            if isinstance(v, str):
                return v
            if isinstance(v, list) and v and isinstance(v[0], dict) and "text" in v[0]:
                return str(v[0]["text"])
        return None
    return None


def fetch_and_convert(
    *,
    session_ids: list[str] | None = None,
    tags: list[str] | None = None,
    from_timestamp: str | datetime | None = None,
    limit: int | None = None,
) -> list[GameTraceBase]:
    """One-call helper: fetch raw traces + observations and convert all."""
    raw = fetch_traces(
        session_ids=session_ids, tags=tags, from_timestamp=from_timestamp, limit=limit
    )
    out: list[GameTraceBase] = []
    for lf in raw:
        tid = lf.get("id")
        obs = fetch_observations(tid) if tid else []
        out.append(langfuse_trace_to_game_trace(lf, obs))
    return out


def using_sdk() -> bool:
    return _HAS_SDK
