"""CLI: load an experiment YAML, expand batches, run games in-process, write traces.

Usage:

    a2a-run path/to/experiment.yaml --max-parallelism 4 --results-dir ./results

Game classes must be registered in ``a2a_engine.registry`` before invocation
(typically by importing the downstream benchmark package).
"""

import argparse
import getpass
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from a2a_engine import (
    GameTraceBase,
    expand_batches,
    get_game,
    list_games,
    load_experiment,
    run_with_parallelism,
    write_trace,
)
from a2a_engine._context import current_conversation_id
from a2a_engine.tracing_otel import get_tracer, init_tracing, shutdown_tracing

log = logging.getLogger("expt_runner")
_manifest_lock = threading.Lock()


def _make_run_context(experiment_name: str, batch_label: str, resolved_cfg: dict,
                      run_idx: int, dry_run: bool) -> dict:
    cfg = dict(resolved_cfg)
    cfg["experiment_name"] = experiment_name
    cfg["experiment_run_id"] = f"{experiment_name}.{batch_label}.{run_idx}"
    cfg.setdefault("game_name", cfg.get("game_name"))
    return {"config": cfg, "dry_run": dry_run, "experiment_name": experiment_name,
            "batch_label": batch_label, "run_idx": run_idx}


def _git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _metadata_path_for(trace_path: Path) -> Path:
    return trace_path.with_name(f"{trace_path.stem}.metadata.json")


def _manifest_path(results_dir: Path, experiment_name: str) -> Path:
    return results_dir / experiment_name / "_run_manifest.jsonl"


def _load_completed_run_ids(results_dir: Path, experiment_name: str) -> set[str]:
    base = results_dir / experiment_name
    completed: set[str] = set()
    manifest = _manifest_path(results_dir, experiment_name)
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            run_id = record.get("experiment_run_id")
            trace_path = record.get("local_trace_path")
            if run_id and (not trace_path or Path(trace_path).exists()):
                completed.add(str(run_id))

    if not base.exists():
        return completed

    for path in base.glob("*.metadata.json"):
        try:
            metadata = json.loads(path.read_text())
        except Exception:
            continue
        run_id = metadata.get("experiment_run_id")
        trace_path = metadata.get("local_trace_path")
        if run_id and (not trace_path or Path(trace_path).exists()):
            completed.add(str(run_id))

    for path in base.glob("*.json"):
        if path.name.endswith(".metadata.json"):
            continue
        try:
            trace = GameTraceBase.model_validate_json(path.read_text())
        except Exception:
            continue
        run_id = trace.config.experiment_run_id
        if run_id:
            completed.add(str(run_id))
    return completed


def _append_manifest(results_dir: Path, metadata: dict) -> None:
    manifest = _manifest_path(results_dir, str(metadata["experiment_name"]))
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with _manifest_lock:
        with manifest.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metadata, sort_keys=True))
            f.write("\n")


def _s3_uri(bucket: str, *parts: str) -> str:
    key = "/".join(part.strip("/") for part in parts if part)
    return f"s3://{bucket}/{key}"


def _upload_to_s3(local_path: Path, s3_uri: str, profile: str | None) -> None:
    cmd = ["aws", "s3", "cp", str(local_path), s3_uri, "--only-show-errors"]
    if profile:
        cmd.extend(["--profile", profile])
    subprocess.run(cmd, check=True)


def _agent_field(spec: object, field: str) -> object:
    if isinstance(spec, dict):
        return spec.get(field)
    return getattr(spec, field, None)


def _write_metadata_and_upload(
    *,
    ctx: dict,
    trace_path: Path,
    results_dir: Path,
    s3_bucket: str | None,
    s3_prefix: str,
    s3_profile: str | None,
    uploader: str,
) -> None:
    cfg = ctx["config"]
    experiment_name = str(ctx["experiment_name"])
    experiment_run_id = str(cfg.get("experiment_run_id"))
    game_id = trace_path.stem
    trace_s3_uri = None
    metadata_s3_uri = None
    if s3_bucket:
        remote_base = _s3_uri(
            s3_bucket,
            s3_prefix,
            uploader,
            experiment_name,
            experiment_run_id,
        )
        trace_s3_uri = f"{remote_base}/{trace_path.name}"
        metadata_s3_uri = f"{remote_base}/{trace_path.stem}.metadata.json"

    metadata_path = _metadata_path_for(trace_path)
    metadata = {
        "schema_version": 1,
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
        "uploader": uploader,
        "hostname": socket.gethostname(),
        "local_user": getpass.getuser(),
        "git_hash": cfg.get("git_hash") or _git_hash(),
        "experiment_name": experiment_name,
        "experiment_run_id": experiment_run_id,
        "batch_label": ctx["batch_label"],
        "run_idx": ctx["run_idx"],
        "game_name": cfg.get("game_name"),
        "game_id": game_id,
        "seed": cfg.get("seed"),
        "task_path": cfg.get("task_path"),
        "task_id": cfg.get("task_id"),
        "num_agents": cfg.get("num_agents"),
        "agent_types": [_agent_field(spec, "type") for spec in cfg.get("agents", [])],
        "agent_models": [_agent_field(spec, "model") for spec in cfg.get("agents", [])],
        "local_trace_path": str(trace_path),
        "local_metadata_path": str(metadata_path),
        "trace_size_bytes": trace_path.stat().st_size if trace_path.exists() else None,
        "s3": {
            "bucket": s3_bucket,
            "prefix": s3_prefix,
            "trace_uri": trace_s3_uri,
            "metadata_uri": metadata_s3_uri,
            "status": "not_configured" if not s3_bucket else "pending",
            "error": None,
        },
    }

    if s3_bucket and trace_s3_uri and metadata_s3_uri:
        try:
            _upload_to_s3(trace_path, trace_s3_uri, s3_profile)
            metadata["s3"]["status"] = "uploaded"
        except Exception as exc:
            metadata["s3"]["status"] = "failed"
            metadata["s3"]["error"] = f"{type(exc).__name__}: {exc}"
            log.warning("S3 upload failed for %s: %s", experiment_run_id, exc)

    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    if metadata["s3"]["status"] == "uploaded" and metadata_s3_uri:
        try:
            _upload_to_s3(metadata_path, metadata_s3_uri, s3_profile)
        except Exception as exc:
            metadata["s3"]["status"] = "metadata_upload_failed"
            metadata["s3"]["error"] = f"{type(exc).__name__}: {exc}"
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
            log.warning("S3 metadata upload failed for %s: %s", experiment_run_id, exc)

    _append_manifest(results_dir, metadata)


def _check_api_keys(cfg: dict) -> None:
    """Validate that each agent config has a non-empty API key. Raises on first missing key."""
    from a2a_engine.llm.factory import detect_provider, get_api_key_for_provider

    agents = cfg.get("agents", [])
    for i, agent_spec in enumerate(agents):
        spec = agent_spec if isinstance(agent_spec, dict) else agent_spec.model_dump()
        model = spec.get("model", "")
        if not model:
            continue
        if spec.get("api_key"):
            continue
        api_format = str(spec.get("api_format") or "")
        if api_format in {"vertexai", "vertexai_anthropic", "vertexai_openai"}:
            log.info("dry-run: api_format %r uses Google ADC; skipping API key check (agent %d)", api_format, i)
            continue
        provider = detect_provider(model)
        if provider is None:
            log.warning("dry-run: unknown provider for model %r (agent %d), skipping key check", model, i)
            continue
        if provider in {"gemini_vertexai", "claude_vertexai", "vertexai_openai"}:
            log.info("dry-run: provider %r uses Google ADC; skipping API key check (agent %d)", provider, i)
            continue
        key = get_api_key_for_provider(provider)
        if not key:
            raise EnvironmentError(
                f"dry-run: missing API key for provider {provider!r} "
                f"(agent {i}, model {model!r})"
            )
        log.info("dry-run: API key present for provider %r (agent %d)", provider, i)


def _run_one(ctx: dict, results_dir: Path, upload_cfg: dict) -> Path:
    cfg = ctx["config"]
    dry_run = ctx["dry_run"]
    game_name = cfg.get("game_name")
    if not game_name:
        raise ValueError("config is missing 'game_name'")
    GameCls = get_game(game_name)

    if dry_run:
        _check_api_keys(cfg)

    game = GameCls(config=cfg, dry_run=dry_run)

    game_id = str(uuid.uuid4())
    tracer = get_tracer()
    with tracer.start_as_current_span(f"game {game_name}") as span:
        span.set_attribute("gen_ai.conversation.id", game_id)
        span.set_attribute("langfuse.session.id", game_id)
        span.set_attribute(
            "langfuse.trace.tags",
            [ctx["experiment_name"], ctx["batch_label"]],
        )
        token = current_conversation_id.set(game_id)
        try:
            trace = game.run()
        finally:
            current_conversation_id.reset(token)

    if not isinstance(trace, GameTraceBase):
        raise TypeError(
            f"Game {game_name} returned {type(trace).__name__}, expected GameTraceBase"
        )
    trace.game_id = game_id

    if dry_run:
        log.info("dry-run: skipping write_trace for %s", cfg.get("experiment_run_id"))
        return results_dir / f"dry-run-{game_id}.json"

    trace_path = write_trace(trace, results_dir, experiment_name=ctx["experiment_name"])
    try:
        _write_metadata_and_upload(
            ctx=ctx,
            trace_path=trace_path,
            results_dir=results_dir,
            **upload_cfg,
        )
    except Exception as exc:
        log.warning(
            "post-run metadata/upload failed for %s without failing the run: %s",
            cfg.get("experiment_run_id"),
            exc,
        )
    return trace_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an a2a-engine experiment YAML.")
    parser.add_argument("yaml_path", help="Path to experiment YAML file")
    parser.add_argument("--max-parallelism", type=int, default=4)
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--dry-run", action="store_true",
                        help="Hint to game classes to skip LLM calls; game decides what this means.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--resume", action="store_true",
                        help="Skip experiment_run_id values already present in local results metadata/traces.")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="Zero-based shard index to run after expanding the experiment.")
    parser.add_argument("--shard-count", type=int, default=1,
                        help="Total number of shards used to partition expanded runs.")
    parser.add_argument("--s3-bucket", default=os.environ.get("A2A_TRACE_BUCKET"),
                        help="Optional S3 bucket for best-effort streaming trace uploads.")
    parser.add_argument("--s3-prefix", default=os.environ.get("A2A_TRACE_PREFIX", "calendar-traces"),
                        help="S3 key prefix for trace uploads.")
    parser.add_argument("--s3-profile", default=os.environ.get("AWS_PROFILE"),
                        help="AWS CLI profile for S3 uploads.")
    parser.add_argument("--s3-uploader", default=os.environ.get("A2A_TRACE_USER") or os.environ.get("USER") or "unknown",
                        help="Uploader prefix/label written to metadata and S3 keys.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    init_tracing("a2a-engine")

    if args.shard_count < 1:
        parser.error("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")

    spec = load_experiment(args.yaml_path)
    log.info("Loaded experiment %r with %d batches", spec.name, len(spec.batches))
    log.info("Registered games: %s", list_games() or "<none>")

    contexts: list[dict] = []
    results_dir = Path(args.results_dir)
    for batch, resolved in expand_batches(spec):
        for i in range(batch.count):
            contexts.append(_make_run_context(spec.name, batch.label, resolved, i, args.dry_run))

    if args.shard_count > 1:
        before = len(contexts)
        contexts = [
            ctx for idx, ctx in enumerate(contexts)
            if idx % args.shard_count == args.shard_index
        ]
        log.info(
            "Shard enabled: running shard %d/%d with %d of %d expanded runs",
            args.shard_index,
            args.shard_count,
            len(contexts),
            before,
        )

    if args.resume:
        completed = _load_completed_run_ids(results_dir, spec.name)
        before = len(contexts)
        contexts = [
            ctx for ctx in contexts
            if ctx["config"]["experiment_run_id"] not in completed
        ]
        log.info("Resume enabled: skipping %d completed runs, %d remaining",
                 before - len(contexts), len(contexts))

    if not contexts:
        log.warning("No runs to execute.")
        return 0

    upload_cfg = {
        "s3_bucket": args.s3_bucket,
        "s3_prefix": args.s3_prefix,
        "s3_profile": args.s3_profile,
        "uploader": args.s3_uploader,
    }
    if args.s3_bucket:
        log.info(
            "S3 trace upload enabled: bucket=%s prefix=%s uploader=%s profile=%s",
            args.s3_bucket,
            args.s3_prefix,
            args.s3_uploader,
            args.s3_profile or "<default>",
        )

    log.info("Launching %d runs (max_parallelism=%d)", len(contexts), args.max_parallelism)
    results, errors = run_with_parallelism(
        fn=lambda ctx: _run_one(ctx, results_dir, upload_cfg),
        items=contexts,
        max_workers=args.max_parallelism,
        on_result=lambda ctx, p: log.info("ok  %s -> %s", ctx["config"]["experiment_run_id"], p),
        on_error=lambda ctx, e: log.error("fail %s: %s", ctx["config"]["experiment_run_id"], e),
    )
    log.info("Done: %d ok, %d failed", len(results), len(errors))
    shutdown_tracing()
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
