"""CLI: cache Langfuse traces locally as GameTraceBase JSON files.

Example:
    uv run python scripts/cache_langfuse.py --output-dir ./langfuse_cache \
        --tag experiment=word_guess_v1 --since 2026-04-01T00:00:00 --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

from a2a_engine.langfuse_cache import fetch_and_convert, using_sdk


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", default="./langfuse_cache")
    ap.add_argument("--session-id", action="append", default=[], help="Repeatable.")
    ap.add_argument("--tag", action="append", default=[], help="Repeatable.")
    ap.add_argument("--since", default=None, help="ISO timestamp.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[cache_langfuse] using_sdk={using_sdk()}")
    traces = fetch_and_convert(
        session_ids=args.session_id or None,
        tags=args.tag or None,
        from_timestamp=args.since,
        limit=args.limit,
    )
    print(f"[cache_langfuse] fetched {len(traces)} trace(s)")

    for t in traces:
        path = out / f"{t.game_id}.json"
        path.write_text(t.model_dump_json(indent=2))
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
