"""OpenTelemetry GenAI tracing setup. Lazy, opt-in via OTEL_* env vars.

If no OTLP endpoint is configured and no console exporter is requested, we
still install a TracerProvider so spans exist as in-memory objects, but no
exporter is attached and nothing is shipped (effectively zero-overhead).
"""

import json
import os
import threading
from pathlib import Path
from typing import Any, Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanExportResult,
)

_initialized = False
_provider: TracerProvider | None = None


class JsonlSpanExporter(SpanExporter):
    """Write completed spans as one JSON object per line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False

    def export(self, spans: Sequence) -> SpanExportResult:
        if self._closed:
            return SpanExportResult.FAILURE
        try:
            lines = [json.dumps(_span_to_dict(span), default=str) for span in spans]
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line)
                        f.write("\n")
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self._closed = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _hex_trace_id(value: int) -> str:
    return f"{value:032x}"


def _hex_span_id(value: int) -> str:
    return f"{value:016x}"


def _ns_to_iso(ns: int | None) -> str | None:
    if ns is None:
        return None
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc).isoformat()


def _attributes_to_dict(attrs: Any) -> dict[str, Any]:
    return dict(attrs or {})


def _span_to_dict(span) -> dict[str, Any]:
    parent = span.parent
    context = span.context
    return {
        "name": span.name,
        "context": {
            "trace_id": _hex_trace_id(context.trace_id),
            "span_id": _hex_span_id(context.span_id),
            "trace_flags": int(context.trace_flags),
        },
        "parent_span_id": _hex_span_id(parent.span_id) if parent else None,
        "kind": str(span.kind).split(".")[-1],
        "start_time": _ns_to_iso(span.start_time),
        "end_time": _ns_to_iso(span.end_time),
        "attributes": _attributes_to_dict(span.attributes),
        "status": {
            "status_code": str(span.status.status_code).split(".")[-1],
            "description": span.status.description,
        },
        "events": [
            {
                "name": event.name,
                "timestamp": _ns_to_iso(event.timestamp),
                "attributes": _attributes_to_dict(event.attributes),
            }
            for event in span.events
        ],
        "resource": _attributes_to_dict(span.resource.attributes),
        "instrumentation_scope": {
            "name": span.instrumentation_scope.name,
            "version": span.instrumentation_scope.version,
        },
    }


def _should_export() -> bool:
    return bool(
        os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    )


def _should_capture_content() -> bool:
    return os.environ.get("A2A_CAPTURE_CONTENT", "").lower() in ("1", "true", "yes")


def init_tracing(service_name: str = "a2a-engine") -> None:
    """Idempotent: install a TracerProvider; attach exporter only when configured."""
    global _initialized, _provider
    if _initialized:
        return
    _initialized = True

    resource = Resource.create({"service.name": service_name})
    _provider = TracerProvider(resource=resource)

    exporter_env = os.environ.get("OTEL_TRACES_EXPORTER", "").lower()
    local_file = os.environ.get("A2A_OTEL_TRACES_FILE")
    if exporter_env == "file" and not local_file:
        local_file = "results/otel-traces.jsonl"

    if exporter_env == "console":
        _provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif local_file:
        _provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(local_file)))
    elif _should_export():
        # Import here only when actually exporting, so the http exporter is optional.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        _provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

    trace.set_tracer_provider(_provider)


def get_tracer():
    """Return the a2a-engine tracer (no-op until init_tracing has run)."""
    return trace.get_tracer("a2a-engine")


def shutdown_tracing() -> None:
    """Flush and shut down the provider so buffered spans aren't dropped."""
    global _provider, _initialized
    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _initialized = False


def should_capture_content() -> bool:
    return _should_capture_content()
