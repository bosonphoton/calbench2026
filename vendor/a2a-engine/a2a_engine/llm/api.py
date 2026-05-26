"""Unified LLM API helpers.

Centralises payload construction, header building, SSE parsing and response
extraction for OpenAI-compatible, Anthropic, and Vertex AI APIs so every call
site uses the same logic and honours ``model_config.py``.

Game-agnostic: no game-specific prompt formatting lives here.
"""

import asyncio
import json
import logging
import random
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from opentelemetry import trace as _otel_trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from pydantic import BaseModel, Field

from a2a_engine._context import current_conversation_id
from a2a_engine.tracing_otel import get_tracer, should_capture_content

try:
    import certifi
except ImportError:
    certifi = None

try:
    from google import genai as _google_genai
    from google.auth import default as _gcp_default
    from google.auth import load_credentials_from_file as _gcp_load_credentials_from_file
    from google.auth.transport.requests import Request as _GcpRequest
except ImportError:
    _google_genai = None
    _gcp_default = None
    _gcp_load_credentials_from_file = None
    _GcpRequest = None

from a2a_engine.llm.model_config import get_model_config


def _provider_name(model: str) -> str:
    from a2a_engine.llm.factory import detect_provider

    return detect_provider(model) or "unknown"


def _server_address(api_base: str) -> str | None:
    if not api_base:
        return None
    try:
        return urllib.parse.urlparse(api_base).hostname
    except Exception:
        return None


def _messages_to_genai(messages: list[dict]) -> str:
    parts = [
        {"role": m.get("role", "user"),
         "parts": [{"type": "text", "content": m.get("content", "")}]}
        for m in messages
    ]
    return json.dumps(parts)


def _output_to_genai(text: str, finish_reason: str | None) -> str:
    return json.dumps([
        {"role": "assistant",
         "parts": [{"type": "text", "content": text}],
         "finish_reason": finish_reason},
    ])


def _start_chat_span(model: str, api_base: str, messages: list[dict],
                     temperature: float | None, max_tokens: int | None,
                     stream: bool):
    """Start a `chat <model>` CLIENT span and seed required gen_ai attributes."""
    span = get_tracer().start_span(f"chat {model}", kind=SpanKind.CLIENT)
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.provider.name", _provider_name(model))
    span.set_attribute("gen_ai.request.model", model)
    if temperature is not None:
        span.set_attribute("gen_ai.request.temperature", float(temperature))
    if max_tokens is not None:
        span.set_attribute("gen_ai.request.max_tokens", int(max_tokens))
    if stream:
        span.set_attribute("gen_ai.request.stream", True)
    addr = _server_address(api_base)
    if addr:
        span.set_attribute("server.address", addr)
    conv_id = current_conversation_id.get()
    if conv_id:
        span.set_attribute("gen_ai.conversation.id", conv_id)
    if should_capture_content():
        try:
            span.set_attribute("gen_ai.input.messages", _messages_to_genai(messages))
        except Exception:
            pass
    return span


def _finish_chat_span(span, *, response_model: str | None = None,
                      input_tokens: int | None = None,
                      output_tokens: int | None = None,
                      finish_reason: str | None = None,
                      output_text: str | None = None) -> None:
    if response_model:
        span.set_attribute("gen_ai.response.model", response_model)
    if input_tokens is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
    if output_tokens is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
    if finish_reason:
        span.set_attribute("gen_ai.response.finish_reasons", [finish_reason])
    if should_capture_content() and output_text is not None:
        try:
            span.set_attribute("gen_ai.output.messages",
                               _output_to_genai(output_text, finish_reason))
        except Exception:
            pass


def _record_chat_error(span, exc: BaseException) -> None:
    span.set_status(Status(StatusCode.ERROR))
    span.set_attribute("error.type", type(exc).__name__)
    span.record_exception(exc)

log = logging.getLogger("a2a_engine.llm")

_ssl_ctx = ssl.create_default_context(cafile=certifi.where()) if certifi else None


# ---------------------------------------------------------------------------
# LLMClient: thin object wrapper around call_llm_oneshot/streaming
# ---------------------------------------------------------------------------


class LLMClient(BaseModel):
    """Minimal LLM client config object. Use ``oneshot`` / ``streaming`` to call."""

    api_format: str = "openai"
    api_base: str = ""
    api_key: str = ""
    model: str
    temperature: float = 0.7
    max_tokens: int | None = None
    vertex_adc_file: str | None = None
    gcp_project: str | None = None
    gcp_location: str | None = None
    extra: dict = Field(default_factory=dict)

    def oneshot(self, messages: list[dict], **kw) -> str:
        return call_llm_oneshot(
            api_format=self.api_format, api_base=self.api_base, api_key=self.api_key,
            model=self.model, messages=messages,
            max_tokens=kw.pop("max_tokens", self.max_tokens),
            temperature=kw.pop("temperature", self.temperature),
            vertex_adc_file=kw.pop("vertex_adc_file", self.vertex_adc_file),
            gcp_project=kw.pop("gcp_project", self.gcp_project),
            gcp_location=kw.pop("gcp_location", self.gcp_location),
            **kw,
        )

    def streaming(self, messages: list[dict], **kw) -> dict:
        return call_llm_streaming(
            api_format=self.api_format, api_base=self.api_base, api_key=self.api_key,
            model=self.model, messages=messages,
            max_tokens=kw.pop("max_tokens", self.max_tokens) or 4096,
            temperature=kw.pop("temperature", self.temperature),
            vertex_adc_file=kw.pop("vertex_adc_file", self.vertex_adc_file),
            gcp_project=kw.pop("gcp_project", self.gcp_project),
            gcp_location=kw.pop("gcp_location", self.gcp_location),
            **kw,
        )

    def streaming_with_retry(
        self,
        messages: list[dict],
        max_retries: int = 5,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
        **kw,
    ) -> dict | None:
        """streaming() with capped exponential backoff on retryable errors."""
        return call_llm_streaming_with_retry(
            api_format=self.api_format, api_base=self.api_base, api_key=self.api_key,
            model=self.model, messages=messages,
            max_tokens=kw.pop("max_tokens", self.max_tokens) or 4096,
            temperature=kw.pop("temperature", self.temperature),
            vertex_adc_file=kw.pop("vertex_adc_file", self.vertex_adc_file),
            gcp_project=kw.pop("gcp_project", self.gcp_project),
            gcp_location=kw.pop("gcp_location", self.gcp_location),
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            **kw,
        )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def call_llm_streaming(
    api_format: str,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    on_first_chunk=None,
    loop=None,
    thinking_config: dict | None = None,
    logprobs: bool | None = None,
    top_logprobs: int | None = None,
    response_mime_type: str | None = None,
    vertex_adc_file: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    timeout: int = 120,
) -> dict:
    """Streaming LLM call. Returns dict with text, model, duration_s, token usage."""
    span = _start_chat_span(model, api_base, messages, temperature, max_tokens, stream=True)
    try:
        with _otel_trace.use_span(span, end_on_exit=False):
            if api_format == "anthropic":
                result = _stream_anthropic(api_base, api_key, model, messages, max_tokens,
                                           temperature, on_first_chunk, loop, timeout=timeout)
            elif api_format == "vertexai_anthropic":
                result = _stream_vertexai_anthropic(model, messages, max_tokens, temperature,
                                                    on_first_chunk, loop, timeout=timeout,
                                                    adc_file=vertex_adc_file,
                                                    project=gcp_project,
                                                    location=gcp_location)
            elif api_format == "vertexai":
                result = _stream_vertexai(model, messages, max_tokens, temperature,
                                          on_first_chunk, loop,
                                          thinking_config=thinking_config,
                                          logprobs=logprobs, top_logprobs=top_logprobs,
                                          response_mime_type=response_mime_type,
                                          adc_file=vertex_adc_file,
                                          project=gcp_project,
                                          location=gcp_location)
            elif api_format == "vertexai_openai":
                result = _stream_openai(_vertexai_openai_base_url(
                                            project=gcp_project,
                                            location=gcp_location,
                                            adc_file=vertex_adc_file),
                                        _get_vertexai_access_token(vertex_adc_file),
                                        model, messages, max_tokens,
                                        temperature, on_first_chunk, loop,
                                        thinking_config=thinking_config,
                                        logprobs=logprobs, top_logprobs=top_logprobs,
                                        timeout=timeout)
            else:
                result = _stream_openai(api_base, api_key, model, messages, max_tokens,
                                        temperature, on_first_chunk, loop,
                                        thinking_config=thinking_config,
                                        logprobs=logprobs, top_logprobs=top_logprobs,
                                        timeout=timeout)
        _finish_chat_span(
            span,
            response_model=result.get("model"),
            input_tokens=result.get("prompt_tokens"),
            output_tokens=result.get("completion_tokens"),
            finish_reason=result.get("finish_reason"),
            output_text=result.get("text"),
        )
        return result
    except BaseException as exc:
        _record_chat_error(span, exc)
        raise
    finally:
        span.end()


def _is_retryable(exc: Exception) -> bool:
    """Return True if the exception is worth retrying (429, 5xx, timeouts)."""
    status = getattr(exc, 'status_code', None) or getattr(exc, 'code', None)
    if isinstance(status, int):
        if 400 <= status < 500 and status != 429:
            return False
        if status in (429, 500, 502, 503, 504):
            return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    msg = str(exc)
    if "timed out" in msg.lower() or "timeout" in msg.lower():
        return True
    if "429" in msg or "Too Many Requests" in msg:
        return True
    if any(code in msg for code in ("500", "502", "503", "504")):
        return True
    return False


def _get_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from an HTTP error, if present."""
    import urllib.error as _ue
    headers = getattr(exc, 'headers', None)
    if headers is None and isinstance(exc, _ue.HTTPError):
        headers = exc.headers
    if headers is None:
        return None
    val = (headers.get("retry-after") or headers.get("Retry-After"))
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def call_llm_streaming_with_retry(
    *args,
    max_retries: int = 5,
    backoff_base: float = 2.0,
    backoff_max: float = 60.0,
    **kwargs,
) -> dict | None:
    """call_llm_streaming with capped exponential backoff on retryable errors.

    Respects Retry-After headers (Anthropic + OpenAI both send them on 429).
    Returns None after exhausting retries rather than raising, so the caller
    can degrade gracefully.
    """
    last_exc = None
    for attempt in range(max_retries):
        try:
            return call_llm_streaming(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt == max_retries - 1:
                raise
            retry_after = _get_retry_after(exc)
            if retry_after is not None:
                base = min(retry_after, backoff_max)
                jitter = random.uniform(0, min(retry_after * 0.2, 10.0))
            else:
                base = min(backoff_base * (2 ** attempt), backoff_max)
                jitter = random.uniform(0, base * 0.25)
            delay = base + jitter
            log.warning(
                "Retryable error (attempt %d/%d), backing off %.1fs: %s",
                attempt + 1, max_retries, delay, exc,
            )
            time.sleep(delay)
    return None


def call_llm_oneshot(
    api_format: str,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int | None = None,
    temperature: float | None = None,
    timeout: int = 300,
    thinking_budget: int | None = None,
    vertex_adc_file: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
) -> str:
    """Non-streaming LLM call returning text."""
    model_cfg = get_model_config(model)
    effective_max = max_tokens or model_cfg.default_max_tokens

    span = _start_chat_span(model, api_base, messages, temperature, effective_max, stream=False)
    try:
        with _otel_trace.use_span(span, end_on_exit=False):
            if api_format == "anthropic":
                text = _oneshot_anthropic(api_base, api_key, model, messages, effective_max,
                                          temperature, timeout, model_cfg)
            elif api_format == "vertexai_anthropic":
                text, _ = _oneshot_vertexai_anthropic(model, messages, effective_max,
                                                      temperature, timeout, model_cfg,
                                                      adc_file=vertex_adc_file,
                                                      project=gcp_project,
                                                      location=gcp_location)
            elif api_format == "vertexai":
                text, _ = _oneshot_vertexai(model, messages, effective_max, temperature,
                                            thinking_budget,
                                            adc_file=vertex_adc_file,
                                            project=gcp_project,
                                            location=gcp_location)
            elif api_format == "vertexai_openai":
                text = _oneshot_openai(_vertexai_openai_base_url(
                                           project=gcp_project,
                                           location=gcp_location,
                                           adc_file=vertex_adc_file),
                                       _get_vertexai_access_token(vertex_adc_file),
                                       model, messages, effective_max, temperature,
                                       timeout, model_cfg)
            else:
                text = _oneshot_openai(api_base, api_key, model, messages, effective_max,
                                       temperature, timeout, model_cfg)
        _finish_chat_span(span, response_model=model, output_text=text)
        return text
    except BaseException as exc:
        _record_chat_error(span, exc)
        raise
    finally:
        span.end()


# ===================================================================
# Vertex AI helpers
# ===================================================================

def _load_vertex_credentials(adc_file: str | None):
    if adc_file:
        if _gcp_load_credentials_from_file is None:
            raise RuntimeError("google-auth not installed; cannot load Vertex AI ADC file")
        return _gcp_load_credentials_from_file(
            adc_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    if _gcp_default is None:
        raise RuntimeError("google-auth not installed; cannot use Vertex AI ADC")
    return _gcp_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])


def _get_vertexai_client(
    *,
    project: str | None = None,
    location: str | None = None,
    adc_file: str | None = None,
):
    if _google_genai is None:
        raise RuntimeError("google-genai not installed. Run: uv add google-genai")
    credentials, adc_project = _load_vertex_credentials(adc_file)
    from a2a_engine.llm.providers import LLM_PROVIDERS
    cfg = LLM_PROVIDERS.get("gemini_vertexai", {})
    gcp_project = project or cfg.get("gcp_project") or adc_project
    gcp_location = location or cfg.get("gcp_location", "global")
    return _google_genai.Client(
        vertexai=True,
        credentials=credentials,
        project=gcp_project,
        location=gcp_location,
    )


def _get_vertexai_project_location(
    provider_key: str,
    *,
    project: str | None = None,
    location: str | None = None,
    adc_file: str | None = None,
) -> tuple[str, str]:
    _credentials, adc_project = _load_vertex_credentials(adc_file)
    from a2a_engine.llm.providers import LLM_PROVIDERS
    cfg = LLM_PROVIDERS.get(provider_key, {})
    gcp_project = project or cfg.get("gcp_project") or adc_project
    gcp_location = location or cfg.get("gcp_location", "global")
    if not gcp_project:
        raise RuntimeError("No GCP project resolved. Set GOOGLE_CLOUD_PROJECT.")
    return gcp_project, gcp_location


def _vertexai_openai_base_url(
    provider_key: str = "vertexai_openai",
    *,
    project: str | None = None,
    location: str | None = None,
    adc_file: str | None = None,
) -> str:
    project, location = _get_vertexai_project_location(
        provider_key,
        project=project,
        location=location,
        adc_file=adc_file,
    )
    if location == "global":
        host = "https://aiplatform.googleapis.com"
    else:
        host = f"https://{location}-aiplatform.googleapis.com"
    return f"{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi"


def _get_vertexai_access_token(adc_file: str | None = None) -> str:
    if _GcpRequest is None:
        raise RuntimeError("google-auth not installed; cannot use Vertex AI ADC")
    credentials, _ = _load_vertex_credentials(adc_file)
    credentials.refresh(_GcpRequest())
    return credentials.token


def _messages_to_vertexai(messages: list[dict]) -> tuple[str | None, list]:
    system_parts: list[str] = []
    contents = []
    for m in messages:
        role = m["role"]
        text = m["content"]
        if role == "system":
            system_parts.append(text)
        elif role == "assistant":
            contents.append(_google_genai.types.Content(role="model", parts=[_google_genai.types.Part(text=text)]))
        else:
            contents.append(_google_genai.types.Content(role="user", parts=[_google_genai.types.Part(text=text)]))
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


def _messages_to_anthropic_vertexai(messages: list[dict]) -> tuple[str, list[dict]]:
    system_parts: list[str] = []
    api_messages: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            api_messages.append({"role": m["role"], "content": m["content"]})
    return "\n\n".join(system_parts), api_messages


def _anthropic_vertexai_url(project: str, location: str, model: str) -> str:
    host = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    return (
        f"https://{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/anthropic/models/{model}:rawPredict"
    )


def _oneshot_vertexai_anthropic(
    model,
    messages,
    max_tokens,
    temperature,
    timeout,
    model_cfg,
    *,
    adc_file: str | None = None,
    project: str | None = None,
    location: str | None = None,
):
    project, location = _get_vertexai_project_location(
        "claude_vertexai",
        project=project,
        location=location,
        adc_file=adc_file,
    )
    system_text, api_messages = _messages_to_anthropic_vertexai(messages)
    body: dict = {
        "anthropic_version": "vertex-2023-10-16",
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system_text:
        body["system"] = system_text
    if model_cfg.supports_temperature and temperature is not None:
        body["temperature"] = temperature
    url = _anthropic_vertexai_url(project, location, model)
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {_get_vertexai_access_token(adc_file)}",
    }, method="POST")

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            log.error("VERTEX ANTHROPIC API ERROR HTTP %d - %s", e.code, e.read().decode("utf-8", errors="replace"))
        except Exception:
            log.error("VERTEX ANTHROPIC API ERROR HTTP %d", e.code)
        raise
    duration_s = round(time.monotonic() - t0, 3)
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    thinking = "".join(b.get("thinking", "") for b in data.get("content", []) if b.get("type") == "thinking").strip() or None
    log.info("  <- vertexai_anthropic %d chars in %.1fs", len(text), duration_s)
    return text, thinking


def _stream_vertexai_anthropic(
    model,
    messages,
    max_tokens,
    temperature,
    on_first_chunk,
    loop,
    timeout=120,
    *,
    adc_file: str | None = None,
    project: str | None = None,
    location: str | None = None,
):
    model_cfg = get_model_config(model)
    t0 = time.monotonic()
    text, thinking = _oneshot_vertexai_anthropic(
        model, messages, max_tokens, temperature, timeout, model_cfg,
        adc_file=adc_file,
        project=project,
        location=location,
    )
    if on_first_chunk and loop:
        asyncio.run_coroutine_threadsafe(on_first_chunk(), loop).result(timeout=5)
    duration_s = round(time.monotonic() - t0, 3)
    result = {
        "text": text,
        "model": model,
        "duration_s": duration_s,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "reasoning_tokens": None,
        "cached_prompt_tokens": None,
        "finish_reason": None,
    }
    if thinking:
        result["reasoning"] = thinking
    result["_raw_response"] = {k: v for k, v in result.items()}
    return result


def _get_obj_value(obj, *names):
    """Read a field from either a pydantic/genai object or a dict."""
    for name in names:
        if obj is None:
            return None
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _to_plain_value(value):
    """Convert SDK enum-like values to JSON-friendly scalars."""
    if value is None:
        return None
    enum_value = getattr(value, "value", None)
    if enum_value is not None and isinstance(enum_value, (str, int, float, bool)):
        return enum_value
    if isinstance(value, (str, int, float, bool)):
        return value
    enum_name = getattr(value, "name", None)
    if enum_name is not None:
        return enum_name
    return str(value)


def _oneshot_vertexai(
    model,
    messages,
    max_tokens,
    temperature,
    thinking_budget=None,
    *,
    adc_file: str | None = None,
    project: str | None = None,
    location: str | None = None,
):
    client = _get_vertexai_client(project=project, location=location, adc_file=adc_file)
    system_instruction, contents = _messages_to_vertexai(messages)
    cfg = {}
    if max_tokens is not None: cfg["max_output_tokens"] = max_tokens
    if temperature is not None: cfg["temperature"] = temperature
    if system_instruction: cfg["system_instruction"] = system_instruction
    if thinking_budget is not None:
        cfg["thinking_config"] = _google_genai.types.ThinkingConfig(
            include_thoughts=True, thinking_budget=thinking_budget)
    gen_config = _google_genai.types.GenerateContentConfig(**cfg) if cfg else None

    t0 = time.monotonic()
    response = client.models.generate_content(model=model, contents=contents, config=gen_config)
    duration_s = round(time.monotonic() - t0, 3)
    parts = response.candidates[0].content.parts if response.candidates else []
    text = "".join(p.text for p in parts if p.text and not p.thought).strip()
    thinking = "".join(p.text for p in parts if p.text and p.thought).strip() or None
    log.info("  <- vertexai %d chars in %.1fs", len(text), duration_s)
    return text, thinking


def _stream_vertexai(
    model,
    messages,
    max_tokens,
    temperature,
    on_first_chunk=None,
    loop=None,
    thinking_config=None,
    logprobs=None,
    top_logprobs=None,
    response_mime_type=None,
    adc_file: str | None = None,
    project: str | None = None,
    location: str | None = None,
):
    client = _get_vertexai_client(project=project, location=location, adc_file=adc_file)
    system_instruction, contents = _messages_to_vertexai(messages)
    cfg = {}
    if max_tokens is not None: cfg["max_output_tokens"] = max_tokens
    if temperature is not None: cfg["temperature"] = temperature
    if system_instruction: cfg["system_instruction"] = system_instruction
    if response_mime_type is not None:
        cfg["response_mime_type"] = response_mime_type
    if thinking_config is not None:
        cfg["thinking_config"] = thinking_config
    if logprobs is not None:
        cfg["response_logprobs"] = bool(logprobs)
    if top_logprobs is not None:
        cfg["logprobs"] = int(top_logprobs)
    gen_config = _google_genai.types.GenerateContentConfig(**cfg) if cfg else None

    t0 = time.monotonic()
    chunks = []
    logprob_chunks = []
    first = True
    usage_metadata = None
    finish_reason = None
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=gen_config):
        if chunk.text:
            if first:
                first = False
                if on_first_chunk and loop:
                    asyncio.run_coroutine_threadsafe(on_first_chunk(), loop).result(timeout=5)
            chunks.append(chunk.text)
        usage_metadata = _get_obj_value(chunk, "usage_metadata", "usageMetadata") or usage_metadata
        candidates = _get_obj_value(chunk, "candidates") or []
        if candidates:
            finish_reason = (
                _to_plain_value(_get_obj_value(candidates[0], "finish_reason", "finishReason"))
                or finish_reason
            )
            logprobs_result = _get_obj_value(candidates[0], "logprobs_result", "logprobsResult")
            if logprobs_result:
                chosen = _get_obj_value(logprobs_result, "chosen_candidates", "chosenCandidates") or []
                top_by_step = _get_obj_value(logprobs_result, "top_candidates", "topCandidates") or []
                content = []
                for idx, chosen_candidate in enumerate(chosen):
                    top_candidates_obj = top_by_step[idx] if idx < len(top_by_step) else None
                    top_candidates = _get_obj_value(top_candidates_obj, "candidates") or []
                    content.append({
                        "token": _get_obj_value(chosen_candidate, "token"),
                        "logprob": _get_obj_value(chosen_candidate, "log_probability", "logProbability"),
                        "top_logprobs": [
                            {
                                "token": _get_obj_value(candidate, "token"),
                                "logprob": _get_obj_value(candidate, "log_probability", "logProbability"),
                            }
                            for candidate in top_candidates
                        ],
                    })
                if content:
                    logprob_chunks.append({"content": content})
    duration_s = round(time.monotonic() - t0, 3)
    text = "".join(chunks).strip()
    prompt_tokens = _get_obj_value(usage_metadata, "prompt_token_count", "promptTokenCount")
    completion_tokens = _get_obj_value(
        usage_metadata,
        "candidates_token_count",
        "candidatesTokenCount",
        "completion_token_count",
        "completionTokenCount",
    )
    total_tokens = _get_obj_value(usage_metadata, "total_token_count", "totalTokenCount")
    reasoning_tokens = _get_obj_value(
        usage_metadata,
        "thoughts_token_count",
        "thoughtsTokenCount",
    )
    cached_prompt_tokens = _get_obj_value(
        usage_metadata,
        "cached_content_token_count",
        "cachedContentTokenCount",
    )
    result = {"text": text, "model": model, "duration_s": duration_s,
              "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
              "total_tokens": total_tokens, "reasoning_tokens": reasoning_tokens,
              "cached_prompt_tokens": cached_prompt_tokens, "finish_reason": finish_reason}
    if logprob_chunks:
        result["logprobs"] = logprob_chunks
    result["_raw_response"] = {k: v for k, v in result.items()}
    return result


# ===================================================================
# OpenAI-compatible streaming
# ===================================================================

def _stream_openai(api_base, api_key, model, messages, max_tokens,
                   temperature, on_first_chunk, loop, thinking_config=None,
                   logprobs=None, top_logprobs=None, timeout=120):
    model_cfg = get_model_config(model)
    url = f"{api_base.rstrip('/')}/chat/completions"
    log.info("  POST %s model=%s msgs=%d (streaming)", url, model, len(messages))

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        model_cfg.max_tokens_param: max_tokens,
    }
    if model_cfg.supports_temperature:
        payload["temperature"] = temperature
    if logprobs is not None:
        payload["logprobs"] = bool(logprobs)
    if top_logprobs is not None:
        payload["top_logprobs"] = int(top_logprobs)

    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    })
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx)
    except urllib.error.HTTPError as e:
        try:
            log.error("OpenAI API ERROR HTTP %d - %s", e.code, e.read().decode("utf-8", errors="replace"))
        except Exception:
            log.error("OpenAI API ERROR HTTP %d", e.code)
        raise

    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    logprob_chunks: list[dict] = []
    usage: dict = {}
    model_name = model
    first_chunk = True
    finish_reason = None
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            if first_chunk:
                first_chunk = False
                if on_first_chunk and loop:
                    asyncio.run_coroutine_threadsafe(on_first_chunk(), loop).result(timeout=5)
            model_name = chunk.get("model", model_name)
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    chunks.append(content)
                rc = delta.get("reasoning_content")
                if rc:
                    reasoning_chunks.append(rc)
                lp = choices[0].get("logprobs")
                if lp:
                    logprob_chunks.append(lp)
                fr = choices[0].get("finish_reason")
                if fr:
                    finish_reason = fr
    finally:
        resp.close()

    duration_s = round(time.monotonic() - t0, 3)
    text = "".join(chunks).strip()
    reasoning_text = "".join(reasoning_chunks).strip() if reasoning_chunks else None
    completion_details = usage.get("completion_tokens_details", {}) or {}
    prompt_details = usage.get("prompt_tokens_details", {}) or {}
    result = {
        "text": text, "model": model_name, "duration_s": duration_s,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "reasoning_tokens": completion_details.get("reasoning_tokens"),
        "cached_prompt_tokens": prompt_details.get("cached_tokens"),
        "finish_reason": finish_reason,
    }
    if reasoning_text:
        result["reasoning"] = reasoning_text
    if logprob_chunks:
        result["logprobs"] = logprob_chunks
    result["_raw_response"] = {k: v for k, v in result.items()}
    return result


# ===================================================================
# Anthropic streaming
# ===================================================================

def _stream_anthropic(api_base, api_key, model, messages, max_tokens,
                      temperature, on_first_chunk, loop, timeout=120):
    model_cfg = get_model_config(model)
    url = f"{api_base.rstrip('/')}/messages"

    system_text = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            api_messages.append(m)

    cached_idx = None
    if api_messages:
        cached_idx = len(api_messages) - 1
        original_content = api_messages[cached_idx]["content"]
        api_messages[cached_idx] = {**api_messages[cached_idx],
            "content": [{"type": "text", "text": original_content,
                         "cache_control": {"type": "ephemeral"}}]}

    body: dict = {"model": model, "max_tokens": max_tokens, "messages": api_messages, "stream": True}
    if model_cfg.supports_temperature:
        body["temperature"] = temperature
    if system_text:
        body["system"] = [{"type": "text", "text": system_text,
                           "cache_control": {"type": "ephemeral"}}]
    payload = json.dumps(body).encode()
    if cached_idx is not None:
        api_messages[cached_idx]["content"] = original_content

    req = urllib.request.Request(url, data=payload, headers={
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    })
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx)
    except urllib.error.HTTPError as e:
        try:
            log.error("ANTHROPIC API ERROR HTTP %d - %s", e.code, e.read().decode("utf-8", errors="replace"))
        except Exception:
            log.error("ANTHROPIC API ERROR HTTP %d", e.code)
        raise

    chunks: list[str] = []
    thinking_chunks: list[str] = []
    usage: dict = {}
    model_name = model
    first_chunk = True
    current_block_type = None
    stop_reason = None
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            et = event.get("type", "")
            if et == "message_start":
                msg = event.get("message", {})
                model_name = msg.get("model", model_name)
                usage = msg.get("usage", {})
            elif et == "content_block_start":
                current_block_type = event.get("content_block", {}).get("type", "text")
            elif et == "content_block_delta":
                delta = event.get("delta", {})
                t = delta.get("text", "")
                if t:
                    if first_chunk:
                        first_chunk = False
                        if on_first_chunk and loop:
                            asyncio.run_coroutine_threadsafe(on_first_chunk(), loop).result(timeout=5)
                    if current_block_type == "thinking":
                        thinking_chunks.append(t)
                    else:
                        chunks.append(t)
            elif et == "content_block_stop":
                current_block_type = None
            elif et == "message_delta":
                d = event.get("delta", {})
                if d.get("stop_reason"):
                    stop_reason = d["stop_reason"]
                du = event.get("usage", {})
                if du:
                    usage.update(du)
            elif et == "message_stop":
                break
    finally:
        resp.close()

    duration_s = round(time.monotonic() - t0, 3)
    text = "".join(chunks).strip()
    thinking_text = "".join(thinking_chunks).strip() if thinking_chunks else None
    cache_created = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    result = {
        "text": text, "model": model_name, "duration_s": duration_s,
        "prompt_tokens": usage.get("input_tokens"),
        "completion_tokens": usage.get("output_tokens"),
        "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0) or None,
        "reasoning_tokens": None,
        "cached_prompt_tokens": cache_read,
        "cache_creation_input_tokens": cache_created,
        "cache_read_input_tokens": cache_read,
        "finish_reason": stop_reason,
    }
    if thinking_text:
        result["reasoning"] = thinking_text
    result["_raw_response"] = {k: v for k, v in result.items()}
    return result


# ===================================================================
# Oneshot helpers
# ===================================================================

def _oneshot_anthropic(api_base, api_key, model, messages, max_tokens,
                       temperature, timeout, model_cfg):
    url = f"{api_base.rstrip('/')}/messages"
    system_text = ""
    api_messages = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            api_messages.append(m)
    body: dict = {"model": model, "max_tokens": max_tokens, "messages": api_messages}
    if system_text:
        body["system"] = system_text
    if model_cfg.supports_temperature and temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
        data = json.loads(resp.read())
    return "".join(b["text"] for b in data.get("content", []) if b.get("type") == "text").strip()


def _oneshot_openai(api_base, api_key, model, messages, max_tokens,
                    temperature, timeout, model_cfg):
    url = f"{api_base.rstrip('/')}/chat/completions"
    body: dict = {"model": model, model_cfg.max_tokens_param: max_tokens, "messages": messages}
    if model_cfg.supports_temperature and temperature is not None:
        body["temperature"] = temperature
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }, method="POST")
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
        raw = resp.read()
    data = json.loads(raw)
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})
    if message.get("refusal"):
        log.warning("model refused: %s", message["refusal"])
        return ""
    content = message.get("content")
    if not content:
        log.warning("empty content: %s", json.dumps(data)[:500])
        return ""
    return content.strip()
