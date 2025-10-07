"""Simple API-backed LLM configuration helper.

Place API endpoint, model name, and api key at the top of this file for easy editing.
When `test.py --api` is passed it will try to load the config returned by
`get_api_llm_config()` and use it as `llm_cfg` for the pipeline.

Supported providers (simple wrappers):
- OpenAI-compatible (OpenAI/ChatGPT / Azure / custom base url)
- Gemini-like (uses OpenAI-compatible HTTP interface if provided)
- Claude (via Anthropic-like endpoint if provided)

This file intentionally keeps the editable values at the top for the user.
"""
from typing import Optional
import os
from typing import Tuple, Dict, Any

# ================== USER-EDITABLE SECTION ==================
DEFAULT_PROVIDER = os.environ.get("API_PROVIDER", "anthropic")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_MODEL = os.environ.get("AZURE_OPENAI_MODEL", "gpt-4o")

CUSTOM_OPENAI_BASE_URL = os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
CUSTOM_OPENAI_API_KEY = os.environ.get("CUSTOM_OPENAI_API_KEY", "")
CUSTOM_OPENAI_MODEL = os.environ.get("CUSTOM_OPENAI_MODEL", "gpt-4o-mini")

API_SERVER = os.environ.get("API_SERVER", "api2.aigcbest.top")
API_PATH = os.environ.get("API_PATH", "/v1/messages")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", " ")
ANTHROPIC_ENDPOINT = os.environ.get("ANTHROPIC_ENDPOINT", " ")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", " ")
# ===========================================================

def get_api_llm_config():
    """Return a minimal dict describing which provider to use and credentials.

    This thin wrapper is intentionally simple: `test.py` will map this dict to
    create a compatible LLMConfig via existing helpers in `videorag._llm`.
    """
    provider = (DEFAULT_PROVIDER or "").lower()
    if provider == "openai":
        return {
            "provider": "openai",
            "api_key": OPENAI_API_KEY,
            "model": OPENAI_MODEL,
        }
    if provider == "azure":
        return {
            "provider": "azure",
            "api_key": AZURE_OPENAI_API_KEY,
            "endpoint": AZURE_OPENAI_ENDPOINT,
            "model": AZURE_OPENAI_MODEL,
        }
    if provider == "custom" or provider == "gemini":
        return {
            "provider": "custom",
            "base_url": CUSTOM_OPENAI_BASE_URL,
            "api_key": CUSTOM_OPENAI_API_KEY,
            "model": CUSTOM_OPENAI_MODEL,
        }
    if provider in ("anthropic", "claude"):
        return {
            "provider": "anthropic",
            "api_key": ANTHROPIC_API_KEY,
            "endpoint": ANTHROPIC_ENDPOINT,
            "model": ANTHROPIC_MODEL,
        }
    # default fallback to openai-like
    return {
        "provider": "openai",
        "api_key": OPENAI_API_KEY,
        "model": OPENAI_MODEL,
    }


def _normalize_provider(provider: Optional[str]) -> str:
    """Normalize provider aliases to one of: 'chatgpt', 'gemini', 'claude'."""
    if not provider:
        provider = (DEFAULT_PROVIDER or "").lower()
    p = (provider or "").lower()
    if p in ("openai", "chatgpt", "gpt", "gpt4", "gpt-4", "gpt-4o"):
        return "chatgpt"
    if p in ("custom", "gemini"):
        return "gemini"
    if p in ("anthropic", "claude"):
        return "claude"
    # fallback to chatgpt/openai
    return "chatgpt"


def build_request_for_provider(provider: Optional[str], messages, model: str = None, *, stream: bool = False, max_tokens: int = 1024, path: str = None) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Return (url, headers, json_payload) for the requested provider.

    provider: one of 'gemini', 'claude', 'chatgpt' (aliases allowed)
    messages: list of messages (string or dict as accepted by builders)
    model/stream/max_tokens: provider-specific options
    path: optional override for request path
    """
    p = _normalize_provider(provider)

    # choose model default if not provided
    if not model:
        if p == 'claude':
            model = ANTHROPIC_MODEL
        elif p == 'gemini':
            model = CUSTOM_OPENAI_MODEL or OPENAI_MODEL
        else:
            model = OPENAI_MODEL

    # build payload based on normalized provider
    if p == 'claude':
        payload = build_claude_payload(messages, model=model, max_tokens=max_tokens)
        req_path = path or API_PATH
    elif p == 'gemini':
        payload = build_gemini_payload(messages, model=model, stream=stream)
        # gemini/chat-like endpoints often use chat/completions
        req_path = path or API_PATH or '/v1/chat/completions'
    else:  # chatgpt/openai
        # For OpenAI/ChatGPT-compatible endpoints, transform content blocks into
        # a simple messages array where each message has a string 'content'.
        payload = build_openai_payload(messages, model=model)
        # for OpenAI style chat completion
        req_path = path or '/v1/chat/completions'

    # build URL: for Anthropic/Claude prefer explicit ANTHROPIC_ENDPOINT if provided
    if p == 'claude' and ANTHROPIC_ENDPOINT:
        base = ANTHROPIC_ENDPOINT.rstrip('/')
        url = base + '/' + (req_path or API_PATH).lstrip('/')
    else:
        url = get_api_url(req_path)

    # headers: allow get_api_headers to pick correct key by priority or pass explicit
    headers = get_api_headers()
    return url, headers, payload


def _render_content_blocks_to_text(content_blocks) -> str:
    """Convert a list of content blocks (text/image_url/image) into a single text string.

    Image blocks will be represented as a short marker with the URL when possible.
    """
    out_parts = []
    for cb in content_blocks:
        if not isinstance(cb, dict):
            out_parts.append(str(cb))
            continue
        t = cb.get('type')
        if t == 'text':
            out_parts.append(cb.get('text', ''))
        elif t == 'image_url':
            url = None
            iu = cb.get('image_url')
            if isinstance(iu, dict):
                url = iu.get('url')
            elif isinstance(iu, str):
                url = iu
            if url:
                out_parts.append(f"[image: {url}]")
        elif t == 'image':
            # base64 image — include a placeholder
            out_parts.append('[image: base64]')
        else:
            # fallback: stringify
            out_parts.append(json_safe_str(cb))
    return '\n'.join([p for p in out_parts if p])


def build_openai_payload(messages, model: str = None):
    """Build a simple OpenAI/ChatGPT-compatible payload.

    Converts our internal content-array format into a messages array of
    {'role':..., 'content': '...'} strings. This is more likely to be accepted
    by OpenAI-compatible endpoints that don't support block-structured content.
    """
    model = model or CUSTOM_OPENAI_MODEL or OPENAI_MODEL
    out_messages = []
    for m in messages:
        role = 'user'
        text = ''
        if isinstance(m, str):
            text = m
        elif isinstance(m, dict):
            # respect explicit role if provided
            role = m.get('role', role)
            if 'content' in m and isinstance(m['content'], list):
                text = _render_content_blocks_to_text(m['content'])
            else:
                # fallback: if dict contains text or image_url keys
                if 'text' in m:
                    text = str(m['text'])
                elif 'image_url' in m:
                    text = f"[image: {m.get('image_url')} ]"
                else:
                    text = json_safe_str(m)
        else:
            text = str(m)

        out_messages.append({'role': role, 'content': text})

    return {'model': model, 'messages': out_messages}


def build_chat_payload(messages, model: str = None):
    """Construct a payload matching the HTTP JSON structure in your example.

    `messages` should be a list where each message is either a string or a
    dict with types like {"type": "text", "text": "..."} and/or image_url blocks.
    This helper normalizes into the `content` array form shown in the example.
    """
    model = model or CUSTOM_OPENAI_MODEL or OPENAI_MODEL
    normalized_messages = []
    for m in messages:
        if isinstance(m, str):
            normalized_messages.append({"role": "user", "content": [{"type": "text", "text": m}]})
        elif isinstance(m, dict):
            # assume already in desired shape
            normalized_messages.append(m)
        else:
            # Try to be helpful
            normalized_messages.append({"role": "user", "content": [{"type": "text", "text": str(m)}]})

    return {
        "model": model,
        "messages": normalized_messages,
    }


def get_api_base_url():
    """Return full base url for requests library (with scheme) if possible."""
    # If CUSTOM_OPENAI_BASE_URL is explicitly provided, prefer it (assumed to include scheme)
    if CUSTOM_OPENAI_BASE_URL:
        return CUSTOM_OPENAI_BASE_URL.rstrip('/')

    if not API_SERVER:
        return ""

    s = API_SERVER.strip()
    # If API_SERVER already contains a scheme, use it as-is
    if s.startswith('http://') or s.startswith('https://'):
        return s.rstrip('/')
    # otherwise assume https
    return ("https://" + s).rstrip('/')


def get_api_url(path: str = None) -> str:
    """Return a full request URL combining base url and path.

    If `path` is omitted, use `API_PATH`.
    """
    base = get_api_base_url()
    if not base:
        return ""
    p = (path or API_PATH).lstrip('/')
    return base + '/' + p


def build_gemini_payload(messages, model: str = None, stream: bool = False):
    """Construct a Gemini-style payload aligned with your example.

    This returns a dict with keys: model, stream, messages. Messages follow the
    same `content` array structure used elsewhere (text/image_url blocks).
    """
    payload = build_chat_payload(messages, model=model)
    payload['stream'] = bool(stream)
    # Ensure model name is present
    if not payload.get('model'):
        payload['model'] = model or CUSTOM_OPENAI_MODEL or OPENAI_MODEL
    return payload


def _encode_image_file_to_base64(path: str) -> str:
    """Read image file and return base64 data string (no data: prefix)."""
    import base64
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except Exception:
        return ""


def build_claude_payload(messages, model: str = None, max_tokens: int = 1024):
    """Construct a Claude-style payload matching your second example.

    Each item in `messages` can be:
    - a plain string -> converted to a text content block
    - a dict representing a prepared message with 'content' field
    - a dict with keys like {'image_path': '/tmp/img.png'} or {'image_base64': '...'}

    The returned dict follows the pattern:
      {"model": ..., "max_tokens": ..., "messages": [ {"role":"user","content":[ ... ]} ] }
    """
    model = model or CUSTOM_OPENAI_MODEL or ANTHROPIC_MODEL
    normalized_messages = []
    for m in messages:
        content_blocks = []
        if isinstance(m, str):
            content_blocks.append({"type": "text", "text": m})
        elif isinstance(m, dict):
            # If user passed an explicit 'content' array already, use it
            if 'content' in m and isinstance(m['content'], list):
                content_blocks = m['content']
            elif 'image_base64' in m or 'image_path' in m:
                img_b64 = m.get('image_base64') or _encode_image_file_to_base64(m.get('image_path'))
                # default to PNG media type if not provided
                media_type = m.get('media_type', 'image/png')
                if img_b64:
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": img_b64
                        }
                    })
                # optionally include a text if present
                if 'text' in m:
                    content_blocks.insert(0, {"type": "text", "text": str(m['text'])})
            elif 'image_url' in m:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {"url": m['image_url']}
                })
            else:
                # fallback: try to stringify dict
                content_blocks.append({"type": "text", "text": json_safe_str(m)})
        else:
            content_blocks.append({"type": "text", "text": str(m)})

        normalized_messages.append({"role": "user", "content": content_blocks})

    return {"model": model, "max_tokens": max_tokens, "messages": normalized_messages}


def json_safe_str(obj) -> str:
    """Return a JSON-safe short string representation of obj."""
    try:
        import json
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def get_api_headers(api_key: str = None):
    """Return default headers for API requests (Bearer auth)."""
    key = api_key or CUSTOM_OPENAI_API_KEY or OPENAI_API_KEY or ANTHROPIC_API_KEY
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
    }
    if key:
        headers['Authorization'] = 'Bearer ' + key
    return headers
