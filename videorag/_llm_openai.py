import numpy as np
import asyncio
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
try:
    import httpx
    _HTTPX_AVAILABLE = True
except Exception:
    httpx = None
    _HTTPX_AVAILABLE = False

from .base import BaseKVStorage
from ._utils import compute_args_hash
from ._llm_common import (
    get_openai_async_client_instance,
    get_custom_openai_async_client_instance,
    APIConnectionError,
    RateLimitError,
    LLMConfig,
)

# ================= OpenAI & Custom OpenAI =================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Remove any image payloads from kwargs to avoid passing unexpected keywords
    kwargs.pop("images_base64", None)
    # Map internal max_new_tokens -> external max_tokens if present
    if 'max_new_tokens' in kwargs and 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )
    def _resp_to_text(resp):
        try:
            if resp is None:
                return ""
            if hasattr(resp, 'choices'):
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
            if isinstance(resp, dict):
                if 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
                    c = resp['choices'][0]
                    if isinstance(c, dict) and 'message' in c:
                        return c['message'].get('content','') or c.get('text','') or str(c)
                    return c.get('text','') if isinstance(c, dict) else str(c)
                if 'message' in resp and isinstance(resp['message'], dict):
                    return resp['message'].get('content','')
                if 'return' in resp:
                    return resp['return'] or ""
                return str(resp)
            return str(resp)
        except Exception:
            return str(resp)

    text = _resp_to_text(response)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": text, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return text


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def custom_openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], base_url=None, api_key=None, **kwargs
) -> str:
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return.get("return") is not None:
            return if_cache_return["return"]

    # Remove image-specific kwarg to avoid passing unknown params to custom client
    images_base64 = kwargs.pop("images_base64", None)
    # Map internal max_new_tokens -> external max_tokens if present
    if 'max_new_tokens' in kwargs and 'max_tokens' not in kwargs:
        kwargs['max_tokens'] = kwargs.pop('max_new_tokens')

    # Try a direct HTTP call using test/api_config.py if present (mirrors your example)
    http_attempted = False
    try:
        import api_config
        # Determine provider based on model name
        model_lower = (model or "").lower()
        if "claude" in model_lower:
            provider = "custom"  # Use OpenAI-compatible endpoint for Claude via proxy
            base_url = api_config.CUSTOM_OPENAI_BASE_URL
            path = ""  # CUSTOM_OPENAI_BASE_URL already includes path
        else:
            provider = "custom"
            base_url = api_config.CUSTOM_OPENAI_BASE_URL
            path = ""  # CUSTOM_OPENAI_BASE_URL already includes path
        
        if base_url:
            api_url = base_url.rstrip('/') + path
        else:
            api_url = api_config.get_api_url()
        
        api_headers = api_config.get_api_headers(api_key)
        if api_url:
            http_attempted = True
            # Build payload based on provider
            # Prepare messages with images if present
            if images_base64:
                content = [{"type": "text", "text": prompt}]
                for b64 in images_base64:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    })
                message = {"role": "user", "content": content}
            else:
                message = {"role": "user", "content": prompt}
            
            try:
                payload = api_config.build_openai_payload([message], model=model)
            except Exception:
                payload = {"model": model, "messages": [message]}

            if _HTTPX_AVAILABLE:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    r = await client.post(api_url, headers=api_headers or {}, json=payload)
                    status = r.status_code
                    ctype = r.headers.get('content-type', '')
                    body_text = r.text
            else:
                import requests
                def _sync_post():
                    rr = requests.post(api_url, headers=api_headers or {}, json=payload, timeout=30)
                    return rr.status_code, rr.headers.get('content-type',''), rr.text
                status, ctype, body_text = await asyncio.to_thread(_sync_post)

            preview = (body_text or '')[:2000]
            if (isinstance(ctype, str) and 'html' in ctype.lower()) or preview.lstrip().lower().startswith('<!doctype') or preview.lstrip().lower().startswith('<html'):
                print(f"[HTTP-Fallback] POST {api_url} returned HTML (status={status}, content-type={ctype}). Preview:\n{preview[:1000]}")
                raise RuntimeError(f"API at {api_url} returned HTML (not JSON). Check endpoint/path/headers.")

            # parse JSON if possible
            try:
                jr = json.loads(body_text)
            except Exception:
                jr = None

            if isinstance(jr, dict):
                # common patterns
                if 'answer' in jr:
                    resp_text = jr.get('answer') or ''
                elif 'choices' in jr and isinstance(jr['choices'], list) and jr['choices']:
                    c = jr['choices'][0]
                    if isinstance(c, dict) and 'message' in c:
                        resp_text = c['message'].get('content','') or c.get('text','') or str(c)
                    else:
                        resp_text = c.get('text','') if isinstance(c, dict) else str(c)
                else:
                    resp_text = json.dumps(jr, ensure_ascii=False)
            else:
                resp_text = body_text

            if hashing_kv is not None:
                await hashing_kv.upsert({args_hash: {"return": resp_text, "model": model}})
                await hashing_kv.index_done_callback()
            return resp_text
    except Exception as e:
        if http_attempted:
            print(f"[HTTP-Fallback][Debug] HTTP attempt failed: {e}")
        # fall back to AsyncOpenAI-compatible client below

    # Fallback to AsyncOpenAI-compatible client
    openai_async_client = get_custom_openai_async_client_instance(base_url, api_key)
    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    def _resp_to_text(resp):
        try:
            if resp is None:
                return ""
            if hasattr(resp, 'choices'):
                try:
                    return resp.choices[0].message.content
                except Exception:
                    return str(resp)
            if isinstance(resp, dict):
                if 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
                    c = resp['choices'][0]
                    if isinstance(c, dict) and 'message' in c:
                        return c['message'].get('content','') or c.get('text','') or str(c)
                    return c.get('text','') if isinstance(c, dict) else str(c)
                if 'message' in resp and isinstance(resp['message'], dict):
                    return resp['message'].get('content','')
                if 'return' in resp:
                    return resp['return'] or ""
                return str(resp)
            return str(resp)
        except Exception:
            return str(resp)

    text = _resp_to_text(response)

    if hashing_kv is not None:
        await hashing_kv.upsert({args_hash: {"return": text, "model": model}})
        await hashing_kv.index_done_callback()
    return text


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def custom_openai_embedding(model_name: str, texts: list[str], base_url: str, api_key: str) -> np.ndarray:
    openai_async_client = get_custom_openai_async_client_instance(base_url, api_key)
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def custom_gpt_complete(
        model_name, prompt, system_prompt=None, history_messages=[], base_url=None, api_key=None, **kwargs
) -> str:
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await custom_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


openai_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)
