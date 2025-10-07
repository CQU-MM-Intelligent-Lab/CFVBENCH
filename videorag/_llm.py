import numpy as np
import asyncio
from io import BytesIO
from PIL import Image
import base64
import torch
import re

from dataclasses import asdict, dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

from ._llm_common import (
    AsyncOpenAI,
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    AsyncClient,
    _OPENAI_AVAILABLE,
    _OLLAMA_AVAILABLE,
    get_openai_async_client_instance,
    get_azure_openai_async_client_instance,
    get_custom_openai_async_client_instance,
    get_ollama_async_client_instance,
    LLMConfig,
)

from ._llm_openai import (
    openai_complete_if_cache,
    custom_openai_complete_if_cache,
    custom_openai_embedding,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    custom_gpt_complete,
    openai_embedding,
    openai_config,
    openai_4o_mini_config,
)

# ===== Default Ollama model names (can be overridden via env) =====
# Note: do not cache these values at import time. Read from the environment on each call
# so runtime changes to OLLAMA_CHAT_MODEL / OLLAMA_EMBED_MODEL (e.g. from test.py --api)
# are respected by modules that call these helpers.
def get_default_ollama_chat_model() -> str:
    return os.environ.get("OLLAMA_CHAT_MODEL", "llava-llama3:8b")


def get_default_ollama_embed_model() -> str:
    return os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

## (Common utilities and LLMConfig moved to _llm_common.py)

##### OpenAI Configuration
## (OpenAI related functions & configs moved to _llm_openai.py)

###### Azure OpenAI Configuration
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_complete_if_cache(
    deployment_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    azure_openai_client = get_azure_openai_async_client_instance()
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
        args_hash = compute_args_hash(deployment_name, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'images_base64'}
    # Map internal max_new_tokens to external max_tokens
    if 'max_new_tokens' in filtered_kwargs and 'max_tokens' not in filtered_kwargs:
        filtered_kwargs['max_tokens'] = filtered_kwargs.pop('max_new_tokens')
    response = await azure_openai_client.chat.completions.create(
        model=deployment_name, messages=messages, **filtered_kwargs
    )
    # Normalize response into text to avoid assumptions about shape
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
            {args_hash: {"return": text, "model": deployment_name}}
        )
        await hashing_kv.index_done_callback()
    return text


async def azure_gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def azure_gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await azure_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def azure_openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    azure_openai_client = get_azure_openai_async_client_instance()
    response = await azure_openai_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


azure_openai_config = LLMConfig(
    embedding_func_raw = azure_openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,    
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    best_model_func_raw = azure_gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,

    cheap_model_func_raw  = azure_gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)


######  Ollama configuration

async def ollama_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    # Remove any mtmd* style parameters early to avoid forwarding them to the client
    kwargs = {k: v for k, v in kwargs.items() if not (isinstance(k, str) and k.startswith('mtmd'))}
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # Pop images_base64 to avoid it being part of the hash
    images_base64: list[str] | None = kwargs.pop("images_base64", None)
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    
    # For Ollama, pass images via the message 'images' field rather than concatenating into text
    if images_base64:
        # Ollama expects raw base64 strings (without data:image/... prefix) in some versions
        # Ensure we strip possible prefix if present
        def _strip_prefix(b64: str) -> str:
            if b64.startswith("data:image"):
                try:
                    return b64.split(",", 1)[1]
                except Exception:
                    return b64
            return b64

        user_message = {
            "role": "user",
            "content": prompt,
            "images": [_strip_prefix(img_b64) for img_b64 in images_base64]
        }
    else:
        user_message = {"role": "user", "content": prompt}
    
    messages.append(user_message)

    if hashing_kv is not None:
        # Note: hash does not include images for simplicity, assuming prompt is unique enough
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Send the request to Ollama
    # 统一固定采样参数
    options = {
        "keep_alive": -1,
        "temperature": 0.1,
        "top_p": 1,
    }
    # Merge any user-provided options, but ensure no mtmd* keys
    user_options = kwargs.pop('options', None) or {}
    if isinstance(user_options, dict):
        user_options = {k: v for k, v in user_options.items() if not (isinstance(k, str) and k.startswith('mtmd'))}
        options.update(user_options)
    response = await ollama_client.chat(
        model=model,
        messages=messages,
        options=options
    )
    # print(messages)
    # print(response['message']['content'])

    
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response['message']['content'], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response['message']['content']


async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await ollama_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # Initialize the Ollama client
    ollama_client = get_ollama_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model=model_name,  
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

ollama_config = LLMConfig(
    embedding_func_raw = ollama_embedding,
    embedding_model_name = get_default_ollama_embed_model(),
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = ollama_complete ,
    best_model_name = get_default_ollama_chat_model(), # use Qwen2.5-VL 7B as generator
    best_model_max_token_size = 32768,
    best_model_max_async  = 1,
    cheap_model_func_raw = ollama_mini_complete,
    cheap_model_name = get_default_ollama_chat_model(),
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1
)

# Add a post-init to wrap model functions to accept images_base64
_original_ollama_post_init = ollama_config.__post_init__
def _ollama_post_init_wrapper(self):
    _original_ollama_post_init(self)
    
    original_best_model_func = self.best_model_func
    self.best_model_func = lambda prompt, *args, **kwargs: original_best_model_func(
        prompt, *args, **kwargs
    )

    original_cheap_model_func = self.cheap_model_func
    self.cheap_model_func = lambda prompt, *args, **kwargs: original_cheap_model_func(
        prompt, *args, **kwargs
    )
ollama_config.__post_init__ = _ollama_post_init_wrapper.__get__(ollama_config)


###### DeepSeek Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 使用DeepSeek API
    import httpx
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # DeepSeek API调用
    async with httpx.AsyncClient() as client:
        # Build request json and normalize possible internal param names
        req_json = {
            "model": model,
            "messages": messages,
            # 统一固定采样参数
            "temperature": 0.1,
            "top_p": 1,
        }
        # allow max_new_tokens mapping
        if 'max_new_tokens' in kwargs and 'max_tokens' not in kwargs:
            req_json['max_tokens'] = kwargs.get('max_new_tokens')
        else:
            req_json['max_tokens'] = kwargs.get('max_tokens', 4096)

        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', 'sk-*******')}",
                "Content-Type": "application/json"
            },
            json=req_json,
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()

        # Normalize result
        def _resp_to_text(resp):
            try:
                if resp is None:
                    return ""
                if isinstance(resp, dict):
                    if 'choices' in resp and isinstance(resp['choices'], list) and resp['choices']:
                        c = resp['choices'][0]
                        if isinstance(c, dict) and 'message' in c:
                            return c['message'].get('content','') or c.get('text','') or str(c)
                        return c.get('text','') if isinstance(c, dict) else str(c)
                    if 'return' in resp:
                        return resp['return'] or ""
                return str(resp)
            except Exception:
                return str(resp)

        content = _resp_to_text(result)

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()

    return content

async def deepseek_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # 使用硅基流动的BAAI/bge-m3嵌入模型
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY', 'sk-******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float"
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)

# DeepSeek + BAAI/bge-m3 配置
deepseek_bge_config = LLMConfig(
    embedding_func_raw = bge_m3_embedding,
    embedding_model_name = "BAAI/bge-m3",
    embedding_dim = 1024,  # bge-m3的嵌入维度
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,
    
    best_model_func_raw = deepseek_complete,
    best_model_name = "deepseek-chat",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
    
    cheap_model_func_raw = deepseek_complete,
    cheap_model_name = "deepseek-chat",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

# Custom OpenAI-compatible API configuration
def create_custom_openai_config(base_url: str, api_key: str, model_name: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
    
    async def custom_embedding_wrapper(model_name: str, texts: list[str], **kwargs) -> np.ndarray:
        return await custom_openai_embedding(model_name, texts, base_url, api_key)
    
    async def custom_model_wrapper(model_name_inner, prompt, system_prompt=None, history_messages=[], **kwargs):
        return await custom_gpt_complete(
            model_name_inner, prompt, system_prompt=system_prompt, 
            history_messages=history_messages, base_url=base_url, api_key=api_key, **kwargs
        )
    
    return LLMConfig(
        embedding_func_raw = custom_embedding_wrapper,
        embedding_model_name = embedding_model,
        embedding_dim = 1536,  # text-embedding-3-small dimension
        embedding_max_token_size = 8192,
        embedding_batch_num = 32,
        embedding_func_max_async = 16,
        query_better_than_threshold = 0.2,
        
        best_model_func_raw = custom_model_wrapper,
        best_model_name = model_name,    
        best_model_max_token_size = 32768,
        best_model_max_async = 16,
        
        cheap_model_func_raw = custom_model_wrapper,
        cheap_model_name = model_name,
        cheap_model_max_token_size = 32768,
        cheap_model_max_async = 16
    )

# ==================================================================================
# == InternVL3_5-8B-HF (local Transformers) configuration (no Ollama dependency)
# ==================================================================================

# Globals to cache loaded resources
_internvl_model = None
_internvl_tokenizer = None
_internvl_processor = None

def _get_internvl_model_path() -> str:
    # Prefer explicit path, fallback to HF-style name under cache_dir
    return os.environ.get(
        "INTERNVL_MODEL_PATH",
        " "
    )

def _ensure_internvl_loaded():
    global _internvl_model, _internvl_tokenizer, _internvl_processor
    # Local InternVL models have been deprecated for this project.
    # The codebase now expects Ollama-compatible models. If you need
    # local Transformers-based InternVL, set up your own loader and
    # ensure callers handle its availability. For now, raise a clear error.
    raise RuntimeError(
        "Local InternVL3_5-8B-HF is disabled in this deployment. "
        "Please use an Ollama-hosted model (set OLLAMA_CHAT_MODEL) or configure a custom LLM via environment variables."
    )
    # Local InternVL support intentionally disabled; loader code removed.
    # If you need to enable local InternVL, implement model/tokenizer/processor
    # loading here and return them. For now this function always raises.

async def _internvl_hf_complete_impl(
    prompt: str,
    images: list[str] | None = None,
    **kwargs,
):
    """
    InternVL-Chat-V1.5 completion implementation
    """
    if not prompt:
        return ""

    system_prompt = kwargs.get("system_prompt")
    images_base64: list[str] | None = kwargs.get("images_base64")
    max_new_tokens: int = int(kwargs.get("max_new_tokens", 512))
    temperature: float = float(kwargs.get("temperature", 0.1))
    top_p: float = float(kwargs.get("top_p", 1))

    # 1) 构造 PIL 图像列表（优先使用 images_base64，其次文件路径）
    def _b64_to_pil(b64s: list[str]) -> list[Image.Image]:
        out = []
        for s in b64s:
            try:
                if s.startswith("data:image"):
                    s = s.split(",", 1)[1]
                im = Image.open(BytesIO(base64.b64decode(s))).convert("RGB")
                out.append(im)
            except Exception:
                continue
        return out

    if images_base64:
        images_pil = _b64_to_pil(images_base64)
    else:
        images_pil = []
        if images:
            for p in images:
                try:
                    images_pil.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue

    # 合并文本
    user_text = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"

    # 2) 同步执行体（供 run_in_executor 使用）
    def _run_sync_call(prompt_text: str, images_list: list[Image.Image]) -> str:
        model, tokenizer, proc = _ensure_internvl_loaded()
        final_prompt = ""
        try:
            if proc is None:
                raise ValueError("InternVL processor is not available.")

            # 无论如何都重新构建 prompt，确保格式正确
            n_img = len(images_list)

            if n_img > 0:
                # 清理原有的所有 <image> 标记和 Image-X: 前缀
                cleaned = re.sub(r'Image-\d+:\s*<image>\s*', '', prompt_text)
                cleaned = re.sub(r'<image>', '', cleaned).strip()
                
                # 按照 LMDeploy 的标准格式重新构建
                placeholder_parts = [f'Image-{i+1}: <image>' for i in range(n_img)]
                placeholder_str = '\n'.join(placeholder_parts)
                final_prompt = f"{placeholder_str}\n{cleaned}"
            else:
                # 没有图像时，清理所有图像相关标记
                final_prompt = re.sub(r'Image-\d+:\s*<image>\s*', '', prompt_text)
                final_prompt = re.sub(r'<image>', '', final_prompt).strip()

            # Debug输出，确认占位符与图片数一致
            print(f"[InternVL DEBUG] images_pil count: {n_img}, <image> count in prompt: {final_prompt.count('<image>')}")

            inputs = proc(
                text=final_prompt,
                images=images_list if n_img > 0 else None,
                return_tensors="pt"
            )

            # 尝试将张量迁移到模型设备（若单设备可用）
            try:
                device = getattr(model, "device", None)
                if device is None:
                    device = next(model.parameters()).device  # 可能是 cuda 或 cpu
                moved = {}
                for k, v in inputs.items():
                    moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
                inputs = moved
            except Exception:
                pass

            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(temperature and temperature > 0.0),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            # 解码：若有 input_ids，则裁掉前缀
            try:
                input_len = inputs["input_ids"].shape[1]
                gen_only = output_ids[:, input_len:]
            except Exception:
                gen_only = output_ids

            try:
                text = tokenizer.decode(gen_only[0], skip_special_tokens=True)
            except Exception:
                # 备用解码
                text = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]

            return text.strip()
        except Exception as e:
            # 根据用户要求，移除日志中冗长的prompt打印
            return f"InternVL generate error: {e}"

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        _run_sync_call,
        user_text,
        images_pil
    )
    return response

async def internvl_hf_complete(
    model_name,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs
) -> str:
    # Accept images_base64 passed by caller
    images_base64 = kwargs.pop("images_base64", None)
    # ---- Fixed sampling defaults ----
    max_new = kwargs.pop("max_new_tokens", 512)
    temperature = kwargs.get("temperature", 0.1)
    top_p = kwargs.get("top_p", 1)

    # Delegate to implementation
    return await _internvl_hf_complete_impl(
        prompt=prompt,
        images=None,
        system_prompt=system_prompt,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        images_base64=images_base64,
    )

async def _dummy_embedding(_model_name: str, texts: list[str]) -> np.ndarray:
    # Minimal placeholder to satisfy interface; not used in current pipeline path
    arr = np.zeros((len(texts), 1536), dtype=float)
    return arr
async def _internvl_unavailable(*args, **kwargs):
    raise RuntimeError(
        "Local InternVL3_5-8B-HF support is disabled. Please use an Ollama-hosted model or configure a custom LLM backend."
    )

internvl_hf_config = LLMConfig(
    embedding_func_raw = _dummy_embedding,
    embedding_model_name = "dummy-embeddings",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 8,
    query_better_than_threshold = 0.2,

    best_model_func_raw = _internvl_unavailable,
    best_model_name = "internvl-disabled",
    best_model_max_token_size = 32768,
    best_model_max_async = 2,

    cheap_model_func_raw = _internvl_unavailable,
    cheap_model_name = "internvl-disabled",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 2
)

# ==================================================================================
# == Specific model for Refinement Evaluation
# ==================================================================================
_refiner_client = None

def get_deepseek_r1_refiner_client():
    global _refiner_client
    if _refiner_client is None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI client is not available for Refiner. Please install 'openai'.")
        _refiner_client = AsyncOpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key="sk-X5tCPutvJTOsTpSHnl2bz3IF0EJkFjK22HekxMVUIQQjhNEm"
        )
    return _refiner_client

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_r1_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """
    A dedicated, non-cached function to call DeepSeek-R1 for refinement evaluation.
    It uses a specific client instance and does not interfere with the main LLM config.
    """
    client = get_deepseek_r1_refiner_client()
    messages = [{"role": "user", "content": prompt}]
    
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != 'images_base64'}
    if 'max_new_tokens' in filtered_kwargs and 'max_tokens' not in filtered_kwargs:
        filtered_kwargs['max_tokens'] = filtered_kwargs.pop('max_new_tokens')
    response = await client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=messages,
        # 统一固定采样参数
        temperature=0.1,
        top_p=1,
        max_tokens=512,
        **filtered_kwargs
    )
    # DeepSeek-R1 may return content=None and put text in reasoning_content
    choice = response.choices[0].message
    text = getattr(choice, "content", None)
    if not text:
        text = getattr(choice, "reasoning_content", None)
    return text or ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """
    A dedicated, non-cached function to call an Ollama model for refinement evaluation.
    """
    # For multi-modal models like llava, the normal chat endpoint might hang on text-only prompts.
    # Delegate to a simpler completion function.
    if "llava" in model_name:
        # 仍走通用通道，但已在 ollama_complete_if_cache 中统一固定 temperature/top_p
        return await ollama_complete(model_name, prompt, **kwargs)

    client = get_ollama_async_client_instance()
    messages = [{"role": "user", "content": prompt}]
    
    response = await client.chat(
        model=model_name,
        messages=messages,
        options={
            # 统一固定采样参数
            "temperature": 0.1,
            "top_p": 1,
        },
        keep_alive=-1  # Keep the model loaded in memory
    )
    return response['message']['content']


async def internvl_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """
    A dedicated, non-cached function to call the local InternVL (HF) model
    for refinement evaluation and keyword generation when Ollama is not used.
    """
    # Delegate to the InternVL HF completion with minimal settings
    return await internvl_hf_complete(
        model_name=model_name,
        prompt=prompt,
        system_prompt=kwargs.get("system_prompt")
    )
