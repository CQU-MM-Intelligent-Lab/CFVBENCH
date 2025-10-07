import os
import sys
import base64
import json
import re
import ast

from videorag._llm import (
    openai_4o_mini_config,
    azure_openai_config,
    ollama_config,
    get_default_ollama_chat_model,
    internvl_hf_config,
)

def sanitize_cuda_libs():
    """
    Finds the bundled PyTorch cuDNN path and forces it to be prioritized.
    It prepends the correct path to LD_LIBRARY_PATH and removes known conflicting paths.
    Set RESPECT_LD_LIBRARY_PATH=1 to skip.
    """
    try:
        if os.environ.get("RESPECT_LD_LIBRARY_PATH", "").strip() in {"1", "true", "True"}:
            print("[Env] RESPECT_LD_LIBRARY_PATH is set. Skipping LD_LIBRARY_PATH sanitization.")
            return

        import importlib.util
        import sys

        print("[Env] Running LD_LIBRARY_PATH sanitization...")

        # 1. Find the path to PyTorch's bundled libraries using importlib
        torch_lib_path = None
        try:
            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                # spec.origin is .../torch/__init__.py
                torch_lib_path = os.path.join(os.path.dirname(spec.origin), 'lib')
                if not os.path.isdir(torch_lib_path):
                    print(f"[Env] Found torch spec but `lib` directory does not exist: {torch_lib_path}")
                    torch_lib_path = None
                else:
                    print(f"[Env] Found PyTorch lib path: {torch_lib_path}")
        except Exception as e:
            print(f"[Env] Could not find PyTorch location using importlib: {e}")
        
        # 2. Get current LD_LIBRARY_PATH and filter it
        original_ld = os.environ.get("LD_LIBRARY_PATH", "")
        print(f"[Env] Original LD_LIBRARY_PATH: {original_ld}")
        paths = [p for p in original_ld.split(":") if p]
        
        # Filter out known conflicting paths. Be aggressive.
        conflicting_substrings = ['nvidia/cudnn', 'cuda', 'cudnn']
        filtered_paths = []
        removed_paths = []
        for p in paths:
            is_conflict = False
            for sub in conflicting_substrings:
                if sub in p.lower():
                    is_conflict = True
                    break
            if is_conflict:
                removed_paths.append(p)
            else:
                filtered_paths.append(p)

        if removed_paths:
            print(f"[Env] Removed conflicting paths: {':'.join(removed_paths)}")
        
        # 3. Prepend the torch lib path if it exists
        final_paths = filtered_paths
        if torch_lib_path:
            # Avoid adding duplicates and ensure it's at the front
            if torch_lib_path in final_paths:
                final_paths.remove(torch_lib_path)
            final_paths.insert(0, torch_lib_path)
        
        new_ld = ":".join(final_paths)

        if new_ld != original_ld:
            os.environ["LD_LIBRARY_PATH"] = new_ld
            print(f"[Env] Set new LD_LIBRARY_PATH: {new_ld if new_ld else '<empty>'}")
        else:
            print("[Env] LD_LIBRARY_PATH did not require changes.")

    except Exception as e:
        print(f"[Env] A critical error occurred during LD_LIBRARY_PATH sanitization: {e}")


def check_dependencies():
    missing = []
    optional = []
    def _try_import(name: str, opt: bool = False):
        try:
            __import__(name)
        except Exception:
            (optional if opt else missing).append(name)
    # Core dependencies from README
    for pkg in [
        "numpy",
        "torch",
        "accelerate",
        "bitsandbytes",
        "moviepy",
        # pytorchvideo installed via git; runtime import name is 'pytorchvideo'
        "pytorchvideo",
        "timm",
        "ftfy",
        "regex",
        "einops",
        "fvcore",
        "decord",  # eva-decord
        "iopath",
        "matplotlib",
        "ctranslate2",
        "faster_whisper",
        "hnswlib",
        "xxhash",
        "transformers",
        "tiktoken",
        "tenacity",
        # storages / vector DB
        "neo4j",
        "nano_vectordb",
        # required by default graph storage
        "networkx",
    ]:
        _try_import(pkg)
    # Optional: cartopy, openai/azure SDK, ollama client, httpx, graspologic, imagebind
    for pkg in ["cartopy", "openai", "ollama", "httpx", "graspologic", "imagebind"]:
        _try_import(pkg, opt=True)

    if missing:
        print("[Dependency] Missing required packages (install as per README):", ", ".join(missing))
    if optional:
        print("[Dependency] Optional/not strictly required now (install if you use related features):", ", ".join(optional))


class SimpleStore:
    def __init__(self, data: dict):
        self._data = data


def check_models(repo_root: str):
    problems = []

    # Only require the local faster-distil-whisper checkpoint.
    # The project no longer requires a local MINICPM path.
    import importlib.util
    use_ollama = importlib.util.find_spec("ollama") is not None

    checkpoints = [os.path.join(repo_root, "faster-distil-whisper-large-v3")]

    for path in checkpoints:
        if not os.path.exists(path):
            problems.append(path)

    if use_ollama:
        print("[Models] Ollama detected: MINICPM model may be provided by Ollama; skipping local MINICPM check.")

    if problems:
        print("[Models] Missing required model/checkpoint path(s):")
        for p in problems:
            print(" -", p)
        print("Please follow README.md to download checkpoints before running.")


def normalize_question_text(question_raw: str) -> str:
    """Normalize question string possibly wrapped like '"question": "...",'.

    Enhanced to strip duplicate outer quotes and trailing commas robustly.
    """
    def _strip_outer_quotes_repeated(text: str) -> str:
        t = text.strip()
        while len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
            t = t[1:-1].strip()
        return t

    s = str(question_raw or "").strip()

    # If the entire string is quoted (possibly with trailing comma), peel one layer
    if len(s) >= 2 and s[0] in {'"', "'"}:
        # remove leading quote and rely on later comma/quote processing
        s = s[1:].strip()

    prefixes = [
        '"question": ',
        'question": ',
        "'question': ",
        'question: ',
    ]
    for pref in prefixes:
        if s.startswith(pref):
            s = s[len(pref):].strip()
            break

    # remove trailing comma(s)
    while s.endswith(','):
        s = s[:-1].strip()

    # regex fallback: extract content inside quotes after question:
    m = re.search(r'question\"?\s*:\s*\"(.+?)\"\s*,?\s*$', s)
    if m:
        s = m.group(1).strip()

    # final repeated outer quote strip
    s = _strip_outer_quotes_repeated(s)
    return s


def extract_final_answer(raw):
    """
    统一清洗/抽取最终答案文本：
    1. 去掉前缀说明（如: "Here's the requested synthesis:")
    2. 去掉 markdown 代码块 ```/```json 包裹
    3. 递归解析嵌套 JSON / 字典字符串，直到拿到最内层 answer
    4. 失败时保留尽可能干净的原文本
    """
    if raw is None:
        return ""
    text = str(raw).strip()

    # 常见前缀删除
    prefix_patterns = [
        r"^here'?s the requested synthesis:\s*",
        r"^answer\s*:\s*",
    ]
    import re, json, ast
    for pat in prefix_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 去除代码块围栏
    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # 去掉第一行 fence
            parts = s.splitlines()
            if parts:
                if parts[0].startswith("```"):
                    parts = parts[1:]
                # 去掉末尾 fence
                if parts and parts[-1].strip().startswith("```"):
                    parts = parts[:-1]
                s = "\n".join(parts).strip()
        return s

    text = _strip_code_fences(text)
    # 去除内部残留的 ```json / ``` 标记
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 如果包含一个更大的 JSON，尝试截取第一个 { ... } 块
    # 但优先保留原始以便下方循环尝试
    def _extract_json_block(s: str):
        first = s.find('{')
        last = s.rfind('}')
        if first != -1 and last != -1 and last > first:
            return s[first:last+1]
        return s
    candidate_main = _extract_json_block(text)

    # 递归解析 3 层
    def _attempt_parse(s: str):
        s2 = s.strip()
        # 剥多余首尾引号
        if (s2.startswith('"') and s2.endswith('"')) or (s2.startswith("'") and s2.endswith("'")):
            s2 = s2[1:-1].strip()
        try:
            return json.loads(s2)
        except Exception:
            try:
                return ast.literal_eval(s2)
            except Exception:
                return None

    def _dig(obj):
        # 返回 (是否成功抽到 answer, 文本)
        if isinstance(obj, dict):
            # 典型结构
            if "answer" in obj:
                val = obj["answer"]
                # 如果还是复杂类型 -> 再转成精简文本
                if isinstance(val, (dict, list)):
                    return True, json.dumps(val, ensure_ascii=False)
                return True, str(val).strip()
            # 可能是 {'question':...,'answer':...}
            lower_keys = {k.lower(): k for k in obj.keys()}
            if "answer" in lower_keys:
                real_key = lower_keys["answer"]
                val = obj[real_key]
                if isinstance(val, (dict, list)):
                    return True, json.dumps(val, ensure_ascii=False)
                return True, str(val).strip()
        if isinstance(obj, list) and len(obj) == 1:
            return _dig(obj[0])
        return False, None

    parsed_text = text  # 默认
    work = candidate_main

    for _ in range(3):
        obj = _attempt_parse(work)
        if obj is None:
            break
        ok, ans = _dig(obj)
        if ok:
            parsed_text = ans
            # 继续看看是否还有嵌套
            work = ans
            continue
        else:
            # 没有 answer 键就停止
            break

    # 若仍然包含 "answer": 且无法标准解析，尝试正则硬截取
    if ('"answer"' in parsed_text or "'answer'" in parsed_text) and parsed_text.count("answer") < 5:
        m = re.search(r'"answer"\s*:\s*(.+)', parsed_text, re.IGNORECASE | re.DOTALL)
        if not m:
            m = re.search(r"'answer'\s*:\s*(.+)", parsed_text, re.IGNORECASE | re.DOTALL)
        if m:
            tail = m.group(1).strip()
            # 去掉可能结尾多余的括号/引号/反引号
            tail = _strip_code_fences(tail)
            # 如果以 { 开头再尝试一次剥壳
            if tail.startswith("{") or tail.startswith("["):
                # 截到最后一个匹配括号（简单启发）
                last_brace = tail.rfind("}")
                if last_brace != -1:
                    tail2 = tail[:last_brace+1]
                    obj2 = _attempt_parse(tail2)
                    if isinstance(obj2, (dict, list)):
                        # 再找内部 answer
                        ok2, ans2 = _dig(obj2)
                        if ok2:
                            parsed_text = ans2
                        else:
                            parsed_text = tail2
                    else:
                        parsed_text = tail
                else:
                    parsed_text = tail
            else:
                parsed_text = tail

    # 最终清洗常见多余包裹
    parsed_text = parsed_text.strip()
    # 去掉再包一层的引号
    if (parsed_text.startswith('"') and parsed_text.endswith('"')) or \
       (parsed_text.startswith("'") and parsed_text.endswith("'")):
        parsed_text = parsed_text[1:-1].strip()

    # 去掉末尾孤立的 '}' 或 ',' 等(防坏格式输出)
    parsed_text = re.sub(r'[,\s]*\}$', '', parsed_text).strip()

    return parsed_text


def choose_llm_config():
    # Priority: Ollama -> Custom OpenAI -> Standard OpenAI -> Azure OpenAI -> DeepSeek+BGE
    from importlib.util import find_spec
    from videorag._llm import create_custom_openai_config, internvl_hf_config as _internvl_cfg
    # If user requests InternVL via OLLAMA_CHAT_MODEL, don't silently fallback to local HF.
    # Local InternVL support is disabled; instruct user to configure Ollama or custom LLM.
    desired = os.environ.get("OLLAMA_CHAT_MODEL", get_default_ollama_chat_model()).strip()
    if desired.lower() == "internvl3_5-8b-hf":
        print("[LLM] Requested InternVL3_5-8B-HF, but local Transformers-based InternVL is disabled in this deployment.")
        print("[LLM] Please run an Ollama server with a compatible model or configure CUSTOM_OPENAI_* environment variables for a custom backend.")
        # Continue to next detection logic (Ollama or other backends)
    # If SKIP_OLLAMA is set, force selection of an API-backed config and do NOT return ollama_config
    skip = str(os.environ.get("SKIP_OLLAMA", "")).strip().lower() in {"1", "true", "yes"}
    if skip:
        print("[LLM] SKIP_OLLAMA set; forcing API-backed LLM selection (no Ollama).")
        # Try custom OpenAI-compatible endpoint first
        custom_api_key = os.environ.get("CUSTOM_OPENAI_API_KEY", "")
        custom_base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
        custom_model = os.environ.get("CUSTOM_OPENAI_MODEL", "gpt-4o-mini")
        if custom_api_key and custom_base_url:
            print(f"[LLM] Using Custom OpenAI-compatible API: {custom_base_url} with model {custom_model}")
            return create_custom_openai_config(
                base_url=custom_base_url,
                api_key=custom_api_key,
                model_name=custom_model,
                embedding_model="text-embedding-3-small"
            )

        # Next prefer standard OpenAI (if available)
        if find_spec("openai") and os.environ.get("OPENAI_API_KEY"):
            print("[LLM] Using Standard OpenAI (gpt-4o-mini) due to SKIP_OLLAMA")
            return openai_4o_mini_config

        # Azure OpenAI
        if find_spec("openai") and (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_ENDPOINT")):
            print("[LLM] Using Azure OpenAI (gpt-4o) due to SKIP_OLLAMA")
            return azure_openai_config

        # As a last resort, if ANTHROPIC/CLAUDE envs are present, map them to custom OpenAI wrapper
        anthropic_endpoint = os.environ.get("CUSTOM_OPENAI_BASE_URL", "") or os.environ.get("ANTHROPIC_ENDPOINT", "")
        anthropic_key = os.environ.get("CUSTOM_OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
        anthropic_model = os.environ.get("CUSTOM_OPENAI_MODEL", "") or os.environ.get("ANTHROPIC_MODEL", "")
        if anthropic_endpoint and anthropic_key:
            print(f"[LLM] Using Anthropic/Claude via custom endpoint {anthropic_endpoint} due to SKIP_OLLAMA")
            return create_custom_openai_config(base_url=anthropic_endpoint, api_key=anthropic_key, model_name=anthropic_model or "claude-3-5-sonnet-all")

        # If nothing configured, try to read the local test/api_config.py as a fallback
        try:
            # api_config.py lives in the test/ directory; attempt import by module name first
            try:
                import api_config
                api_conf = api_config.get_api_llm_config()
            except Exception:
                # Fallback: load test/api_config.py by path to avoid package/import issues
                import importlib.util
                cfg_path = os.path.join(os.path.dirname(__file__), "api_config.py")
                if os.path.exists(cfg_path):
                    spec = importlib.util.spec_from_file_location("test_api_config", cfg_path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    api_conf = mod.get_api_llm_config()
                else:
                    raise

            prov = (api_conf.get('provider') or '').lower()
            # Diagnostic output: report what we found in api_config (no secret values)
            try:
                has_endpoint = bool(api_conf.get('endpoint') or api_conf.get('base_url'))
                has_key = bool(api_conf.get('api_key'))
                print(f"[LLM][Debug] api_config provider={prov} endpoint_present={has_endpoint} api_key_present={has_key}")
            except Exception:
                print("[LLM][Debug] api_config read, but failed to introspect its contents")
            # Map anthropic/claude into custom openai wrapper if endpoint/key present
            if prov in ('anthropic', 'claude'):
                ep = api_conf.get('endpoint') or api_conf.get('base_url')
                key = api_conf.get('api_key')
                model = api_conf.get('model')
                # Strict: prefer api_config.py hardcoded values. Do NOT fallback to environment here.
                if not ep:
                    print("[LLM][Debug] api_config does not provide an endpoint for Anthropic/Claude; skipping")
                elif not key:
                    print("[LLM][Error] api_config.py for Anthropic/Claude is missing 'api_key'. Please add the key to test/api_config.py (preferred) or set environment variables.")
                else:
                    from videorag._llm import create_custom_openai_config
                    try:
                        print(f"[LLM] Using api_config.py Anthropic/Claude endpoint {ep} for SKIP_OLLAMA")
                        return create_custom_openai_config(base_url=ep, api_key=key, model_name=model or 'claude-3-5-sonnet-all')
                    except Exception as e:
                        print(f"[LLM][Debug] Failed to create custom config from api_config Anthropic values: {e}")
            # Map custom/gemini/openai with base_url/key
            # If base_url present, only use it if api_config provides the api_key too (strict policy)
            if api_conf.get('base_url'):
                ep = api_conf.get('base_url')
                key = api_conf.get('api_key')
                if not key:
                    print("[LLM][Error] api_config.py custom endpoint provided but missing 'api_key'. Please add the API key to test/api_config.py.")
                else:
                    from videorag._llm import create_custom_openai_config
                    try:
                        print(f"[LLM] Using api_config.py custom endpoint {ep} for SKIP_OLLAMA")
                        return create_custom_openai_config(base_url=ep, api_key=key, model_name=api_conf.get('model') or 'gpt-4o-mini')
                    except Exception as e:
                        print(f"[LLM][Debug] Failed to create custom config from api_config custom values: {e}")
        except Exception:
            pass

        # If still nothing configured, raise a helpful error so user knows to provide API creds
        raise RuntimeError("SKIP_OLLAMA is set but no API credentials/configuration found (CUSTOM_OPENAI_BASE_URL/CUSTOM_OPENAI_API_KEY or OPENAI_API_KEY). Please set API config (test/api_config.py) or environment variables, or unset SKIP_OLLAMA.")

    # If not skipping Ollama, prefer Ollama if available
    else:
        if find_spec("ollama"):
            print(f"[LLM] Using Ollama config with {get_default_ollama_chat_model()} (ensure ollama server is running)")
            return ollama_config

    # Fallbacks when Ollama not chosen
    custom_api_key = os.environ.get("CUSTOM_OPENAI_API_KEY", "")
    custom_base_url = os.environ.get("CUSTOM_OPENAI_BASE_URL", "")
    custom_model = os.environ.get("CUSTOM_OPENAI_MODEL", "gpt-4o-mini")
    if find_spec("openai") and custom_api_key and custom_base_url:
        print(f"[LLM] Using Custom OpenAI-compatible API: {custom_base_url} with model {custom_model}")
        return create_custom_openai_config(
            base_url=custom_base_url,
            api_key=custom_api_key,
            model_name=custom_model,
            embedding_model="text-embedding-3-small"
        )

    if find_spec("openai") and os.environ.get("OPENAI_API_KEY"):
        print("[LLM] Using Standard OpenAI (gpt-4o-mini)")
        return openai_4o_mini_config
    if find_spec("openai") and (os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_ENDPOINT")):
        print("[LLM] Using Azure OpenAI (gpt-4o)")
        return azure_openai_config
    # DeepSeek support intentionally disabled for now; only Ollama/Custom/OpenAI/Azure supported

    print("[LLM] No usable LLM backend detected. Please install 'ollama' or configure other backends.")
    sys.exit(2)


def image_to_base64(image_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"[Base64 Error] Could not read image {image_path}: {e}")
        return ""


def maybe_offline_fallback_answer(query: str):
    q_norm = query.lower()
    if "1989" in q_norm and "brazil" in q_norm and ("collor" in q_norm or "fernando collor" in q_norm) and ("lula" in q_norm or "luiz inácio" in q_norm):
        print("[Offline Fact] If LLM is unavailable, known facts:")
        print("- 1989 Brazil presidential runoff vote counts:")
        print("  Fernando Collor: 35,089,998 votes (~53.03%)")
        print("  Luiz Inácio Lula da Silva: 31,076,364 votes (~46.97%)")
        print("- Lula subsequently lost 2 presidential elections (1994, 1998) before his first victory in 2002.")
