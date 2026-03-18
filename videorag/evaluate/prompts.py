from pathlib import Path


def _load_eval_prompt() -> str:
    prompt_path = Path(__file__).resolve().parents[2] / "prompts" / "Evaluation.md"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Evaluation prompt not found: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    if "{test_input}" not in template:
        raise ValueError(f"Evaluation prompt {prompt_path.name} is missing placeholder: {{test_input}}")
    return template


EVAL_PROMPT_TEMPLATE = _load_eval_prompt()
