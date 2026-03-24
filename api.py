"""
api.py — AI call infrastructure. Zero knowledge of domain logic.

All structured AI calls go through structured_ai_call(). Supports:
  - anthropic: direct API via stdlib urllib (no SDK)
  - manual:    write/read files for human-in-the-loop workflows
"""

import json
import os
import random as _random
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

_RETRY_STATUS = {429, 500, 503, 529}



_MAX_RETRIES = 3


def _urlopen_with_backoff(req, timeout: int = 300) -> dict:
    """urllib.request.urlopen with exponential backoff on rate-limit / server errors."""
    for attempt in range(_MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code not in _RETRY_STATUS or attempt == _MAX_RETRIES - 1:
                err = json.loads(e.read().decode())
                raise RuntimeError(
                    f"Anthropic API {e.code}: {err.get('error', {}).get('message', err)}"
                )
            wait = (2 ** attempt) + _random.random()
            print(f"  [api] HTTP {e.code} — retrying in {wait:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
            time.sleep(wait)


def get_ai_backend() -> str:
    """Return the configured AI backend."""
    return os.environ.get("PLAYBOOK_ML_AI_BACKEND", "anthropic").strip().lower() or "anthropic"


def _valid_anthropic_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    return bool(key) and "..." not in key and not key.lower().startswith("your_")


def ai_backend_available() -> bool:
    """Whether an AI backend is available for structured calls."""
    backend = get_ai_backend()
    if backend == "manual":
        return True
    if backend == "anthropic":
        return _valid_anthropic_key()
    return False


def _validate_top_level_required(data: dict, schema: dict, task_name: str):
    """Lightweight validation for manual responses."""
    required = schema.get("required", [])
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Manual {task_name} response missing required keys: {missing}")


def _manual_ai_roundtrip(
    *,
    task_name: str,
    domain_path: Path,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    metadata: dict | None = None,
) -> dict:
    """Write a manual AI request file and wait for a response file."""
    manual_dir = domain_path / "manual_ai"
    requests_dir = manual_dir / "requests"
    responses_dir = manual_dir / "responses"
    requests_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}-{task_name}"
    request_path = requests_dir / f"{request_id}.json"
    response_path = responses_dir / f"{request_id}.json"

    payload = {
        "task": task_name,
        "request_id": request_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "schema": schema,
        "response_path": str(response_path),
        "metadata": metadata or {},
    }
    request_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(
        f"manual-ai: waiting for `{task_name}` response\n"
        f"  request:  {request_path}\n"
        f"  reply to: {response_path}"
    )

    timeout_s = int(os.environ.get("PLAYBOOK_ML_MANUAL_TIMEOUT_SECONDS", "1800"))
    poll_s = float(os.environ.get("PLAYBOOK_ML_MANUAL_POLL_SECONDS", "1.0"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if response_path.exists():
            data = json.loads(response_path.read_text())
            if isinstance(data, dict) and "response" in data and isinstance(data["response"], dict):
                data = data["response"]
            if not isinstance(data, dict):
                raise ValueError(f"Manual {task_name} response must be a JSON object")
            _validate_top_level_required(data, schema, task_name)
            return data
        time.sleep(poll_s)

    raise TimeoutError(f"Timed out waiting for manual {task_name} response at {response_path}")


def structured_ai_call(
    *,
    task_name: str,
    domain_path: Path,
    model: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    schema: dict,
    metadata: dict | None = None,
    thinking: bool = False,
) -> dict:
    """Run a structured AI call via the configured backend."""
    backend = get_ai_backend()
    if backend == "manual":
        return _manual_ai_roundtrip(
            task_name=task_name,
            domain_path=domain_path,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            metadata=metadata,
        )

    if backend == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }
        if thinking:
            budget = max(1024, max_tokens - 4000)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        body = _urlopen_with_backoff(req, timeout=300)
        try:
            text_block = next(b for b in body["content"] if b["type"] == "text")
        except StopIteration:
            raise RuntimeError(f"No text block in Anthropic API response: {body.get('content')}")
        return json.loads(text_block["text"])

    raise RuntimeError(f"Unsupported AI backend: '{backend}'. Set PLAYBOOK_ML_AI_BACKEND=anthropic or manual.")
