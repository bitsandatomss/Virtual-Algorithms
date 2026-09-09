from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
import random
from typing import Any

from rlraft.env import load_env
from rlraft.sim.sim import TIMEOUT_ARMS, NodeObservation


# ---------------------------------------------------------------------------
# Module-level state (thread-safe via lock)
# ---------------------------------------------------------------------------

LAST_NODE_LLM_ERROR: str | None = None
LAST_NODE_LLM_KEY_SOURCE: str | None = None

_state_lock = threading.Lock()

# Tracks which keys are rate-limited and when they reset (epoch seconds).
# Key: env var name, Value: reset timestamp (0 = not rate-limited).
_RATE_LIMITED_UNTIL: dict[str, float] = {}


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

# Priority order: Groq (faster, lower latency) then OpenRouter (broader model
# access). Within each provider keys are tried in order 1 → 2 → 3.
_KEY_ORDER = [
    ("DEEPSEEK_API_KEY",     "deepseek"),
    ("GROQ_API_KEY",         "groq"),
    ("GROQ_API_KEY_2",       "groq"),
    ("GROQ_API_KEY_3",       "groq"),
    ("OPENROUTER_API_KEY",   "openrouter"),
    ("OPENROUTER_API_KEY_2", "openrouter"),
    ("OPENROUTER_API_KEY_3", "openrouter"),
]

_GROQ_ENDPOINT       = "https://api.groq.com/openai/v1/chat/completions"
_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
_DEEPSEEK_ENDPOINT   = "https://api.deepseek.com/chat/completions"

# Model selection per provider (both support the OpenAI chat completions format)
_GROQ_MODEL       = "llama-3.1-8b-instant"
_OPENROUTER_MODEL = "meta-llama/llama-3.2-3b-instruct:free"
_DEEPSEEK_MODEL   = "deepseek-chat"

_SYSTEM_PROMPT = (
    "You are one Raft consensus node choosing only your own election timeout. "
    "Fast, reliable, fresh-log nodes should choose shorter timeouts so they "
    "start elections before slow peers. Slow, lossy, stale-log, or split-prone "
    "nodes must back off with longer timeouts. "
    "Return ONLY a JSON object with exactly one field: 'action'."
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class LLMNodeDecision:
    source: str        # "groq", "openrouter", "llm_cache", "deterministic_fallback"
    action: str        # one of TIMEOUT_ARMS keys
    timeout_ms: float
    notes: str = ""


# ---------------------------------------------------------------------------
# Public client classes
# ---------------------------------------------------------------------------

class LLMNodePolicyClient:
    """Direct LLM timeout policy for Raft nodes.

    Tries all 6 API keys (Groq first, then OpenRouter) in order. When a key
    returns HTTP 429 (rate limited) it is skipped and the next is tried. Keys
    are re-enabled after their Retry-After window expires. If all keys are
    exhausted the client falls back to a deterministic heuristic.

    Decisions are cached by observation bucket so large experiments do not
    spend one API call per election timer reset.
    """

    def __init__(self, require_llm: bool = False):
        self.require_llm = require_llm
        self.cache: dict[str, str] = {}

    def decide_from_sim_observation(
        self,
        observation: NodeObservation,
        rng: random.Random,
    ) -> LLMNodeDecision:
        payload = {
            "node_id": observation.node_id,
            "mean_rtt_ms": observation.mean_rtt_ms,
            "loss_rate": observation.loss_rate,
            "log_gap": observation.log_gap,
            "recent_split": observation.recent_split,
        }
        return self.decide(payload, rng)

    def decide(self, observation: dict[str, Any], rng: random.Random) -> LLMNodeDecision:
        key = _observation_key(observation)
        cached_action = self.cache.get(key)
        if cached_action is not None:
            low, high = TIMEOUT_ARMS[cached_action]
            return LLMNodeDecision(
                source="llm_cache",
                action=cached_action,
                timeout_ms=rng.uniform(float(low), float(high)),
                notes="cached_by_observation_bucket",
            )

        result = _request_llm_action(observation)
        if result is not None:
            action, source = result
            self.cache[key] = action
            low, high = TIMEOUT_ARMS[action]
            return LLMNodeDecision(
                source=source,
                action=action,
                timeout_ms=rng.uniform(float(low), float(high)),
                notes="live_llm_decision",
            )

        # All keys exhausted or unavailable
        if self.require_llm:
            with _state_lock:
                err = LAST_NODE_LLM_ERROR
            raise RuntimeError(
                f"LLM node policy required but unavailable: {err}"
            )

        action = _fallback_action(observation)
        low, high = TIMEOUT_ARMS[action]
        return LLMNodeDecision(
            source="deterministic_fallback",
            action=action,
            timeout_ms=rng.uniform(float(low), float(high)),
            notes="all_keys_exhausted_or_rate_limited",
        )


class LLMNodeTimeoutPolicy:
    def __init__(self, require_llm: bool = False):
        self.client = LLMNodePolicyClient(require_llm=require_llm)

    def timeout_ms(self, observation: NodeObservation, rng: random.Random) -> float:
        return self.client.decide_from_sim_observation(observation, rng).timeout_ms


# ---------------------------------------------------------------------------
# Diagnostic helper
# ---------------------------------------------------------------------------

def check_llm_node_policy() -> dict[str, Any]:
    rng = random.Random(1)
    client = LLMNodePolicyClient(require_llm=False)
    decision = client.decide(
        {
            "node_id": 0,
            "mean_rtt_ms": 42.0,
            "loss_rate": 0.01,
            "log_gap": 0,
            "recent_split": False,
        },
        rng,
    )
    with _state_lock:
        key_source = LAST_NODE_LLM_KEY_SOURCE
        llm_error = LAST_NODE_LLM_ERROR
    return _scrub_secrets({
        "source": decision.source,
        "action": decision.action,
        "timeout_ms": decision.timeout_ms,
        "llm_key_source": key_source,
        "llm_error": llm_error,
        "rate_limited_keys": [
            k for k, t in _RATE_LIMITED_UNTIL.items() if t > time.time()
        ],
    })


def _scrub_secrets(report: dict[str, Any]) -> dict[str, Any]:
    """Redact any loaded API-key values from a printable status report.

    Only key *names* may ever reach terminal output. Provider error bodies
    are untrusted strings, so any known secret value found inside them is
    replaced in memory before the report is returned. Values shorter than
    8 chars are skipped to avoid scrubbing ordinary words.
    """
    secrets = []
    seen: set[str] = set()
    # _KEY_ORDER names plus any other *_API_KEY variable present: belt and
    # braces, so a key the router doesn't even know about still can't leak
    # through an echoed error body.
    names = [name for name, _ in _KEY_ORDER]
    names += [k for k in os.environ
              if k.endswith("_API_KEY") or "_API_KEY_" in k]
    for key_name in names:
        api_key = os.environ.get(key_name, "").strip()
        if api_key and len(api_key) >= 8 and api_key not in seen:
            seen.add(api_key)
            secrets.append(api_key)
    if not secrets:
        return report

    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for secret in secrets:
                if secret in value:
                    value = value.replace(secret, "***")
            return value
        if isinstance(value, list):
            return [scrub(v) for v in value]
        if isinstance(value, dict):
            return {k: scrub(v) for k, v in value.items()}
        return value

    return scrub(report)


# ---------------------------------------------------------------------------
# Core multi-provider request logic
# ---------------------------------------------------------------------------

def _request_llm_action(observation: dict[str, Any]) -> tuple[str, str] | None:
    """Try each of the 6 API keys in priority order.

    Returns (action, source_label) on success, or None if all keys fail.
    On rate limit (429) the key is skipped. On auth error (401/403) the key
    is also skipped. On network errors the remaining keys are still tried.
    """
    global LAST_NODE_LLM_ERROR, LAST_NODE_LLM_KEY_SOURCE
    load_env()

    available_keys = _available_keys()
    if not available_keys:
        with _state_lock:
            LAST_NODE_LLM_ERROR = "all_keys_missing_or_rate_limited"
        return None

    errors: list[str] = []
    now = time.time()

    for key_name, provider, api_key in available_keys:
        # Double-check rate limit window hasn't expired since we built the list
        with _state_lock:
            reset_at = _RATE_LIMITED_UNTIL.get(key_name, 0.0)
        if reset_at > now:
            errors.append(f"{key_name}:still_rate_limited_until_{int(reset_at)}")
            continue

        result = _call_provider(key_name, provider, api_key, observation)
        if result is None:
            # Error was recorded inside _call_provider; check _last_error
            with _state_lock:
                errors.append(f"{key_name}:{LAST_NODE_LLM_ERROR or 'unknown'}")
            continue

        action, source = result
        if action in TIMEOUT_ARMS:
            with _state_lock:
                LAST_NODE_LLM_KEY_SOURCE = key_name
                LAST_NODE_LLM_ERROR = None
            return action, source

        errors.append(f"{key_name}:invalid_action:{action}")

    with _state_lock:
        LAST_NODE_LLM_ERROR = " | ".join(errors) if errors else "unknown_api_failure"
    return None


def _available_keys() -> list[tuple[str, str, str]]:
    """Return list of (key_name, provider, api_key) for keys that are present
    and not currently rate-limited."""
    now = time.time()
    result = []
    for key_name, provider in _KEY_ORDER:
        api_key = os.environ.get(key_name, "").strip()
        if not api_key:
            continue
        with _state_lock:
            reset_at = _RATE_LIMITED_UNTIL.get(key_name, 0.0)
        if reset_at > now:
            continue
        result.append((key_name, provider, api_key))
    return result


def _call_provider(
    key_name: str,
    provider: str,
    api_key: str,
    observation: dict[str, Any],
) -> tuple[str, str] | None:
    """Call a single API key. Returns (raw_action_str, source_label) or None."""
    if provider == "deepseek":
        return _call_chat_completions(
            key_name=key_name,
            api_key=api_key,
            endpoint=_DEEPSEEK_ENDPOINT,
            model=_DEEPSEEK_MODEL,
            source_label="deepseek",
            observation=observation,
            extra_headers=None,
        )
    elif provider == "groq":
        return _call_chat_completions(
            key_name=key_name,
            api_key=api_key,
            endpoint=_GROQ_ENDPOINT,
            model=_GROQ_MODEL,
            source_label="groq",
            observation=observation,
            extra_headers=None,
        )
    elif provider == "openrouter":
        return _call_chat_completions(
            key_name=key_name,
            api_key=api_key,
            endpoint=_OPENROUTER_ENDPOINT,
            model=_OPENROUTER_MODEL,
            source_label="openrouter",
            observation=observation,
            extra_headers={
                "HTTP-Referer": "https://github.com/dwan-ith/RL-Raft",
                "X-Title": "RL-Raft MARL Research",
            },
        )
    return None


def _call_chat_completions(
    key_name: str,
    api_key: str,
    endpoint: str,
    model: str,
    source_label: str,
    observation: dict[str, Any],
    extra_headers: dict[str, str] | None,
) -> tuple[str, str] | None:
    """Generic OpenAI-compatible chat completions call."""
    global LAST_NODE_LLM_ERROR
    user_content = json.dumps(
        {
            "local_observation": observation,
            "valid_actions": list(TIMEOUT_ARMS),
            "schema": {"action": "one of valid_actions"},
            "instruction": (
                "Pick the action that best matches how this node should behave. "
                "Return ONLY {\"action\": \"<chosen_action>\"}."
            ),
        },
        indent=2,
    )
    body = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 64,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RL-Raft-Client/1.0",
    }
    if extra_headers:
        headers.update(extra_headers)

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=4) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = _extract_chat_text(payload)
        parsed = json.loads(_extract_json_object(text))
        action = str(parsed.get("action", "")).strip()
        return action, source_label

    except urllib.error.HTTPError as exc:
        body_text = _safe_error_body(exc)
        if exc.code == 429:
            # Parse Retry-After header if present
            retry_after = _parse_retry_after(exc, body_text)
            with _state_lock:
                _RATE_LIMITED_UNTIL[key_name] = time.time() + retry_after
                LAST_NODE_LLM_ERROR = f"rate_limited_retry_after_{retry_after}s"
        elif exc.code in {401, 403}:
            # Bad key — mark as permanently unavailable for this session
            with _state_lock:
                _RATE_LIMITED_UNTIL[key_name] = time.time() + 86400.0
                LAST_NODE_LLM_ERROR = f"auth_error_{exc.code}"
        else:
            with _state_lock:
                LAST_NODE_LLM_ERROR = f"http_{exc.code}:{body_text[:200]}"
        return None

    except Exception as exc:
        with _state_lock:
            # Back off this key for 30s to prevent repeated timeouts on every cache miss
            _RATE_LIMITED_UNTIL[key_name] = time.time() + 30.0
            LAST_NODE_LLM_ERROR = f"network_or_parse_error:{type(exc).__name__}:{exc}"
        return None

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        with _state_lock:
            LAST_NODE_LLM_ERROR = f"parse_error:{exc}"
        return None


def _parse_retry_after(exc: urllib.error.HTTPError, body_text: str) -> float:
    """Extract retry-after seconds from headers or body. Default 60s."""
    try:
        header = exc.headers.get("Retry-After", "")
        if header:
            return max(5.0, float(header))
    except (ValueError, AttributeError):
        pass
    # Try to find a number in the body like "retry after 30 seconds"
    match = re.search(r"retry.{0,20}?(\d+)", body_text, re.IGNORECASE)
    if match:
        return max(5.0, float(match.group(1)))
    return 60.0


# ---------------------------------------------------------------------------
# Deterministic fallback heuristic
# ---------------------------------------------------------------------------

def _fallback_action(observation: dict[str, Any]) -> str:
    """Map local observation to a timeout arm without any API call.

    Uses the same scoring formula as the observation_quality_score in sim.py
    so the fallback is internally consistent with the reward signal.
    """
    rtt = min(float(observation.get("mean_rtt_ms", observation.get("estimated_rtt_ms", 200.0))) / 350.0, 1.5)
    loss = min(float(observation.get("loss_rate", 0.0)) / 0.22, 1.5)
    log_gap = min(float(observation.get("log_gap", 0.0)) / 6.0, 1.0)
    score = 0.62 * rtt + 0.28 * loss + 0.10 * log_gap
    if observation.get("recent_split") or observation.get("recent_split_vote"):
        score += 0.15
    if score < 0.16:
        return "very_short"
    if score < 0.30:
        return "short"
    if score < 0.52:
        return "medium"
    if score < 0.85:
        return "long"
    return "very_long"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _observation_key(observation: dict[str, Any]) -> str:
    rtt = round(float(observation.get("mean_rtt_ms", observation.get("estimated_rtt_ms", 0.0))) / 25.0)
    loss = round(float(observation.get("loss_rate", 0.0)) / 0.025)
    log_gap = int(observation.get("log_gap", 0))
    split = bool(observation.get("recent_split", observation.get("recent_split_vote", False)))
    return f"rtt={rtt};loss={loss};log={log_gap};split={int(split)}"


def _extract_chat_text(payload: dict[str, Any]) -> str:
    """Extract content text from an OpenAI-compatible chat completions response."""
    try:
        choices = payload.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, str):
                return content
    except (KeyError, IndexError, TypeError):
        pass
    return str(payload)


def _extract_json_object(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else text


def _safe_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")
    except Exception:
        return "unreadable_error_body"
    body = re.sub(r"sk-[A-Za-z0-9_\-]+", "sk-REDACTED", body)
    body = re.sub(r"gsk_[A-Za-z0-9_\-]+", "gsk-REDACTED", body)
    return body[:500]
