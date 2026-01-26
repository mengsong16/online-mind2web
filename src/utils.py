import base64
import io
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)
import os
import backoff
import tempfile

# ---- Evaluation debug stats (cross-process via append-only log file) ----
_EVAL_STATS_PATH = os.path.join(tempfile.gettempdir(), 'open2mind_eval_stats.log')


_EVAL_STATS_SHARED = None
_EVAL_STATS_LOCK = None

def set_eval_stats_shared(shared_dict, lock=None):
    """Attach a multiprocessing.Manager().dict() for per-event stats (no disk writes)."""
    global _EVAL_STATS_SHARED, _EVAL_STATS_LOCK
    _EVAL_STATS_SHARED = shared_dict
    _EVAL_STATS_LOCK = lock

def set_eval_stats_path(path: str):
    """Set the shared eval stats log path (call once in the main process before workers spawn)."""
    global _EVAL_STATS_PATH
    _EVAL_STATS_PATH = path
    try:
        os.makedirs(os.path.dirname(_EVAL_STATS_PATH), exist_ok=True)
    except Exception:
        pass

def append_eval_log_lines(lines):
    """Append raw log lines to the shared eval stats log (used for the final summary)."""
    try:
        with open(_EVAL_STATS_PATH, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(str(line) + "\n")
    except Exception:
        pass


def reset_eval_stats():
    """Clear shared eval stats (and remove summary log file)."""
    # clear shared counters
    try:
        if _EVAL_STATS_SHARED is not None:
            if _EVAL_STATS_LOCK is not None:
                with _EVAL_STATS_LOCK:
                    _EVAL_STATS_SHARED.clear()
            else:
                _EVAL_STATS_SHARED.clear()
    except Exception:
        pass

    # clear summary log file
    try:
        if os.path.exists(_EVAL_STATS_PATH):
            os.remove(_EVAL_STATS_PATH)
    except Exception:
        pass

def log_eval_stat(kind: str, value: int = 1):
    """Increment a stat counter in shared dict; do NOT write to disk."""
    try:
        if _EVAL_STATS_SHARED is None:
            return
        if _EVAL_STATS_LOCK is not None:
            with _EVAL_STATS_LOCK:
                _EVAL_STATS_SHARED[kind] = int(_EVAL_STATS_SHARED.get(kind, 0)) + int(value)
        else:
            _EVAL_STATS_SHARED[kind] = int(_EVAL_STATS_SHARED.get(kind, 0)) + int(value)
    except Exception:
        pass

def read_eval_stats():
    """Read aggregated stats from shared dict."""
    try:
        if _EVAL_STATS_SHARED is None:
            return {}
        if _EVAL_STATS_LOCK is not None:
            with _EVAL_STATS_LOCK:
                return dict(_EVAL_STATS_SHARED)
        return dict(_EVAL_STATS_SHARED)
    except Exception:
        return {}


def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_predication(response, mode):
    """Extract the prediction from the response.

    NOTE: We count cases where `Status:` cannot be parsed (missing or malformed).
    """
    if mode == "WebVoyager_eval":
        # WebVoyager uses a different convention in this repo.
        resp = "" if response is None else str(response)
        return 0 if "FAILURE" in resp else 1

    # Modes that rely on parsing `Status:`
    if mode in {"Autonomous_eval", "AgentTrek_eval", "WebJudge_Online_Mind2Web_eval", "WebJudge_general_eval"}:
        resp = "" if response is None else str(response)
        low = resp.lower()
        if "status:" not in low:
            log_eval_stat("status_parse_error", 1)
            return 0
        try:
            tail = low.split("status:", 1)[1]
            return 1 if "success" in tail else 0
        except Exception:
            log_eval_stat("status_parse_error", 1)
            return 0

    raise ValueError(f"Unknown mode: {mode}")
class OpenaiEngine():
    def __init__(
        self,
        api_key=None,
        stop=[],
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri = "",
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.client = OpenAI(
                        api_key=api_key,
                    )

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=log_error
    )
    def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs):
        model = model if model else self.model

        # OpenAI reasoning models (e.g., o4-mini / o3-*) do NOT accept `max_tokens`.
        # They require `max_completion_tokens` instead.
        m = (model or "").lower()
        is_reasoning_model = m.startswith("o")  # covers o4-mini, o3-*, o1-*, etc.

        req_kwargs = dict(kwargs)
        if is_reasoning_model:
            # Avoid passing unsupported/ignored sampling params for reasoning models.
            req_kwargs.pop("max_tokens", None)
            req_kwargs.pop("temperature", None) # temperature=1
            req_kwargs["max_completion_tokens"] = max_new_tokens
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **req_kwargs,
            )
        else:
            req_kwargs.pop("max_completion_tokens", None)
            req_kwargs["max_tokens"] = max_new_tokens
            req_kwargs["temperature"] = temperature
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                **req_kwargs,
            )
        return [choice.message.content for choice in response.choices]
    