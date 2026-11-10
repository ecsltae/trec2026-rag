"""
Thin wrapper over the Ollama Python SDK for text and multimodal inference.
Requires: pip install ollama (already in MPvenv 0.6.1)
"""
import base64
import time
from pathlib import Path

import ollama


class OllamaClient:
    def __init__(self, model="qwen3:32b", timeout=None, temperature=0.0):
        """
        Args:
            timeout: per-request timeout in seconds, enforced at the HTTP layer.
                     None (default) = no limit. Set this for batch pipelines —
                     runaway generations have been observed to run for hours.
        """
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        # Dedicated client when a timeout is requested; module-level calls
        # (no timeout) are kept for backward compatibility.
        self._client = ollama.Client(timeout=timeout) if timeout else ollama

    def is_available(self):
        try:
            models = [m.model for m in ollama.list().models]
            return any(self.model in m for m in models)
        except Exception:
            return False

    def list_models(self):
        try:
            return [m.model for m in ollama.list().models]
        except Exception:
            return []

    def generate(self, prompt, system=None, temperature=None, think=None,
                 format=None, num_predict=None, keep_alive=None, num_ctx=16384):
        """Text-only generation. Returns response string.

        Args:
            think: None (default) | True | False — passed as Ollama's NATIVE think
                   parameter for qwen3-family models. (The old '/no_think' prompt hack
                   silently fails on qwen3.5 — the model thinks anyway and burns the
                   whole num_predict budget; verified 2026-07-08.)
            format: "json" to constrain output to valid JSON (Ollama structured output).
            num_predict: max tokens to generate (latency cap).
            keep_alive: model residency after the call; 0 unloads immediately
                        (use on the last call before a GPU-heavy non-Ollama phase).
            num_ctx: context window (default 16384 — the model-default 262k KV cache
                     wastes VRAM and slows the 122b).
        """
        temp = temperature if temperature is not None else self.temperature
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        options = {"temperature": temp}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        kwargs = {}
        if format is not None:
            kwargs["format"] = format
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if think is not None and "qwen3" in self.model.lower():
            kwargs["think"] = think
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs,
        )
        content = resp.message.content.strip()
        # Strip any residual <think>...</think> blocks (safety net)
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def generate_messages(self, messages, temperature=None):
        """Generate from an explicit messages list."""
        temp = temperature if temperature is not None else self.temperature
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options={"temperature": temp},
        )
        return resp.message.content.strip()

    def chat(self, messages, tools=None, temperature=None, think=None,
             num_predict=None, keep_alive=None, num_ctx=16384):
        """Chat with optional native tool-calling.

        Args:
            messages: list of {"role", "content", ...} dicts. Tool results are
                appended as {"role": "tool", "content": <json str>, "tool_name": <name>}.
            tools: list of tool schemas in OpenAI/Ollama function format:
                {"type": "function", "function": {"name", "description", "parameters"}}
            think: qwen3 thinking control (True/False/None), applied via chat 'think' param.

        Returns:
            (content: str, tool_calls: list) — tool_calls is a list of
            {"name": str, "arguments": dict}; empty list if the model answered directly.
        """
        temp = temperature if temperature is not None else self.temperature
        options = {"temperature": temp}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        kwargs = {}
        if tools:
            kwargs["tools"] = tools
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if think is not None and "qwen3" in self.model.lower():
            kwargs["think"] = think
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs,
        )
        content = (resp.message.content or "").strip()
        import re
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        tool_calls = []
        for tc in (resp.message.tool_calls or []):
            args = tc.function.arguments
            if isinstance(args, str):
                import json as _json
                try:
                    args = _json.loads(args)
                except _json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append({"name": tc.function.name, "arguments": dict(args)})
        return content, tool_calls

    def unload(self):
        """Force-unload this model from VRAM (before a GPU-heavy non-Ollama phase)."""
        try:
            ollama.chat(model=self.model, messages=[{"role": "user", "content": ""}],
                        keep_alive=0)
        except Exception:
            pass

    def generate_multimodal(self, prompt, images, system=None, temperature=None,
                            think=None, num_predict=None, num_ctx=16384,
                            keep_alive=None):
        """Vision-language generation with image inputs.

        Args:
            prompt: text prompt string
            images: list of image sources — each can be:
                - str/Path: file path to JPEG/PNG
                - bytes: raw image bytes
                - str starting with 'data:': already base64-encoded data URI
            system: optional system message
            temperature: optional override
            think: qwen3-family native thinking control (True/False/None)
            num_predict: max tokens to generate (latency cap for runaway loops)
            num_ctx: context window. Defaults to 16384 to match generate().
                     Leaving this unset makes ollama load qwen3-vl at its
                     default 262144-token context, which reserves ~48 GB of KV
                     cache for prompts a few thousand tokens long: measured
                     48 GB -> 9.7 GB and 32.1s -> 17.5s per topic once set.
                     Pass None only if you genuinely need the model default.
            keep_alive: model residency after the call

        Returns:
            str: model response
        """
        temp = temperature if temperature is not None else self.temperature
        encoded = []
        for img in images:
            if isinstance(img, (str, Path)) and not str(img).startswith("data:"):
                with open(img, "rb") as f:
                    encoded.append(base64.b64encode(f.read()).decode())
            elif isinstance(img, bytes):
                encoded.append(base64.b64encode(img).decode())
            else:
                encoded.append(img)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt, "images": encoded})

        options = {"temperature": temp}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if num_ctx is not None:
            options["num_ctx"] = num_ctx
        kwargs = {}
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        if think is not None and "qwen3" in self.model.lower():
            kwargs["think"] = think
        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            **kwargs,
        )
        return resp.message.content.strip()

    def generate_with_retry(self, prompt, system=None, retries=3, backoff=5.0,
                            think=None, **kwargs):
        """generate() with exponential backoff retry."""
        for attempt in range(retries):
            try:
                return self.generate(prompt, system=system, think=think, **kwargs)
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff * (2 ** attempt))

    def generate_multimodal_with_retry(self, prompt, images, system=None,
                                       retries=3, backoff=5.0, **kwargs):
        """generate_multimodal() with exponential backoff retry (timeouts included)."""
        for attempt in range(retries):
            try:
                return self.generate_multimodal(prompt, images, system=system, **kwargs)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(backoff * (2 ** attempt))
