"""
Robust JSON extraction from LLM output.

parse_json_array / parse_json_object handle the usual LLM quirks: <think> blocks,
markdown fences, prose around the JSON. parse_json_object optionally does one
"repair" round-trip: re-prompt the model with the parse error and format="json".
"""
import json
import re


def _strip_noise(text: str) -> str:
    """Remove <think> blocks and markdown code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    return text.strip()


def parse_json_array(text: str, fallback: list = None) -> list:
    """Extract a JSON array from LLM output, with fallback."""
    text = _strip_noise(text)
    # First [...] block (non-greedy then greedy — nested arrays defeat non-greedy)
    for pattern in (r"\[.*\]", r"\[.*?\]"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                continue
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return fallback if fallback is not None else []


def parse_json_object(text: str, fallback: dict = None,
                      client=None, repair_prompt: str = None) -> dict:
    """Extract a JSON object from LLM output.

    If parsing fails and `client` is given, does ONE repair round-trip:
    re-prompts with the broken output + error using format="json".

    Args:
        text: raw LLM output
        fallback: returned when all parsing fails (default {})
        client: optional OllamaClient for the repair retry
        repair_prompt: context line for the repair prompt (e.g. the original task)
    """
    stripped = _strip_noise(text)
    # Greedy {...} block first (JSON objects nest), then full text
    for candidate in _object_candidates(stripped):
        try:
            result = json.loads(candidate)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    if client is not None:
        try:
            error_hint = _first_parse_error(stripped)
            prompt = (
                f"{repair_prompt or 'Fix the JSON below.'}\n\n"
                f"The following output was supposed to be a single valid JSON object "
                f"but failed to parse ({error_hint}):\n\n{stripped[:4000]}\n\n"
                f"Output ONLY the corrected JSON object, nothing else."
            )
            repaired = client.generate(prompt, think=False, format="json",
                                       num_predict=2048)
            result = json.loads(_strip_noise(repaired))
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    return fallback if fallback is not None else {}


def _object_candidates(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        yield m.group(0)
    yield text


def _first_parse_error(text: str) -> str:
    try:
        json.loads(text)
        return "unknown error"
    except json.JSONDecodeError as e:
        return str(e)
