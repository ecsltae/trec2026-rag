"""
Generation module for TREC RAG 2026.

Takes a topic + retrieved segments, produces a TREC RAG JSONL answer:
  {
    "topic_id": "...",
    "run_id": "...",
    "answer": [
      {"text": "Sentence one.", "citations": ["seg_id_a"]},
      ...
    ]
  }

Constraints enforced:
  - ≤ 400 words total
  - ≤ 100 unique cited segments
  - Each sentence has at least one citation from the retrieved set
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from trec_shared.ollama_client import OllamaClient
from trec_shared.trec_utils import count_words

# TREC RAG 2026 system instruction:
# - Ground every claim in retrieved passages, no parametric knowledge
# - Prioritise "vital" information (directly answers the query)
# - Follow the narrative closely (includes vital/okay sub-narratives)
# - Citation format: [passage_id] inline after each sentence
_SYSTEM = """\
You are a precise research assistant participating in the TREC RAG 2026 evaluation.

Your task: synthesize a factual answer from the retrieved passages below.

STRICT RULES:
1. Use ONLY information explicitly stated in the retrieved passages. Never add facts from your training knowledge.
2. Every sentence in your answer must end with one or more citation tags: [passage_id]
3. Use only passage IDs that appear in the provided list. Never invent passage IDs.
4. Write 4–8 sentences. Total answer must not exceed 400 words.
5. Focus on information directly relevant to the query title and narrative.
6. If a passage does not contain relevant information, do not cite it.
7. Do not start your answer with phrases like "Based on the passages" or "According to".
"""

_FALLBACK_TMPL = """\
QUERY TITLE: {query}

The retrieved passages may only be tangentially related to this query.
Identify any passage segments that contain ANY relevant information, even partial.
Write 1-3 sentences using only what is explicitly stated. Cite each sentence.
If truly no relevant information exists in any passage, write exactly one sentence:
"The retrieved passages do not contain information directly relevant to this query." [PASSAGE_ID]

PASSAGES:
{passages}\
"""

_USER_TMPL = """\
QUERY TITLE: {query}

NARRATIVE (what a complete answer must cover):
{narrative}

RETRIEVED PASSAGES:
{passages}

Write your answer now. End each sentence with [passage_id] citations. Use only the passage IDs listed above.\
"""


class RAGGenerator:
    def __init__(self, config):
        self.run_id = config["run_id"]
        gen_cfg = config.get("generation", {})
        self.max_words = gen_cfg.get("max_answer_words", 400)
        self.max_segs = gen_cfg.get("max_cited_segments", 100)
        # How many passages to include in the prompt (cost/quality tradeoff)
        self.max_prompt_segs = gen_cfg.get("max_prompt_segments", 50)
        # Enable/disable qwen3 /think mode: None=default, True=force, False=suppress
        self.think = gen_cfg.get("think", None)
        self.client = OllamaClient(
            model=gen_cfg.get("ollama_model", "qwen3:32b"),
            temperature=gen_cfg.get("temperature", 0.1),
        )

    def generate(self, topic_id, query, narrative, segments):
        """
        Args:
            topic_id: str
            query: str — the query title
            narrative: str — longer narrative (vital/okay sub-narratives)
            segments: dict {seg_id: text} or list of (seg_id, text) tuples

        Returns:
            dict with topic_id, run_id, answer list
        """
        if isinstance(segments, dict):
            seg_items = list(segments.items())
        else:
            seg_items = list(segments)

        selected = seg_items[:self.max_prompt_segs]
        valid_ids = {seg_id for seg_id, _ in selected}

        passages_text = "\n\n".join(
            f"[{seg_id}] {text[:700].strip()}"
            for seg_id, text in selected
        )
        prompt = _USER_TMPL.format(
            query=query,
            narrative=narrative or query,
            passages=passages_text,
        )
        raw = self.client.generate(prompt, system=_SYSTEM, think=self.think)
        # Strip qwen3 <think>...</think> blocks (chain-of-thought leakage)
        raw = _strip_thinking(raw)
        answer = _parse_answer(raw, valid_ids, self.max_words, self.max_segs)

        # Fallback: if model produced no valid cited sentences, retry with a
        # more permissive prompt that asks for the most relevant partial info.
        if not answer and selected:
            raw2 = self.client.generate(
                _FALLBACK_TMPL.format(query=query, passages=passages_text),
                system=_SYSTEM,
                think=self.think,
            )
            raw2 = _strip_thinking(raw2)
            answer = _parse_answer(raw2, valid_ids, self.max_words, self.max_segs)

        return {"topic_id": topic_id, "run_id": self.run_id, "answer": answer}


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks emitted by qwen3 in thinking mode."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping inline citation tags attached.

    Delegates to shared.trec_utils.split_sentences. The previous local regex split after ANY
    "period + space + capital", which shattered abbreviations and shipped headless fragments to
    three validated submissions ("Roe v." + "Wade established ...", "Franklin D." + "Roosevelt
    leads ..."). See the comment block in shared/trec_utils.py.
    """
    from trec_shared.trec_utils import split_sentences
    return split_sentences(text)


def _parse_answer(raw_text: str, valid_seg_ids: set, max_words: int, max_segs: int) -> list:
    """Parse model output into sentence+citation list, enforcing constraints."""
    citation_pattern = re.compile(r"\[([^\]\s]+)\]")

    # Clean up the raw text
    raw_text = re.sub(r"\n{2,}", "\n", raw_text)

    # Split into candidate sentences (newline-or sentence boundary)
    candidates = []
    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip list markers
        line = re.sub(r"^\d+[.)]\s*", "", line)
        line = re.sub(r"^[-*•]\s*", "", line)
        # Further split on sentence boundaries within a line
        for sent in _split_sentences(line):
            candidates.append(sent)

    sentences = []
    total_words = 0
    seen_citations: set = set()

    for raw_sent in candidates:
        citations_found = citation_pattern.findall(raw_sent)
        valid_citations = [c for c in citations_found if c in valid_seg_ids]

        # Remove citation brackets from displayed text
        text = citation_pattern.sub("", raw_sent).strip()
        # Collapse whitespace left by removed citations, fix " ." → "."
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = text.rstrip(",;:")
        # The synthesis prompt shows the model its facet blueprint ("- f1 (Name): ..."), and the
        # model sometimes echoes those internal ids into the answer. 36 such sentences shipped to
        # three validated submissions. This is the single parse choke point for both the plain and
        # agentic paths, so scrubbing here is the guarantee rather than a hope.
        from trec_shared.trec_utils import scrub_internal_labels
        text = scrub_internal_labels(text)

        if not text:
            continue
        if not valid_citations:
            continue

        words_here = count_words(text)
        if total_words + words_here > max_words:
            break

        # Enforce max cited segments (deduplicate, respect cap)
        deduped = []
        for c in valid_citations:
            if c not in seen_citations:
                if len(seen_citations) >= max_segs:
                    # Cap reached — can still cite already-seen ones
                    continue
                seen_citations.add(c)
            if c not in deduped:
                deduped.append(c)

        if not deduped:
            continue

        sentences.append({"text": text, "citations": deduped})
        total_words += words_here

    return sentences
