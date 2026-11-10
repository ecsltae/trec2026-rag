"""
Tool library for the v3 agent loop.

Each tool is a thin wrapper over existing retrieval/pool code. The ToolBox binds
one topic's harvested pool + embeddings and exposes OpenAI/Ollama-format schemas
plus an execute() dispatcher. The harness (agent_loop.py) enforces budgets.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from systems.rag.agentic_v2.harvest import read_pool

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_pool",
            "description": "Semantic search over this topic's harvested document pool. "
                           "Returns top document snippets with ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look for"},
                    "k": {"type": "integer", "description": "results to return (default 5, max 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "Read the full text of a document by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {"type": "string"},
                },
                "required": ["docid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_note",
            "description": "Record a finding for the final answer: one factual sentence "
                           "with the document ids that support it. Notes become the answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "facet_id": {"type": "string", "description": "which facet this addresses"},
                    "sentence": {"type": "string", "description": "one factual, specific sentence"},
                    "docids": {"type": "array", "items": {"type": "string"},
                               "description": "supporting document ids (must have been read or seen in search results)"},
                },
                "required": ["facet_id", "sentence", "docids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Declare research complete. Call when every facet has at least one note.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class ToolBox:
    """Per-topic tool executor bound to the harvested pool + local dense index."""

    def __init__(self, topic_dir: Path, embedder, snippet_chars: int = 300,
                 read_chars: int = 2500):
        self.d = topic_dir
        self.embedder = embedder
        self.snippet_chars = snippet_chars
        self.read_chars = read_chars

        pool = read_pool(topic_dir / "pool.jsonl")
        self.texts = {e["docid"]: e["text"] for e in pool}
        self.docids = json.loads((topic_dir / "emb_docids.json").read_text())
        self.emb = np.load(topic_dir / "embeddings.npy")

        self.notes = []          # accumulated take_note calls
        self.seen_docids = set() # docids surfaced via search/read (citation validity)
        self.finished = False

    def execute(self, name: str, args: dict) -> str:
        """Dispatch one tool call; returns the tool-result string for the model."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return json.dumps({"error": f"unknown tool '{name}'"})
        try:
            return handler(**args)
        except TypeError as e:
            return json.dumps({"error": f"bad arguments: {e}"})

    # -- tools ---------------------------------------------------------------

    def _tool_search_pool(self, query: str, k: int = 5) -> str:
        k = min(int(k), 10)
        q = self.embedder.encode(
            [f"Represent this sentence for searching relevant passages: {query}"],
            normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        sims = self.emb @ q[0]
        top = np.argsort(-sims)[:k]
        results = []
        for j in top:
            docid = self.docids[j]
            self.seen_docids.add(docid)
            results.append({"docid": docid,
                            "snippet": self.texts[docid][:self.snippet_chars],
                            "score": round(float(sims[j]), 3)})
        return json.dumps({"results": results})

    def _tool_read_doc(self, docid: str) -> str:
        text = self.texts.get(docid)
        if text is None:
            return json.dumps({"error": f"docid '{docid}' not in this topic's pool"})
        self.seen_docids.add(docid)
        return json.dumps({"docid": docid, "text": text[:self.read_chars]})

    def _tool_take_note(self, facet_id: str, sentence: str, docids: list) -> str:
        valid = [d for d in docids if d in self.seen_docids and d in self.texts]
        rejected = [d for d in docids if d not in valid]
        if not valid:
            return json.dumps({"error": "no valid docids — cite only documents "
                                        "you have seen via search_pool or read_doc",
                               "rejected": rejected})
        self.notes.append({"facet_id": facet_id,
                           "text": sentence.strip(),
                           "citations": valid})
        msg = {"noted": True, "notes_so_far": len(self.notes)}
        if rejected:
            msg["rejected_docids"] = rejected
        return json.dumps(msg)

    def _tool_finish(self) -> str:
        self.finished = True
        return json.dumps({"finished": True, "total_notes": len(self.notes)})
