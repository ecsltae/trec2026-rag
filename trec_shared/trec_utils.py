"""
Shared TREC I/O utilities: run file read/write, RRF fusion, format validation.
"""
import hashlib
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Retrieval run (6-field TSV)
# ---------------------------------------------------------------------------

def write_trec_run(results, path, run_id):
    """Write standard TREC retrieval run file.

    Args:
        results: list of (topic_id, doc_id, rank, score)
        path: output file path
        run_id: run identifier string
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for topic_id, doc_id, rank, score in results:
            f.write(f"{topic_id} Q0 {doc_id} {rank} {score:.6f} {run_id}\n")


def read_trec_run(path):
    """Read a TREC run file.

    Returns:
        dict: topic_id -> list of (doc_id, rank, score) sorted by rank
    """
    results = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            topic_id, _, doc_id, rank, score, _ = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4]), parts[5]
            results.setdefault(topic_id, []).append((doc_id, rank, score))
    for tid in results:
        results[tid].sort(key=lambda x: x[1])
    return results


def validate_trec_run(path, max_per_topic=100):
    """Validate a TREC run file. Returns list of error strings (empty = valid)."""
    errors = []
    topic_counts = {}
    with open(path) as f:
        for i, line in enumerate(f, 1):
            parts = line.strip().split()
            if len(parts) != 6:
                errors.append(f"Line {i}: expected 6 fields, got {len(parts)}")
                continue
            topic_id = parts[0]
            topic_counts[topic_id] = topic_counts.get(topic_id, 0) + 1
    for tid, count in topic_counts.items():
        if count > max_per_topic:
            errors.append(f"Topic {tid}: {count} results (max {max_per_topic})")
    return errors


# ---------------------------------------------------------------------------
# RAG generation JSONL  (TREC RAG 2026 format)
# ---------------------------------------------------------------------------
# Output format per topic:
# {
#   "metadata": {"topic_id": "...", "run_id": "...", "type": "automatic"},
#   "references": ["docid_0", "docid_1", ...],   <- all cited docs, deduped
#   "answer": [
#     {"text": "Sentence.", "citations": [0, 1]}  <- zero-indexed into references
#   ]
# }
# Rules: every reference cited by ≥1 sentence; every sentence cites ≥1 reference.

def write_rag_jsonl(results, path):
    """Write TREC RAG 2026 generation run as JSONL (one object per topic).

    Args:
        results: list of dicts produced by RAGGenerator.generate()
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            obj = _to_rag_2026_format(r)
            _validate_rag_answer_2026(obj)
            f.write(json.dumps(obj) + "\n")


def _to_rag_2026_format(r):
    """Convert internal generate() output to 2026 JSONL format."""
    # Internal format: answer = [{text, citations: [seg_id, ...]}]
    # 2026 format:     references = [seg_id, ...], answer sentences cite by index
    ref_list = []
    ref_index = {}
    answer_2026 = []
    for sentence in r.get("answer", []):
        citation_indices = []
        for seg_id in sentence.get("citations", []):
            if seg_id not in ref_index:
                ref_index[seg_id] = len(ref_list)
                ref_list.append(seg_id)
            citation_indices.append(ref_index[seg_id])
        answer_2026.append({"text": sentence["text"], "citations": citation_indices})
    return {
        "metadata": {
            "topic_id": r["topic_id"],
            "run_id": r["run_id"],
            "type": "automatic",
        },
        "references": ref_list,
        "answer": answer_2026,
    }


def _validate_rag_answer_2026(obj):
    topic_id = obj["metadata"]["topic_id"]
    refs = obj["references"]
    sentences = obj["answer"]
    cited_indices = {i for s in sentences for i in s.get("citations", [])}

    for s in sentences:
        if not s.get("citations"):
            raise ValueError(f"Topic {topic_id}: sentence with no citations: {s['text'][:60]}")
    # Every reference must be cited by at least one sentence
    uncited = [refs[i] for i in range(len(refs)) if i not in cited_indices]
    if uncited:
        raise ValueError(f"Topic {topic_id}: uncited references: {uncited[:3]}")
    # Citation indices must be valid
    oob = [i for s in sentences for i in s["citations"] if i >= len(refs)]
    if oob:
        raise ValueError(f"Topic {topic_id}: out-of-bounds citation indices: {oob[:3]}")


def read_rag_jsonl(path):
    """Read a RAG generation JSONL file. Returns list of result dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(ranked_lists, k=60):
    """Merge multiple ranked lists with RRF.

    Args:
        ranked_lists: list of lists of (doc_id, score), each sorted best-first
        k: RRF constant (default 60)

    Returns:
        list of (doc_id, rrf_score) sorted descending by score
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Topics and qrels
# ---------------------------------------------------------------------------

def load_topics_tsv(path):
    """Load TREC topics TSV (topic_id <TAB> narrative).

    Returns:
        dict: topic_id -> narrative string
    """
    topics = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                topics[parts[0]] = parts[1]
    return topics


def load_topics_jsonl(path):
    """Load TREC RAG topics JSONL (one topic per line) or JSON array.

    Supports:
      - RAG25/2026 JSON array: [{"id": "...", "narrative": "..."}, ...]
      - JSONL: one {"id"/"topic_id", "title"?, "narrative"} per line

    Returns:
        dict: topic_id -> {"title": str, "narrative": str}
    """
    with open(path) as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            # JSON array format (trec25_narratives_final.json)
            objs = json.load(f)
        else:
            objs = [json.loads(line) for line in f if line.strip()]
    topics = {}
    for obj in objs:
        tid = obj.get("id") or obj.get("topic_id")
        narrative = obj.get("narrative", obj.get("description", ""))
        title = obj.get("title", narrative[:80])
        topics[tid] = {"title": title, "narrative": narrative}
    return topics


def load_topics(path):
    """Auto-detect TSV vs JSON/JSONL and load topics.

    Returns:
        dict: topic_id -> {"title": str, "narrative": str}
        (TSV format is wrapped to same shape for pipeline compatibility)
    """
    if str(path).endswith((".jsonl", ".json")):
        return load_topics_jsonl(path)
    # TSV: wrap plain narrative string into dict
    raw = load_topics_tsv(path)
    return {tid: {"title": narrative.split("\n")[0], "narrative": narrative}
            for tid, narrative in raw.items()}


def load_qrels(path):
    """Load TREC qrels file (topic_id Q0 doc_id relevance).

    Returns:
        dict: topic_id -> dict of doc_id -> relevance_int
    """
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id, _, doc_id, rel = parts[0], parts[1], parts[2], int(parts[3])
            qrels.setdefault(topic_id, {})[doc_id] = rel
    return qrels


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def passage_hash(text):
    """Short hash of a passage string for cache keys."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def count_words(text):
    return len(text.split())


def truncate_to_words(text, max_words):
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# --------------------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------------------
# Both systems/rag/generate.py and systems/ragtime/generate_report.py independently used
#   re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
# which splits after ANY period followed by whitespace and a capital. On the TREC RAG 2026
# submissions that shattered abbreviations and produced HEADLESS FRAGMENTS that shipped:
#   "Roe v. Wade established ..."          -> "Roe v."  + "Wade established ..."
#   "under the U. S. Constitution, a ..."  -> "... U."   + "S."  + "Constitution to refuse ..."
#   "assassination of Franklin D. Roose.." -> "... D."   + "Roosevelt leads to a world ..."
#   "J.R.R. Tolkien's world"               -> "J.R.R."   + "Tolkien's world."
# Twelve such fragments reached three validated submissions and had to be repaired by hand;
# two of them were undetectable by regex afterwards because they read as ordinary sentences.
# RAGTIME generates multilingual reports with the same sentence+citation structure, so it
# carried the identical latent bug. Fixed once here.

_SENT_GUARD = ""          # private-use codepoint; cannot occur in corpus text

_ABBREVIATIONS = (
    "v", "vs",                                                   # legal citations
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "St", "Rev",     # titles
    "Hon", "Gen", "Sen", "Rep", "Gov", "Col", "Lt", "Capt",
    "etc", "cf", "al", "viz", "ca", "approx", "est",              # scholarly / latin
    "No", "Nos", "Fig", "Figs", "Vol", "Vols", "Ch", "Sec",       # references
    "pp", "Eq", "Ref", "Art", "Inc", "Ltd", "Co", "Corp",
)


def _guard_abbreviations(text: str) -> str:
    """Replace the period of a known abbreviation with a sentinel so it cannot split."""
    # Initials and dotted initialisms: "U.S.", "U. S.", "J.R.R.", "Franklin D. Roosevelt".
    # Requires a SINGLE-letter word, so real sentence ends ("...the FBI. Meanwhile") are
    # untouched: \b cannot match inside "FBI".
    text = re.sub(r"\b([A-Za-z])\.(?=\s*[A-Z])", lambda m: m.group(1) + _SENT_GUARD, text)
    text = re.sub(r"\be\.g\.", "e" + _SENT_GUARD + "g" + _SENT_GUARD, text, flags=re.I)
    text = re.sub(r"\bi\.e\.", "i" + _SENT_GUARD + "e" + _SENT_GUARD, text, flags=re.I)
    for ab in _ABBREVIATIONS:
        text = re.sub(rf"\b({re.escape(ab)})\.", rf"\1{_SENT_GUARD}", text, flags=re.I)
    return text


def split_sentences(text: str) -> list:
    """Split text into sentences without shattering abbreviations.

    Keeps any trailing inline citation bracket attached to the sentence it follows.
    """
    guarded = _guard_abbreviations(text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\"'(])", guarded)
    out = []
    for p in parts:
        p = p.replace(_SENT_GUARD, ".").strip()
        if p:
            out.append(p)
    return out


# --------------------------------------------------------------------------------------
# Internal-label scrubbing
# --------------------------------------------------------------------------------------
# The synthesis prompt shows the model its facet blueprint as "- f1 (Name): why ...", because
# the coverage self-check needs those ids to report {"uncovered": ["f2"]}. The model sometimes
# echoes them straight into the answer. 36 sentences shipped to three validated TREC RAG 2026
# submissions carrying leaked ids, in two shapes:
#     "f3 (Mortality Risk Timeline): The widowhood effect is defined as ..."   (prefix)
#     "... does not provide the figures requested in f1."                      (mid-sentence)
# The prefix form was caught by a start-anchored regex; the mid-sentence form was not, and needed
# a second repair pass. Scrubbing at the single parse choke point is the guarantee.

_FACET_PREFIX = re.compile(r"^\**\s*f\d+\**\s*(\([^)]*\))?\s*[:.]\s*", re.IGNORECASE)
_FACET_PHRASES = [
    (re.compile(r"\bleaving facets?\s+f\d+(?:\s*,\s*f\d+)*(?:\s*,?\s*and\s+f\d+)?\b", re.I),
     "leaving those aspects"),
    (re.compile(r"\bas (?:requested|required|specified) in f\d+\b", re.I), "as requested"),
    (re.compile(r"\b(?:requested|required|specified) in f\d+\b", re.I),
     "requested in the narrative"),
    (re.compile(r"\bfor facet f\d+\b", re.I), ""),
    (re.compile(r"\b(?:in|per|from|under|addressing) f\d+\b", re.I), ""),
]


def scrub_internal_labels(text: str) -> str:
    """Remove leaked internal facet identifiers from a generated sentence.

    Strips a leading "f3 (Name):" scaffold and rewrites mid-sentence references such as
    "as requested in f4". Leaves the substantive claim untouched.
    """
    out = _FACET_PREFIX.sub("", text)
    for pat, repl in _FACET_PHRASES:
        out = pat.sub(repl, out)
    out = re.sub(r"\s{2,}", " ", out).replace(" ,", ",").replace(" .", ".").strip()
    if out and out[0].islower():
        out = out[0].upper() + out[1:]
    return out
