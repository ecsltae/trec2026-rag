"""
Stage 5 — Synthesize: blueprint-scaffolded answer generation.

Context = anchor top-10 + top-6 per facet (deduped). One scaffolded synthesis
call ("address each facet, cite every sentence"), then a coverage self-check
(JSON), then at most ONE gap round: re-retrieve for uncovered facets from the
local pool rankings and run a repair synthesis for the missing facets only.

Model note: works with qwen3:32b or qwen3.5:122b (pipeline flag --synth-model).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from trec_shared.ollama_client import OllamaClient
from trec_shared.llm_json import parse_json_object
from systems.rag.generate import _parse_answer, _strip_thinking
from systems.rag.agentic_v2.harvest import read_pool

MAX_WORDS = 450
MAX_SEGS = 30
# The official cap is 1024 words for the whole answer. The shipped arm cannot reach this
# (450 + at most 450 from the gap round), so the final trim only ever bites the longer arms.
TOTAL_WORDS = 1000

_SYNTH_SYSTEM = """You are a research writer producing a fully-cited answer for a demanding reader.
Rules:
- EVERY sentence must end with one or more citations like [docid] taken ONLY from the provided passages
- Never invent document ids; never write an uncited sentence
- ONE claim per sentence: each sentence must be FULLY supported by the passage(s) it cites.
  Do not append an unsupported elaboration to a supported fact — split or drop it.
- If a facet is NOT covered by the passages, SILENTLY OMIT it. Never write meta-commentary
  like "the passages do not mention X" or "no information is available on Y" — such
  sentences are unsupported and forbidden. Only write what the evidence supports.
- Prefer facets you CAN support well; a shorter fully-grounded answer beats a longer hedged one.
- Be concrete: numbers, mechanisms, named actors — not generic filler
- Flowing prose, no headers; 1-3 sentences per covered facet
- The facet ids below (f1, f2, ...) are INTERNAL labels for your planning only. NEVER write them
  in the answer — not as a prefix ("f3 (Name): ...") and not inside a sentence ("as requested in
  f4"). The reader has never seen them."""

# Post-results arm (eval/analysis/PREREG_post_results_rerun.md). The official generation
# measures turned out to be coverage of expected facts plus human preference; citation support
# was not scored. This prompt drops the two rules written for support ("silently omit", "shorter
# beats longer") and asks for coverage instead. Every sentence must still be cited.
_SYNTH_SYSTEM_COVERAGE = """You are a research writer producing a complete, fully-cited answer for a demanding reader.
Rules:
- EVERY sentence must end with one or more citations like [docid] taken ONLY from the provided passages
- Never invent document ids; never write an uncited sentence
- ONE claim per sentence: each sentence must be supported by the passage(s) it cites
- COVER THE WHOLE REQUEST: address every facet below and every question the request asks. For each,
  give the specific facts the passages provide: numbers, names, dates, steps, conditions, trade-offs
  and differing views. If the passages support a facet only thinly, still write the best-supported
  sentence you can for it.
- Never write meta-commentary like "the passages do not mention X" or "no information is available on Y"
- Be concrete, not generic; no filler and no repetition
- Flowing prose, no headers; 3-6 sentences per facet; aim for 600-900 words in total
- The facet ids below (f1, f2, ...) are INTERNAL labels for your planning only. NEVER write them
  in the answer — not as a prefix ("f3 (Name): ...") and not inside a sentence ("as requested in
  f4"). The reader has never seen them."""

# Synthesis arms. "shipped" is exactly the configuration behind the submitted runs.
ARMS = {
    "shipped":  {"system": _SYNTH_SYSTEM, "max_words": MAX_WORDS, "max_segs": MAX_SEGS,
                 "sent_scale": 1, "char_per_doc": 700, "best_window": False},
    "budget":   {"system": _SYNTH_SYSTEM, "max_words": 950, "max_segs": 50,
                 "sent_scale": 2, "char_per_doc": 700, "best_window": False},
    "coverage": {"system": _SYNTH_SYSTEM_COVERAGE, "max_words": 950, "max_segs": 50,
                 "sent_scale": 2, "char_per_doc": 700, "best_window": False},
    "evidence": {"system": _SYNTH_SYSTEM_COVERAGE, "max_words": 950, "max_segs": 50,
                 "sent_scale": 2, "char_per_doc": 1000, "best_window": True},
    # The lab's idea of 2026-11-09 (PREREG_post_results_rerun.md, addendum A): evidence, plus a
    # weight per sub-topic, plus up to three review passes that also repair the sub-topics the
    # reviewer finds too thin for their weight. Runs through _synthesize_multipass.
    "multipass": {"system": _SYNTH_SYSTEM_COVERAGE, "max_words": 950, "max_segs": 50,
                  "sent_scale": 2, "char_per_doc": 1000, "best_window": True,
                  "passes": 3, "weights": True},
}

_SYNTH_TMPL = """REQUEST:
{narrative}

FACETS a complete answer must cover:
{facet_lines}

PASSAGES (cite by id):
{passages}

Write the answer now (target {n_sent_lo}-{n_sent_hi} sentences total, every sentence cited):"""

_COVERAGE_SYSTEM = """You are a strict reviewer. Given an answer and a list of required facets,
output JSON: {"uncovered": ["facet ids not adequately addressed"], "weak": ["facet ids addressed only superficially"]}"""

_COVERAGE_TMPL = """ANSWER:
{answer_text}

REQUIRED FACETS:
{facet_lines}

Output the JSON verdict:"""

_REPAIR_TMPL = """REQUEST:
{narrative}

The following facets are MISSING from an otherwise complete answer. For each facet
that the passages actually support, write 1-3 cited sentences (citations [docid] from
the passages only). One fully-supported claim per sentence. If the passages do NOT
support a facet, write nothing for it — never write "no information available".

MISSING FACETS:
{facet_lines}

PASSAGES:
{passages}

Write only the new supported sentences:"""


def _select_context(rankings: dict, pool_texts: dict,
                    anchor_n: int = 10, facet_n: int = 6) -> list:
    """Ordered, deduped [(docid, text)] context: anchor first, then facets."""
    chosen = []
    seen = set()
    for docid, _ in rankings.get("_anchor", [])[:anchor_n]:
        if docid not in seen and pool_texts.get(docid):
            seen.add(docid)
            chosen.append(docid)
    for fid, ranking in rankings.items():
        if fid == "_anchor":
            continue
        for docid, _ in ranking[:facet_n]:
            if docid not in seen and pool_texts.get(docid):
                seen.add(docid)
                chosen.append(docid)
    return [(d, pool_texts[d]) for d in chosen]


def _passages_block(context: list, char_per_doc: int = 700, terms: set = None) -> str:
    if terms:
        return "\n\n".join(f"[{d}] {_best_window(t, terms, char_per_doc)}" for d, t in context)
    return "\n\n".join(f"[{d}] {t[:char_per_doc]}" for d, t in context)


_STOP = set("about after also among and are because been being between both but can could does "
            "each from have having here how into its more most much must other over same should "
            "some such than that the their them then there these they this those through very "
            "want were what when where which while who whom why will with would your".split())


def _query_terms(narrative: str, facets: list) -> set:
    """Content words of the request, the facet names and the facet queries."""
    text = " ".join([narrative] + [f["name"].replace("_", " ") for f in facets]
                    + [q for f in facets for q in f.get("queries", [])])
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in _STOP}


def _best_window(text: str, terms: set, width: int, stride: int = 250, max_chars: int = 30000) -> str:
    """The width-char window of the document that contains the most distinct query terms.

    Documents average ~24 KB and the shipped arm shows the writer only their first 700 characters,
    which are often a title or boilerplate. Ties go to the earlier window, so a document whose head
    is as good as anything later still shows its head.
    """
    best, best_score = text[:width], -1
    for start in range(0, max(1, min(len(text), max_chars) - width + stride), stride):
        window = text[start:start + width]
        score = len(terms & set(re.findall(r"[a-z0-9]+", window.lower())))
        if score > best_score:
            best, best_score = window, score
    return best


def _facet_lines(facets: list, only_ids: set = None) -> str:
    lines = []
    for f in facets:
        if only_ids is not None and f["id"] not in only_ids:
            continue
        lines.append(f"- {f['id']} ({f['name']}): {f['why']}")
    return "\n".join(lines)


_WEIGHT_SYSTEM = "You decide how much space an answer should give to each part of a request. Output JSON only."

_WEIGHT_TMPL = """REQUEST:
{narrative}

FACETS of a complete answer:
{facet_lines}

How central is each facet to what this reader asked for? 3 = central, asked for directly;
2 = important; 1 = useful background. Output JSON: {{"weights": {{"<facet id>": <1-3>, ...}}}}"""

_REVIEW_SYSTEM = """You are a strict reviewer. Given an answer and the facets it must cover, each with a weight
(3 = central to the request, 1 = background), output JSON:
{"uncovered": ["facet ids the answer does not address"], "weak": ["facet ids addressed too thinly for their weight"]}"""

_REVIEW_TMPL = """ANSWER:
{answer_text}

REQUIRED FACETS (with weights):
{facet_lines}

Output the JSON verdict:"""

_REPAIR_MP_TMPL = """REQUEST:
{narrative}

The answer below is missing or too thin on the facets listed. For each of them, write 2-4 NEW
sentences with specific facts from the passages (numbers, names, steps, conditions, trade-offs)
that the answer does not already state. Cite every sentence with [docid] from the passages. Do not
repeat the answer. If the passages add nothing new for a facet, write nothing for it, and never
write that information is missing.

FACETS TO STRENGTHEN:
{facet_lines}

THE ANSWER SO FAR:
{answer_text}

PASSAGES:
{passages}

Write only the new sentences:"""

_WEIGHT_LABEL = {3: "central", 2: "important", 1: "background"}


def _weighted_lines(facets: list, weights: dict, only_ids: set = None) -> str:
    lines = []
    for f in facets:
        if only_ids is not None and f["id"] not in only_ids:
            continue
        w = weights.get(f["id"], 2)
        lines.append(f"- {f['id']} ({f['name']}, weight {w}, {_WEIGHT_LABEL[w]}): {f['why']}")
    return "\n".join(lines)


def _words(answer: list) -> int:
    return sum(len(s["text"].split()) for s in answer)


def _synthesize_multipass(client: OllamaClient, d: Path, topic_id: str, narrative: str,
                          out_name: str, cfg: dict) -> dict:
    """The multipass arm: weight the sub-topics, write, then review and repair up to cfg["passes"] times.

    Same evidence as every other arm (context = anchor top 10 + top 6 per sub-topic, shown as
    best-matching windows). Each review lists the sub-topics still uncovered and those too thin for
    their weight; each repair shows up to four further documents per such sub-topic from its own
    ranking (never one already shown, at most 16 per pass) and appends only new sentences. Stops
    early when the reviewer finds nothing, when no unseen document is left, or at the word cap.
    """
    bp = json.loads((d / "blueprint.json").read_text())
    rankings = json.loads((d / "facet_rankings.json").read_text())
    pool_texts = {e["docid"]: e["text"] for e in read_pool(d / "pool.jsonl")}
    facets = bp["facets"]
    ids = {f["id"] for f in facets}
    context = _select_context(rankings, pool_texts)
    shown = {docid for docid, _ in context}
    terms = _query_terms(narrative, facets)

    def passages(ctx):
        return _passages_block(ctx, cfg["char_per_doc"], terms)

    # 1. weights
    w_raw = client.generate(_WEIGHT_TMPL.format(narrative=narrative[:2000], facet_lines=_facet_lines(facets)),
                            system=_WEIGHT_SYSTEM, think=False, format="json", num_predict=256)
    got = parse_json_object(w_raw).get("weights", {}) or {}
    weights = {}
    for f in facets:
        try:
            weights[f["id"]] = min(3, max(1, int(got.get(f["id"], 2))))
        except (TypeError, ValueError):
            weights[f["id"]] = 2

    # 2. first draft: the evidence arm's call, with the weights in the facet list
    n_facets, scale = len(facets), cfg["sent_scale"]
    system = cfg["system"] + "\n- Give each facet space in proportion to its weight: central facets get the most sentences."
    prompt = _SYNTH_TMPL.format(narrative=narrative[:2000], facet_lines=_weighted_lines(facets, weights),
                                passages=passages(context),
                                n_sent_lo=max(6 * scale, (n_facets * 2 - 2) * scale),
                                n_sent_hi=n_facets * 3 * scale)
    raw = _strip_thinking(client.generate(prompt, system=system, think=False, num_predict=4096))
    answer = _parse_answer(raw, shown, cfg["max_words"], cfg["max_segs"])
    diagnostics = {"n_context_docs": len(context), "n_facets": n_facets, "weights": weights,
                   "first_draft_words": _words(answer), "passes": []}

    # 3. review and repair
    for p in range(1, cfg["passes"] + 1):
        if not answer or _words(answer) >= TOTAL_WORDS:
            break
        answer_text = " ".join(s["text"] for s in answer)
        verdict = parse_json_object(client.generate(
            _REVIEW_TMPL.format(answer_text=answer_text[:6000], facet_lines=_weighted_lines(facets, weights)),
            system=_REVIEW_SYSTEM, think=False, format="json", num_predict=512))
        unc = [u for u in dict.fromkeys(verdict.get("uncovered", []) or []) if u in ids]
        weak = [w for w in dict.fromkeys(verdict.get("weak", []) or [])
                if w in ids and w not in unc and weights.get(w, 2) >= 2]
        targets = sorted(unc + weak, key=lambda f: -weights.get(f, 2))
        rec = {"pass": p, "uncovered": unc, "weak": weak, "new_docs": 0, "new_sentences": 0}
        diagnostics["passes"].append(rec)
        if not targets:
            break
        gap = []
        for fid in targets:
            taken = 0
            for docid, _ in rankings.get(fid, []):
                if taken == 4 or len(gap) >= 16:
                    break
                if docid not in shown and pool_texts.get(docid):
                    gap.append((docid, pool_texts[docid]))
                    shown.add(docid)
                    taken += 1
        rec["new_docs"] = len(gap)
        if not gap:
            break
        raw2 = _strip_thinking(client.generate(
            _REPAIR_MP_TMPL.format(narrative=narrative[:1200],
                                   facet_lines=_weighted_lines(facets, weights, set(targets)),
                                   answer_text=answer_text[:6000], passages=passages(gap)),
            system=cfg["system"], think=False, num_predict=2048))
        extra = _parse_answer(raw2, shown, cfg["max_words"], cfg["max_segs"])
        rec["new_sentences"] = len(extra)
        answer = answer + extra
        rec["words"] = _words(answer)

    if not answer:  # same fallback as synthesize_topic
        fallback_prompt = (
            f"QUERY: {narrative[:400]}\n\n"
            f"PASSAGES:\n{passages(context[:15])}\n\n"
            "Write 5 sentences answering the query. "
            "End each sentence with [docid] citations from the passages."
        )
        raw3 = client.generate(fallback_prompt, system=cfg["system"], think=False, num_predict=2048)
        answer = _parse_answer(_strip_thinking(raw3), shown, cfg["max_words"], cfg["max_segs"])
        diagnostics["fallback_used"] = True

    while answer and _words(answer) > TOTAL_WORDS:
        answer = answer[:-1]
        diagnostics["trimmed_to_cap"] = True

    result = {"topic_id": topic_id, "answer": answer, "diagnostics": diagnostics}
    (d / out_name).write_text(json.dumps(result, indent=2))
    return result


def synthesize_topic(client: OllamaClient, d: Path, topic_id: str,
                     narrative: str, out_name: str = "answer.json", arm: str = "shipped") -> dict:
    """Produce answer.json (or a named variant for ablations) for one topic dir."""
    cfg = ARMS[arm]
    if cfg.get("passes", 1) > 1:
        return _synthesize_multipass(client, d, topic_id, narrative, out_name, cfg)
    bp = json.loads((d / "blueprint.json").read_text())
    rankings = json.loads((d / "facet_rankings.json").read_text())
    pool = read_pool(d / "pool.jsonl")
    pool_texts = {e["docid"]: e["text"] for e in pool}

    context = _select_context(rankings, pool_texts)
    valid_ids = {docid for docid, _ in context}
    n_facets = len(bp["facets"])
    terms = _query_terms(narrative, bp["facets"]) if cfg["best_window"] else None

    def passages(ctx):
        return _passages_block(ctx, cfg["char_per_doc"], terms)

    # --- Call 1: scaffolded synthesis ---
    scale = cfg["sent_scale"]
    prompt = _SYNTH_TMPL.format(
        narrative=narrative[:2000],
        facet_lines=_facet_lines(bp["facets"]),
        passages=passages(context),
        n_sent_lo=max(6 * scale, (n_facets * 2 - 2) * scale),
        n_sent_hi=n_facets * 3 * scale,
    )
    raw = client.generate(prompt, system=cfg["system"], think=False,
                          num_predict=4096)
    raw = _strip_thinking(raw)
    answer = _parse_answer(raw, valid_ids, cfg["max_words"], cfg["max_segs"])

    diagnostics = {"n_context_docs": len(context), "gap_round": False,
                   "uncovered_initial": [], "n_facets": n_facets}

    # --- Call 2: coverage self-check ---
    if answer:
        answer_text = " ".join(s["text"] for s in answer)
        cov_prompt = _COVERAGE_TMPL.format(
            answer_text=answer_text[:4000],
            facet_lines=_facet_lines(bp["facets"]),
        )
        cov_raw = client.generate(cov_prompt, system=_COVERAGE_SYSTEM,
                                  think=False, format="json", num_predict=512)
        verdict = parse_json_object(cov_raw)
        uncovered = [u for u in verdict.get("uncovered", [])
                     if any(f["id"] == u for f in bp["facets"])]
        diagnostics["uncovered_initial"] = uncovered

        # --- Optional single gap round ---
        if uncovered:
            diagnostics["gap_round"] = True
            gap_ids = set(uncovered)
            gap_context = []
            seen = valid_ids.copy()
            for fid in uncovered:
                for docid, _ in rankings.get(fid, [])[:10]:
                    if docid not in seen and pool_texts.get(docid):
                        seen.add(docid)
                        gap_context.append((docid, pool_texts[docid]))
            gap_context = gap_context[:15]
            if gap_context:
                repair_prompt = _REPAIR_TMPL.format(
                    narrative=narrative[:1200],
                    facet_lines=_facet_lines(bp["facets"], only_ids=gap_ids),
                    passages=passages(gap_context),
                )
                raw2 = client.generate(repair_prompt, system=cfg["system"],
                                       think=False, num_predict=2048)
                raw2 = _strip_thinking(raw2)
                extra_valid = valid_ids | {docid for docid, _ in gap_context}
                extra = _parse_answer(raw2, extra_valid, cfg["max_words"], cfg["max_segs"])
                answer = answer + extra

    # --- Fallback if everything failed ---
    if not answer:
        fallback_prompt = (
            f"QUERY: {narrative[:400]}\n\n"
            f"PASSAGES:\n{passages(context[:15])}\n\n"
            "Write 5 sentences answering the query. "
            "End each sentence with [docid] citations from the passages."
        )
        raw3 = client.generate(fallback_prompt, system=cfg["system"], think=False,
                               num_predict=2048)
        answer = _parse_answer(_strip_thinking(raw3), valid_ids, cfg["max_words"], cfg["max_segs"])
        diagnostics["fallback_used"] = True

    # --- Official cap: the whole answer, gap-round sentences included ---
    while answer and sum(len(s["text"].split()) for s in answer) > TOTAL_WORDS:
        answer = answer[:-1]
        diagnostics["trimmed_to_cap"] = True

    result = {"topic_id": topic_id, "answer": answer, "diagnostics": diagnostics}
    (d / out_name).write_text(json.dumps(result, indent=2))
    return result
