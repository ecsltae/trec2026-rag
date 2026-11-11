"""
Checks requested by the review of 2026-11-10, each scored by ONE judge so that every list is fully
labelled (no unjudged documents, so no lower/upper-bound arguments).

    source ~/TREC/venv/bin/activate
    python eval/analysis/review_checks_2026_11_10.py dev22-label      # GPU: qwen3:32b, ~15 min
    python eval/analysis/review_checks_2026_11_10.py depth-label      # GPU: qwen3:32b, ~1.6 h
    python eval/analysis/review_checks_2026_11_10.py support-shown    # GPU: qwen3:32b, ~40 min
    python eval/analysis/review_checks_2026_11_10.py report           # CPU; writes the .json next to it

dev22-label    Table 1's rows, top ten each, on the 22 development topics, all labelled by our
               relevance labeller (the official UMBRELA "bing" prompt on qwen3:32b, the same judge
               and cache as our official119 labels). Reported: precision at ten (grade >= 2) and
               paired differences from anchor-first, with standard errors.
depth-label    The counterfactual for "the facet rankings earned the recall": on the 82 narratives
               the organisers judged, our scored run against the anchor-first ranking cut at the
               same depth per narrative (the whole-topic ranking down to rank 200), both fully
               labelled by the same labeller. Reported: relevant documents per narrative, paired.
support-shown  The support judge (RAGDoll's prompt on qwen3:32b) re-run against the passage the
               writer was shown: the first 700 characters for the submitted writer and the
               coverage-prompt variant, the best-matching 1,000-character window for the
               best-passages and multi-pass variants. The original judge read the first 4,000
               characters, which a best-matching window can lie beyond. Writes
               runs/rag/<run>.support_shown.jsonl and the filtered runs <run>_shown_grounded.jsonl
               (sentences with at least one fully supporting citation), for the nugget judge.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "eval/RAGDoll/src"))
sys.path.insert(0, str(Path(__file__).parent))

DEV = ROOT / "data/rag/cache/agentic_v2/dev22"
LABELS = ROOT / "data/rag/cache/umbrela_labels.jsonl"
MODEL, PTYPE = "qwen3:32b", "bing"
OFF = ROOT / "data/rag/official_scores/retrieval_2026-10-29/Bitem/retrieval-document-level-scores.tsv"
OUT = Path(__file__).with_suffix(".json")
RUNS = ROOT / "runs/rag"
SHOWN = {"bp_agentic_v2_dev22": "head700", "bp_v2_coverage_dev22": "head700",
         "bp_v2_evidence_dev22": "best1000", "bp_v2_multipass_dev22": "best1000"}


def load_labels():
    lab = {}
    if LABELS.exists():
        for line in open(LABELS):
            r = json.loads(line)
            lab[r["key"]] = r["grade"]
    return lab


def key(t, d):
    return f"{t}|{d}|{MODEL}|{PTYPE}"


def label_pairs(pairs, query_of, text_of, tag):
    """Label (topic, docid) pairs with the UMBRELA prompt, appending to the shared cache."""
    from ragdoll.umbrela.prompts import render_umbrela_prompt, parse_umbrela_judgment
    from trec_shared.ollama_client import OllamaClient
    lab = load_labels()
    todo = [(t, d) for t, d in pairs if key(t, d) not in lab]
    print(f"[{tag}] {len(pairs)} pairs, {len(todo)} to label", flush=True)
    client = OllamaClient(model=MODEL, temperature=0.0)
    missing = 0
    with open(LABELS, "a") as f:
        for i, (t, d) in enumerate(todo, 1):
            text = text_of(t, d)
            if not text.strip():
                missing += 1
                continue
            raw = client.generate(render_umbrela_prompt(query=query_of(t), passage=text[:4000],
                                                        prompt_type=PTYPE), think=False, num_predict=64)
            g = parse_umbrela_judgment(raw)
            f.write(json.dumps({"key": key(t, d), "topic_id": t, "docid": d,
                                "grade": 0 if g is None else g}) + "\n")
            f.flush()
            if i % 200 == 0:
                print(f"[{tag}] {i}/{len(todo)}", flush=True)
    print(f"[{tag}] done; {missing} documents without text", flush=True)


# ---------------------------------------------------------------- dev22: Table 1 rows, top ten
def dev22_lists():
    import fusion_judged_share as F
    T = F.T
    return T, {
        "anchor-first": {t: [d for d, _ in json.loads((DEV / t / "final_ranking.json").read_text())][:10] for t in T},
        "RRF, whole topic x2": {t: F.rrf(t, 2)[:10] for t in T},
        "RRF, whole topic x8": {t: F.rrf(t, 8)[:10] for t in T},
        "cross-encoder alone": {t: v[:10] for t, v in F.cross_encoder_alone().items()},
        "naive sub-questions": {t: v[:10] for t, v in F.run("agentic_iterative_sq_v1_full105").items()},
        "plain BM25": {t: v[:10] for t, v in F.run("api_climbmix_eval22_fullnarrative").items()},
        "anchor-first, bge reranker": {t: [d for d, _ in json.loads(
            (DEV / t / "facet_rankings_bge.json").read_text())["_anchor"]][:10] for t in T},
    }


def dev22_text_of():
    """Text from every local pool cache, then from the search API (needs PYSERINI_API_TOKEN) for
    the few top-ten documents of the sub-question run that no cache holds."""
    from eval.ragdoll_local_judge import build_text_index
    index = {}
    for c in sorted((ROOT / "data/rag/cache/agentic_v2").iterdir()):
        index.update(build_text_index(c))
    api = {}

    def text_of(t, d):
        if d in index:
            return index[d]
        if "r" not in api:
            from systems.rag.retrieve import ApiRetriever
            api["r"] = ApiRetriever()
        return api["r"].get_text(d) or ""
    return text_of


def dev22_label():
    from systems.rag.agentic_v2.pipeline import load_topic_set
    topics = load_topic_set("dev22")
    T, lists = dev22_lists()
    pairs = sorted({(t, d) for L in lists.values() for t in T for d in L[t]})
    label_pairs(pairs, lambda t: topics[t].get("narrative", ""), dev22_text_of(), "dev22")


# ---------------------------------------------------------------- official119: depth-matched
def runfile(p):
    d = defaultdict(list)
    for line in open(p):
        t, _, x, *_ = line.split()
        d[t].append(x)
    return d


def depth_lists():
    judged = sorted({r["narrative-id"] for r in csv.DictReader(open(OFF), delimiter="\t")
                     if r["narrative-id"] != "all"})
    sc = runfile(ROOT / "submissions/bp_agentic_v2/r_output_trec_rag_2026.tsv")
    deep = runfile(ROOT / "submissions/BiTeM_deep_v2/r_output_trec_rag_2026.tsv")
    return judged, {t: sc[t] for t in judged}, {t: deep[t][:len(sc[t])] for t in judged}


def depth_label():
    from trec_shared.trec_utils import load_topics
    from eval.ragdoll_local_judge import build_text_index
    topics = load_topics(str(ROOT / "data/rag/topics/trec_rag_2026_queries.tsv"))
    index = {}
    for c in ("official119", "merged_official119", "official119_probe"):
        index.update(build_text_index(ROOT / "data/rag/cache/agentic_v2" / c))
    judged, sc, af = depth_lists()
    pairs = sorted({(t, d) for t in judged for d in sc[t] + af[t]})
    label_pairs(pairs, lambda t: topics[t]["narrative"], lambda t, d: index.get(d, ""), "depth")


# ---------------------------------------------------------------- support against the shown passage
def support_shown():
    from ragdoll.support.prompts import render_support_prompt, parse_support_label
    from trec_shared.ollama_client import OllamaClient
    from eval.ragdoll_local_judge import cache_key, load_judge_cache, JUDGE_CACHE, LABEL_TO_SCORE
    from systems.rag.agentic_v2.harvest import read_pool
    from systems.rag.agentic_v2.synthesize import _best_window, _query_terms
    from systems.rag.agentic_v2.pipeline import load_topic_set
    topics = load_topic_set("dev22")
    client = OllamaClient(model=MODEL, temperature=0.0)
    cache = load_judge_cache()
    for run, mode in SHOWN.items():
        rows, new = [], 0
        for line in open(RUNS / f"{run}.jsonl"):
            obj = json.loads(line)
            t = str(obj["metadata"]["topic_id"])
            texts = {e["docid"]: e["text"] for e in read_pool(DEV / t / "pool.jsonl")}
            bp = json.loads((DEV / t / "blueprint.json").read_text())
            terms = _query_terms(topics[t].get("narrative", ""), bp["facets"])
            refs = obj["references"]
            sents = []
            for s in obj["answer"]:
                cits = []
                for c in s["citations"]:
                    d = refs[c] if isinstance(c, int) else c
                    full = texts.get(d, "")
                    shown = full[:700] if mode == "head700" else _best_window(full, terms, 1000)
                    k = cache_key(s["text"], shown, MODEL, 4000)
                    if k not in cache:
                        lab = parse_support_label(client.generate(
                            render_support_prompt(statement=s["text"], citation=shown), think=False, num_predict=16))
                        cache[k] = LABEL_TO_SCORE.get(lab, 0)
                        with open(JUDGE_CACHE, "a") as f:
                            f.write(json.dumps({"key": k, "support": cache[k]}) + "\n")
                        new += 1
                    cits.append({"docid": d, "support": cache[k]})
                sents.append({"text": s["text"], "citations": cits})
            rows.append({"topic_id": t, "run_id": run, "sentences": sents})
        with open(RUNS / f"{run}.support_shown.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # filtered copy for the nugget judge: sentences with >= 1 fully supporting citation
        keep = {(r["topic_id"], i) for r in rows for i, s in enumerate(r["sentences"])
                if any(c["support"] == 2 for c in s["citations"])}
        with open(RUNS / f"{run}_shown_grounded.jsonl", "w") as f:
            for line in open(RUNS / f"{run}.jsonl"):
                obj = json.loads(line)
                t = str(obj["metadata"]["topic_id"])
                obj["answer"] = [s for i, s in enumerate(obj["answer"]) if (t, i) in keep]
                f.write(json.dumps(obj) + "\n")
        print(f"[support-shown] {run} ({mode}): {new} new judgements", flush=True)


# ---------------------------------------------------------------- report
def paired(a, b):
    d = [y - x for x, y in zip(a, b)]
    se = st.stdev(d) / math.sqrt(len(d))
    return st.mean(d), se, sum(x > 0 for x in d), sum(x < 0 for x in d)


def sign_p(w, l):
    n, k = w + l, min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0


def report():
    lab = load_labels()
    res = {}
    # dev22
    T, lists = dev22_lists()
    g = lambda t, d: lab.get(key(t, d))
    unl = sum(g(t, d) is None for L in lists.values() for t in T for d in L[t])
    p10 = {n: [sum((g(t, d) or 0) >= 2 for d in L[t]) / 10 for t in T] for n, L in lists.items()}
    for n, L in lists.items():
        miss = sum(g(t, d) is None for t in T for d in L[t])
        if miss:
            print(f"  {n}: {miss} of {10 * len(T)} top-ten documents without text, counted as not relevant")
    res["dev22_p10_ourlabels"] = {"unlabelled": unl}
    print(f"\nTable 1 rows, top ten, all labelled by our labeller ({unl} unlabelled):")
    for n, v in p10.items():
        m, se, w, l = paired(p10["anchor-first"], v)
        res["dev22_p10_ourlabels"][n] = {"P@10": st.mean(v), "minus_anchor_first": m, "se": se, "up": w, "down": l}
        print(f"  {n:28s} P@10 {st.mean(v):.3f}   row minus anchor-first {m:+.3f} (SE {se:.3f}, {w} up, {l} down, sign p {sign_p(w, l):.3f})")
    # depth-matched
    judged, sc, af = depth_lists()
    unl = sum(g(t, d) is None for t in judged for d in sc[t] + af[t])
    hs = [sum((g(t, d) or 0) >= 2 for d in sc[t]) for t in judged]
    ha = [sum((g(t, d) or 0) >= 2 for d in af[t]) for t in judged]
    m, se, w, l = paired(ha, hs)
    ps = st.mean(h / len(sc[t]) for h, t in zip(hs, judged)); pa = st.mean(h / len(af[t]) for h, t in zip(ha, judged))
    over = st.mean(len(set(sc[t]) & set(af[t])) / len(sc[t]) for t in judged)
    res["depth_matched"] = {"narratives": len(judged), "unlabelled": unl, "rel_scored_mean": st.mean(hs),
                            "rel_anchorfirst_mean": st.mean(ha), "diff": m, "se": se, "up": w, "down": l,
                            "P_scored": ps, "P_anchorfirst": pa, "overlap": over}
    print(f"\nDepth-matched, {len(judged)} judged narratives ({unl} unlabelled): relevant per narrative "
          f"scored {st.mean(hs):.1f} vs anchor-first cut {st.mean(ha):.1f}; scored minus cut {m:+.1f} "
          f"(SE {se:.1f}; {w} up, {l} down, sign p {sign_p(w, l):.4f}); precision {ps:.3f} vs {pa:.3f}; overlap {over:.0%}")
    # support against the shown passage
    res["support_shown"] = {}
    print("\nSentences with >= 1 fully supporting citation (original judge: first 4,000 chars | shown passage):")
    for run in SHOWN:
        def share(path):
            if not path.exists():
                return None
            s = [x for r in map(json.loads, open(path)) for x in r["sentences"]]
            return sum(any(c["support"] == 2 for c in x["citations"]) for x in s) / len(s)
        a, b = share(RUNS / f"{run}.support_judgments.jsonl"), share(RUNS / f"{run}.support_shown.jsonl")
        res["support_shown"][run] = {"original": a, "shown": b}
        print(f"  {run:26s} {a if a is None else round(a, 3)} | {b if b is None else round(b, 3)}")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    {"dev22-label": dev22_label, "depth-label": depth_label, "support-shown": support_shown,
     "report": report}[sys.argv[1]]()
