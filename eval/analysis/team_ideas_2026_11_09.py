"""
The team's ideas from the lab presentation of 9 November 2026, recomputed from what is on disk.

    source ~/TREC/venv/bin/activate
    python eval/analysis/team_ideas_2026_11_09.py

Prints one block per idea and writes team_ideas_2026_11_09.json next to this file. Notes and
reading of the numbers: IDEAS_team_2026-11-09.md.

  1  evaluate at other ranks: dev22, P@k / R@k / nDCG@k for k = 5 ... 1000, and the set measures at
     the depth our ce_floor rule picks, as in the official scoring
  2  several passes that review what is still uncovered: how often the shipped single review round
     fired, and what the systems that already do more than one pass scored on the dev nuggets
  3  topics that are too mainstream: per-narrative official scores against how many relevant
     documents the official pool holds for that narrative (exploratory)
  4  is a ~1,900-document pool small: how many of the dev qrels' relevant documents ever enter it
  5  sub-topic generation: what the sub-topic lists add that the whole-question list does not,
     on dev22 (qrels) and in the submitted lists (our own labels)
  6  re-ranking at its stage: every list before and after the MiniLM re-ranker, against the dev
     qrels, with the sub-topic queries or the whole question as the re-ranking query

dev22 = the organisers' 22 RAG25 dev topics, qrels v2 (qwen3.5-9b, grades 0-4; v1 alongside).
"Relevant" is grade >= 2, as in the official set precision and recall. Unjudged = not relevant,
so the J@k columns (share of the top k that the qrels judged at all) say where that matters, and
the primed columns (P'@k, nDCG'@k) score the condensed list, unjudged documents removed (Sakai 2007).
The dev qrels were pooled from whole-question runs, so lists built from sub-topic queries are
mostly unjudged: read their unprimed scores as lower bounds and their primed ones as upper bounds.

Only block 6 loads models (bge-large-en-v1.5 for the sub-topic queries, the MiniLM-L6 re-ranker),
about a minute on the A100. It replays systems/rag/agentic_v2/tighten.py and prints how closely
the replay matches the cached rankings before using it.
"""
import csv
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from eval_rag import ndcg_at_k
from trec_shared.trec_utils import read_trec_run, load_qrels
from systems.rag.agentic_v2.harvest import read_pool
from systems.rag.agentic_v2.pipeline import load_topic_set

CACHE = ROOT / "data/rag/cache/agentic_v2"
V1 = ROOT / "data/rag/qrels/rag25-climbmix-umbrela-qwen3.5-9b.qrels"
V2 = ROOT / "data/rag/qrels_v2/rag25-climbmix-umbrela-qwen3.5-9b-v2.qrels"
OFF_RET = ROOT / "data/rag/official_scores/retrieval_2026-10-29/Bitem/retrieval-document-level-scores.tsv"
OFF_GEN = ROOT / "data/rag/official_scores/generation_2026-11-03/Bitem"
LABELS = ROOT / "data/rag/cache/umbrela_labels.jsonl"
OUT = Path(__file__).with_suffix(".json")
KS = (5, 10, 20, 50, 100, 200, 500, 1000)
REL = 2
RUNS = {  # dev22 run files, as in paper_revision_checks.py
    "anchor-first (shipped)": "bp_agentic_v2_dev22",
    "whole question": "api_bm25_reranked_full105",
    "plain BM25": "api_climbmix_eval22_fullnarrative",
    "naive sub-questions": "agentic_iterative_sq_v1_full105",
    "probe-guided": "bp_agentic_v2_dev22_probe",
}
RESULT = {}


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def load(p):
    return json.loads(Path(p).read_text())


def rel_of(q, tid):
    return {d for d, g in q.get(tid, {}).items() if g >= REL}


def p_at(ranked, rel, k):
    return sum(d in rel for d in ranked[:k]) / k


def r_at(ranked, rel, k):
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def j_at(ranked, q, k):
    top = ranked[:k]
    return sum(d in q for d in top) / len(top) if top else 0.0


def condensed(ranked, q):
    """Sakai's condensed list: the ranking with the unjudged documents removed."""
    return [d for d in ranked if d in q]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def best_ce(rankings):
    """docid -> best fused re-ranker score over all lists, exactly as build_variable_depth_runs.py."""
    best = {}
    for lst in rankings.values():
        for d, s in lst:
            if d not in best or s > best[d]:
                best[d] = s
    return best


def cited(d, name="answer.json"):
    a = load(d / name)
    return list(dict.fromkeys(c for s in a.get("answer", []) for c in s.get("citations", [])))


def ce_floor(ce, refs):
    """The submitted depth rule: every document at least as high as the weakest cited one."""
    cs = [ce[x] for x in refs if x in ce]
    if not cs:
        return list(refs)
    keep = {x: s for x, s in ce.items() if s >= min(cs)}
    for x in refs:
        keep.setdefault(x, ce.get(x, min(cs)))
    return [x for x, _ in sorted(keep.items(), key=lambda x: -x[1])]


def spearman(xs, ys):
    from scipy.stats import spearmanr
    r = spearmanr(xs, ys)
    return {"rho": round(float(r.statistic), 3), "p": round(float(r.pvalue), 3), "n": len(xs)}


def show(title, header, rows):
    print(f"\n{title}")
    w = max(len(r[0]) for r in rows) + 2
    print(" " * w + "".join(f"{h:>9s}" for h in header))
    for r in rows:
        print(f"{r[0]:<{w}s}" + "".join(f"{v:9.3f}" if isinstance(v, float) else f"{str(v):>9s}" for v in r[1:]))


QV1, QV2 = load_qrels(str(V1)), load_qrels(str(V2))
DEV = sorted(QV2)
TOPICS = load_topic_set("dev22")


def dev_dir(tid, probe=False):
    return CACHE / ("dev22_probe" if probe else "dev22") / tid


# ---------------------------------------------------------------------------------------------
# 1. other ranks
# ---------------------------------------------------------------------------------------------

def best_ce_order(d):
    """The order the submitted lists use (best fused score over all lists), padded with the pool."""
    rk = load(d / "facet_rankings.json")
    ce = best_ce(rk)
    order = [x for x, _ in sorted(ce.items(), key=lambda x: -x[1])]
    seen = set(order)
    order += [e["docid"] for e in read_pool(d / "pool.jsonl") if e["docid"] not in seen]
    return order[:1000], ce


def idea1():
    runs = {}
    for label, name in RUNS.items():
        r = read_trec_run(str(ROOT / "runs/rag" / f"{name}.tsv"))
        runs[label] = {t: [d for d, _, _ in r[t]] for t in DEV if t in r}
    runs["best-score order (submitted)"] = {t: best_ce_order(dev_dir(t))[0] for t in DEV}

    out = {}
    for qname, q in (("v2", QV2), ("v1", QV1)):
        out[qname] = {}
        for label, rr in runs.items():
            m = {}
            for k in KS:
                m[f"P@{k}"] = mean(p_at(rr[t], rel_of(q, t), k) for t in DEV if t in rr)
                m[f"R@{k}"] = mean(r_at(rr[t], rel_of(q, t), k) for t in DEV if t in rr)
                m[f"nDCG@{k}"] = mean(ndcg_at_k(rr[t], q[t], k) for t in DEV if t in rr)
                m[f"J@{k}"] = mean(j_at(rr[t], q[t], k) for t in DEV if t in rr)
                m[f"P'@{k}"] = mean(p_at(condensed(rr[t], q[t]), rel_of(q, t), k) for t in DEV if t in rr)
                m[f"nDCG'@{k}"] = mean(ndcg_at_k(condensed(rr[t], q[t]), q[t], k) for t in DEV if t in rr)
            m["topics"] = sum(t in rr for t in DEV)
            out[qname][label] = m
    for metric in ("nDCG", "P", "R", "J", "nDCG'", "P'"):
        show(f"1. dev22, qrels v2, {metric}@k (relevant = grade >= 2; J = share judged; ' = condensed list)",
             [str(k) for k in KS],
             [[label] + [out["v2"][label][f"{metric}@{k}"] for k in KS] for label in runs])

    # The set measures at the depth our rule picks, against fixed depths on the same order.
    depth, sp, sr = [], [], []
    for t in DEV:
        _, ce = best_ce_order(dev_dir(t))
        lst = ce_floor(ce, cited(dev_dir(t)))
        rel = rel_of(QV2, t)
        depth.append(len(lst))
        sp.append(sum(d in rel for d in lst) / len(lst))
        sr.append(len(set(lst) & rel) / len(rel) if rel else 0.0)
    order = runs["best-score order (submitted)"]
    fixed = {k: (mean(p_at(order[t], rel_of(QV2, t), k) for t in DEV),
                 mean(r_at(order[t], rel_of(QV2, t), k) for t in DEV)) for k in (10, 20, 50, 100, 200)}
    out["ce_floor_dev22"] = {"median_depth": st.median(depth), "range": [min(depth), max(depth)],
                             "set_P": mean(sp), "set_R": mean(sr),
                             "fixed_depth_same_order": {str(k): {"P": p, "R": r} for k, (p, r) in fixed.items()}}
    print(f"\n   ce_floor depth on dev22: median {st.median(depth)} (range {min(depth)}-{max(depth)})"
          f"   set P {mean(sp):.3f}   set R {mean(sr):.3f}")
    print("   same order, fixed depth: " + "   ".join(f"@{k} P {p:.3f} R {r:.3f}" for k, (p, r) in fixed.items()))
    RESULT["1_ranks"] = out


# ---------------------------------------------------------------------------------------------
# 2. several review passes
# ---------------------------------------------------------------------------------------------

def paired(a, b):
    """Mean paired difference b - a and its standard error over the topics both have."""
    ts = sorted(set(a) & set(b))
    d = [b[t] - a[t] for t in ts]
    se = st.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
    return mean(d), se, sum(x > 0 for x in d), sum(x < 0 for x in d)


def idea2():
    out = {"review_round": {}, "nuggets": {}}
    print("\n2. the shipped writer reviews its draft once (uncovered sub-topics -> one repair round)")
    for label, cache, name in (("dev22, shipped", "dev22", "answer.json"),
                               ("dev22, evidence arm", "dev22", "answer_evidence.json"),
                               ("official119, primary", "official119", "answer.json"),
                               ("official119, probe", "official119_probe", "answer.json")):
        fired, unc, nf = 0, [], []
        dirs = [d for d in sorted((CACHE / cache).iterdir()) if (d / name).exists()]
        for d in dirs:
            g = load(d / name).get("diagnostics", {})
            fired += bool(g.get("gap_round"))
            unc.append(len(g.get("uncovered_initial", [])))
            nf.append(g.get("n_facets", 0))
        out["review_round"][label] = {"topics": len(dirs), "repair_round_fired": fired,
                                      "mean_uncovered": mean(unc), "mean_subtopics": mean(nf)}
        print(f"   {label:24s} {len(dirs):3d} topics: repair round fired on {fired:3d}"
              f" ({fired / len(dirs):.0%}); mean uncovered {mean(unc):.2f} of {mean(nf):.1f} sub-topics")

    runs = {}
    for f in ("nugget_dev22_existing.json", "nugget_dev22_arms.json"):
        runs.update(load(Path(__file__).parent / f)["runs"])
    base = {t: v["strict_all_score"] for t, v in runs["bp_agentic_v2_dev22"]["per_topic"].items()}
    print("   dev22 official nuggets, our local judge (qwen3:32b, RAGDoll assigner prompt), all_strict:")
    for name, v in runs.items():
        pt = {t: x["strict_all_score"] for t, x in v["per_topic"].items()}
        dm, se, w, l = paired(base, pt)
        out["nuggets"][name] = {"all_strict": v["strict_all_score"], "vital_strict": v["strict_vital_score"],
                                "median_words": v["median_words"], "delta_vs_shipped": dm, "se": se,
                                "wins": w, "losses": l}
        print(f"   {name:28s} {v['strict_all_score']:.4f}  vital {v['strict_vital_score']:.4f}"
              f"  words {v['median_words']:6.1f}  vs shipped {dm:+.4f} (SE {se:.4f}, {w}-{l})")
    RESULT["2_passes"] = out


# ---------------------------------------------------------------------------------------------
# 3. mainstream topics
# ---------------------------------------------------------------------------------------------

def idea3():
    rows = [r for r in csv.DictReader(open(OFF_RET), delimiter="\t") if r["narrative-id"] != "all"]
    by = defaultdict(dict)
    for r in rows:
        by[r["narrative-id"]][r["run"]] = r
    qlen = {l.split("\t")[0]: len(l.split("\t")[1].split())
            for l in open(ROOT / "data/rag/topics/trec_rag_2026_queries.tsv") if l.strip()}
    pool_size = {d.name: sum(1 for _ in open(d / "pool.jsonl"))
                 for d in (CACHE / "official119").iterdir() if (d / "pool.jsonl").exists()}
    prim, probe = "bp_agentic_retrieval", "bp_probe_ret"
    n_rel, x = {}, defaultdict(dict)
    for t, rr in by.items():
        a = rr[prim]
        P, R, k = float(a["set_P"]), float(a["set_recall"]), int(a["num_ret"])
        if R > 0:
            n_rel[t] = P * k / R
        x["primary P - median"][t] = P - float(a["set_P_median"])
        x["primary R - median"][t] = R - float(a["set_recall_median"])
        x["primary P"][t] = P
        x["primary R"][t] = R
        if probe in rr:
            x["probe - primary, P"][t] = float(rr[probe]["set_P"]) - P
            x["probe - primary, R"][t] = float(rr[probe]["set_recall"]) - R
        x["question length (words)"][t] = qlen[t]
        x["our pool size"][t] = pool_size.get(t, float("nan"))
    # sanity: the probe row implies the same judged pool when its recall is non-zero
    agree = [abs(float(rr[probe]["set_P"]) * int(rr[probe]["num_ret"]) / float(rr[probe]["set_recall"]) - n_rel[t])
             for t, rr in by.items() if t in n_rel and probe in rr and float(rr[probe]["set_recall"]) > 0]
    for fname, label in (("rag-nugget-scores-batch2-automatic.tsv", "coverage - median (batch 2, auto)"),
                         ("rag-rubric-scores-batch1-human.tsv", "coverage - median (batch 1, human)")):
        for r in csv.DictReader(open(OFF_GEN / fname), delimiter="\t"):
            if r["run"] == "bp_agentic_v2" and r["narrative-id"] != "all":
                x[label][r["narrative-id"]] = float(r["all_strict"]) - float(r["all_strict_median"])
    out = {"relevant_in_official_pool": {"median": st.median(n_rel.values()), "min": min(n_rel.values()),
                                         "max": max(n_rel.values()), "n": len(n_rel)},
           "probe_row_consistency_max_abs_diff": max(agree) if agree else None, "spearman": {}}
    print(f"\n3. 'mainstream' proxy = relevant documents in the official pool for the narrative (P*k/R):"
          f" median {st.median(n_rel.values()):.0f}, range {min(n_rel.values()):.0f}-{max(n_rel.values()):.0f},"
          f" n {len(n_rel)} (probe rows agree within {max(agree):.2f})")
    for label, ys in x.items():
        ts = [t for t in n_rel if t in ys and not math.isnan(ys[t])]
        s = spearman([n_rel[t] for t in ts], [ys[t] for t in ts])
        out["spearman"][label] = s
        print(f"   Spearman with the proxy: {label:36s} rho {s['rho']:+.3f}  p {s['p']:.3f}  n {s['n']}")
    # the top and bottom thirds, for the two head-to-head differences
    ts = sorted(n_rel, key=n_rel.get)
    third = len(ts) // 3
    for label in ("probe - primary, P", "probe - primary, R", "primary R - median"):
        lo = mean(x[label][t] for t in ts[:third] if t in x[label])
        hi = mean(x[label][t] for t in ts[-third:] if t in x[label])
        out[f"thirds: {label}"] = {"fewest_relevant": lo, "most_relevant": hi}
        print(f"   {label:24s} narratives with the fewest relevant docs {lo:+.3f}, with the most {hi:+.3f}")
    RESULT["3_mainstream"] = out


# ---------------------------------------------------------------------------------------------
# 4. pool size, and 5. sub-topics (dev22 with qrels, official119 with our labels)
# ---------------------------------------------------------------------------------------------

def is_facet_source(s):
    return s not in ("narrative", "anchor")


def idea4_5():
    out4, out5 = defaultdict(list), defaultdict(list)
    per_facet = []
    for t in DEV:
        d = dev_dir(t)
        rel = rel_of(QV2, t)
        judged = set(QV2[t])
        pool = read_pool(d / "pool.jsonl")
        pool_ids = {e["docid"] for e in pool}
        rk = load(d / "facet_rankings.json")
        anchor = {x for x, _ in rk["_anchor"]}
        facets = {f: {x for x, _ in lst} for f, lst in rk.items() if f != "_anchor"}
        lists = anchor.union(*facets.values())
        final = [x for x, _ in load(d / "final_ranking.json")]
        ctx_anchor = [x for x, _ in rk["_anchor"][:10]]
        ctx = list(dict.fromkeys(ctx_anchor + [x for f, lst in rk.items() if f != "_anchor" for x, _ in lst[:6]]))
        refs = cited(d)
        # 4: how much of the relevant set ever reaches each stage
        out4["relevant"].append(len(rel))
        out4["pool"].append(len(pool_ids))
        out4["in pool"].append(len(rel & pool_ids) / len(rel))
        out4["in the re-ranked lists"].append(len(rel & lists) / len(rel))
        out4["in final top 100"].append(len(rel & set(final[:100])) / len(rel))
        out4["in final top 1000"].append(len(rel & set(final)) / len(rel))
        out4["relevant in pool, lost before the lists"].append(len((rel & pool_ids) - lists) / max(1, len(rel & pool_ids)))
        out4["pool judged at all"].append(len(pool_ids & judged) / len(pool_ids))
        out4["judged, in pool"].append(len(pool_ids & judged) / len(judged))
        # 5: what the sub-topic searches and lists add
        only_facet_src = {e["docid"] for e in pool if all(is_facet_source(s) for s in e["sources"])}
        out5["relevant in pool, found only by sub-topic searches"].append(
            len(rel & only_facet_src) / max(1, len(rel & pool_ids)))
        out5["relevant in lists, only in sub-topic lists"].append(
            len(rel & (lists - anchor)) / max(1, len(rel & lists)))
        out5["sub-topic-only documents, relevant"].append(p_set(lists - anchor, rel))
        out5["whole-question list (200), relevant"].append(p_set(anchor, rel))
        out5["context from whole question (10), relevant"].append(p_set(set(ctx_anchor), rel))
        out5["context from sub-topics, relevant"].append(p_set(set(ctx) - set(ctx_anchor), rel))
        out5["cited, from sub-topic lists only"].append(sum(x not in anchor for x in refs) / max(1, len(refs)))
        jac = [len(a & b) / len(a | b) for i, a in enumerate(facets.values())
               for j, b in enumerate(facets.values()) if i < j]
        out5["sub-topic lists, mean pairwise overlap (Jaccard)"].append(mean(jac))
        for f, s in facets.items():
            others = anchor.union(*(v for g, v in facets.items() if g != f))
            per_facet.append({"topic": t, "facet": f, "P@100": len(s & rel) / len(s),
                              "unique_relevant": len((s & rel) - others)})
    print("\n4. does the ~1,900-document pool hold the relevant documents? dev22, qrels v2, grade >= 2"
          "\n   (share of each topic's relevant documents; mean / median over the 22 topics)")
    for k, v in out4.items():
        print(f"   {k:42s} mean {mean(v):8.3f}   median {st.median(v):8.3f}")
    print("\n5. what the sub-topics add, dev22 (shares; mean / median over topics)")
    for k, v in out5.items():
        print(f"   {k:52s} mean {mean(v):.3f}   median {st.median(v):.3f}")
    zero = sum(f["P@100"] == 0 for f in per_facet)
    no_unique = sum(f["unique_relevant"] == 0 for f in per_facet)
    print(f"   {len(per_facet)} sub-topic lists: {zero} hold no relevant document, {no_unique} add no relevant"
          f" document that the whole-question list or another sub-topic does not already have;"
          f" P@100 median {st.median(f['P@100'] for f in per_facet):.2f}"
          f" (range {min(f['P@100'] for f in per_facet):.2f}-{max(f['P@100'] for f in per_facet):.2f})")

    # official119: the submitted lists, by origin, scored with our own (lenient) labels
    lab = {}
    for line in open(LABELS):
        r = json.loads(line)
        lab[(r["topic_id"], r["docid"])] = r["grade"]
    sub5 = {}
    for sub, cache in (("bp_agentic_v2", "official119"), ("bp_probe_guided", "official119_probe")):
        lists = defaultdict(list)
        for line in open(ROOT / "submissions" / sub / "r_output_trec_rag_2026.tsv"):
            t, _, d, *_ = line.split()
            lists[t].append(d)
        share, rate_a, rate_f, n_a, n_f = [], [], [], 0, 0
        for t, lst in lists.items():
            anchor = {x for x, _ in load(CACHE / cache / t / "facet_rankings.json")["_anchor"]}
            fo = [x for x in lst if x not in anchor]
            share.append(len(fo) / len(lst))
            for x in lst:
                g = lab.get((t, x))
                if g is None:
                    continue
                (rate_a if x in anchor else rate_f).append(g >= REL)
        sub5[sub] = {"narratives": len(lists), "share_not_in_whole_question_list_mean": mean(share),
                     "median": st.median(share), "labelled_relevant_rate_whole_question_docs": mean(rate_a),
                     "labelled_relevant_rate_subtopic_only_docs": mean(rate_f),
                     "docs": [len(rate_a), len(rate_f)]}
        print(f"   submitted {sub:18s}: {mean(share):.0%} of each list (median {st.median(share):.0%})"
              f" is not in the whole-question top 200; our labeller calls {mean(rate_a):.0%} of the"
              f" whole-question documents relevant and {mean(rate_f):.0%} of the sub-topic-only ones"
              f" ({len(rate_a)} / {len(rate_f)} documents)")
    RESULT["4_pool"] = {k: {"mean": mean(v), "median": st.median(v)} for k, v in out4.items()}
    RESULT["5_subtopics"] = {"dev22": {k: {"mean": mean(v), "median": st.median(v)} for k, v in out5.items()},
                             "dev22_per_list": per_facet, "submitted_official119": sub5}


def p_set(s, rel):
    return len(s & rel) / len(s) if s else 0.0


# ---------------------------------------------------------------------------------------------
# 6. re-ranking at each stage (replay of tighten.py)
# ---------------------------------------------------------------------------------------------

def sig(x):
    return 1.0 / (1.0 + math.exp(-x))


def fused(ce, init, alpha=0.5):
    """rerank_with_fusion's arithmetic, quirks included (systems/rag/rerank.py)."""
    mx, mn = (max(init) or 1.0), (min(init) or 0.0)
    rng = mx - mn or 1.0
    return [alpha * sig(c) + (1 - alpha) * ((s - mn) / rng) for c, s in zip(ce, init)]


def order(ids, scores):
    return [d for d, _ in sorted(zip(ids, scores), key=lambda x: -x[1])]


def idea6():
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from systems.rag.rerank import CrossEncoderReranker
    ce_model = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce_model._load()
    emb_model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cuda")
    prefix = "Represent this sentence for searching relevant passages: "  # tighten.py

    def ce(query, ids, text):
        return ce_model._model.predict([(query, text[i][:800]) for i in ids], batch_size=64,
                                       show_progress_bar=False, convert_to_numpy=True).tolist()

    lists = defaultdict(dict)       # variant -> topic -> ranked docids
    facet_lists = defaultdict(list)  # variant -> [(topic, ranked docids)]
    check = {"anchor_identical_top10": 0, "anchor_same_order_share": [], "facet_kept_overlap": []}
    for t in DEV:
        d = dev_dir(t)
        narrative = TOPICS[t].get("narrative", "")
        pool = read_pool(d / "pool.jsonl")
        text = {e["docid"]: e["text"] for e in pool}
        bm25 = {e["docid"]: e["bm25_score"] for e in pool}
        rk = load(d / "facet_rankings.json")
        bp = load(d / "blueprint.json")
        ids_pool = [e["docid"] for e in pool]
        ce_narr = dict(zip(ids_pool, ce(narrative, ids_pool, text)))

        # whole-question list: BM25 top 200 of the pool
        top200 = [e["docid"] for e in sorted(pool, key=lambda e: e["bm25_score"], reverse=True)[:200]]
        init = [bm25[x] for x in top200]
        lists["anchor: BM25 order"][t] = top200
        lists["anchor: MiniLM only"][t] = order(top200, [ce_narr[x] for x in top200])
        shipped = order(top200, fused([ce_narr[x] for x in top200], init))
        lists["anchor: MiniLM + BM25 (shipped)"][t] = shipped
        ce_anchor_q = ce(bp["anchor_query"], top200, text)
        lists["anchor: MiniLM + BM25, short anchor query"][t] = order(top200, fused(ce_anchor_q, init))
        bge = load(d / "facet_rankings_bge.json")["_anchor"] if (d / "facet_rankings_bge.json").exists() else None
        if bge:
            lists["anchor: bge-reranker-v2-m3 + BM25 (R7)"][t] = [x for x, _ in bge]
        cached = [x for x, _ in rk["_anchor"]]
        check["anchor_identical_top10"] += shipped[:10] == cached[:10]
        check["anchor_same_order_share"].append(sum(a == b for a, b in zip(shipped, cached)) / len(cached))

        # the whole pool instead of the BM25 top 200
        lists["pool: MiniLM only"][t] = order(ids_pool, [ce_narr[x] for x in ids_pool])
        lists["pool: MiniLM + BM25"][t] = order(ids_pool, fused([ce_narr[x] for x in ids_pool],
                                                                [bm25[x] for x in ids_pool]))

        # sub-topic lists: dense top 150 per query, union, re-rank
        emb = np.load(d / "embeddings.npy")
        docids = load(d / "emb_docids.json")
        union_all = set(cached)
        for f in bp["facets"]:
            qv = emb_model.encode([prefix + q for q in f["queries"]], normalize_embeddings=True,
                                  convert_to_numpy=True).astype(np.float32)
            best_sim = {}
            for row in (emb @ qv.T).T:  # one row of similarities per query
                for j in np.argsort(-row, kind="stable")[:min(150, len(docids))]:
                    best_sim[docids[j]] = max(best_sim.get(docids[j], -1.0), float(row[j]))
            cand = list(best_sim)
            sims = [best_sim[x] for x in cand]
            ce_f = ce(" ; ".join(f["queries"]), cand, text)
            kept = order(cand, fused(ce_f, sims))[:100]
            facet_lists["sub-topic: dense only"].append((t, order(cand, sims)[:100]))
            facet_lists["sub-topic: MiniLM only, sub-topic queries"].append((t, order(cand, ce_f)[:100]))
            facet_lists["sub-topic: MiniLM + dense (shipped)"].append((t, kept))
            facet_lists["sub-topic: MiniLM + dense, whole question"].append(
                (t, order(cand, fused([ce_narr[x] for x in cand], sims))[:100]))
            cached_f = {x for x, _ in rk[f["id"]]}
            check["facet_kept_overlap"].append(len(set(kept) & cached_f) / len(cached_f))
            union_all |= cached_f

        # the merged list: the submitted best-score order, against one re-rank of the union
        lists["merged: best score over lists (submitted)"][t] = best_ce_order(d)[0][:len(union_all)]
        u = sorted(union_all)
        lists["merged: union re-ranked, MiniLM + BM25"][t] = order(u, fused([ce_narr[x] for x in u], [bm25[x] for x in u]))
        lists["merged: anchor-first (shipped dev run)"][t] = [x for x, _ in load(d / "final_ranking.json")][:len(union_all)]

    print(f"\n6. re-ranking at each stage, dev22, qrels v2 (replay of tighten.py: anchor top 10 identical on"
          f" {check['anchor_identical_top10']}/22 topics, same position for"
          f" {mean(check['anchor_same_order_share']):.0%} of the 200; sub-topic lists keep"
          f" {mean(check['facet_kept_overlap']):.0%} of the cached 100 on average)")
    out = {"replay_check": {"anchor_identical_top10": check["anchor_identical_top10"],
                            "anchor_same_position": mean(check["anchor_same_order_share"]),
                            "facet_kept_overlap": mean(check["facet_kept_overlap"])}, "lists": {}, "subtopic_lists": {}}
    ks = (10, 50, 100, 200)
    rows = []
    for v, per in lists.items():
        m = {f"P@{k}": mean(p_at(per[t], rel_of(QV2, t), k) for t in per) for k in ks}
        m.update({"nDCG@10": mean(ndcg_at_k(per[t], QV2[t], 10) for t in per),
                  "R@200": mean(r_at(per[t], rel_of(QV2, t), 200) for t in per),
                  "J@10": mean(j_at(per[t], QV2[t], 10) for t in per),
                  "J@100": mean(j_at(per[t], QV2[t], 100) for t in per),
                  "nDCG'@10": mean(ndcg_at_k(condensed(per[t], QV2[t]), QV2[t], 10) for t in per),
                  "P'@10": mean(p_at(condensed(per[t], QV2[t]), rel_of(QV2, t), 10) for t in per),
                  "P'@100": mean(p_at(condensed(per[t], QV2[t]), rel_of(QV2, t), 100) for t in per)})
        out["lists"][v] = m
        rows.append([v] + [m["nDCG@10"]] + [m[f"P@{k}"] for k in ks] + [m["R@200"], m["J@10"], m["J@100"],
                                                                       m["nDCG'@10"], m["P'@10"], m["P'@100"]])
    show("   whole-question and merged lists", ["nDCG@10"] + [f"P@{k}" for k in ks]
         + ["R@200", "J@10", "J@100", "nDCG'@10", "P'@10", "P'@100"], rows)
    rows = []
    for v, items in facet_lists.items():
        by_t = defaultdict(list)
        for t, lst in items:
            by_t[t].append(lst)
        m = {f"P@{k}": mean(mean(p_at(l, rel_of(QV2, t), k) for l in ls) for t, ls in by_t.items()) for k in (10, 50, 100)}
        m["J@10"] = mean(mean(j_at(l, QV2[t], 10) for l in ls) for t, ls in by_t.items())
        m["P'@10"] = mean(mean(p_at(condensed(l, QV2[t]), rel_of(QV2, t), 10) for l in ls) for t, ls in by_t.items())
        out["subtopic_lists"][v] = m
        rows.append([v, m["P@10"], m["P@50"], m["P@100"], m["J@10"], m["P'@10"]])
    show("   sub-topic lists (mean over a topic's lists, then over topics)",
         ["P@10", "P@50", "P@100", "J@10", "P'@10"], rows)
    RESULT["6_rerank_stage"] = out


if __name__ == "__main__":
    idea1()
    idea2()
    idea3()
    idea4_5()
    if "--no-models" not in sys.argv:
        idea6()
    OUT.write_text(json.dumps(RESULT, indent=1, default=float))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
