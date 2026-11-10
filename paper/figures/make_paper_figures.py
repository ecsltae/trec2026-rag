"""
Figures for the BiTeM TREC 2026 RAG notebook paper (all but the pipeline, which is TikZ).

Every plotted value is read from files on disk, and each figure asserts that what it plots
matches the numbers printed in the paper, so a figure cannot silently disagree with the text.

Style, as in the VQA paper (repos/trec-vqa/paper/figures/make_figures.py): no text below 9 pt at
printed size; every figure is drawn at its exact LaTeX width (column 3.33 in, page 7.0 in) and
included unscaled; Liberation Sans. Colours are the RAG talk deck's: ink for the whole topic,
dataviz slots 1-5 for the facets (validated with the dataviz validator: CVD and normal-vision
separation pass; three slots fall under 3:1 contrast, so every coloured mark is also named in a
legend, and series in the charts also differ by marker and line style).

  fig_rerank_map.pdf   page     reranking and context selection on rag2026-0 (t-SNE document map)
  fig_scored_map.pdf   column   what was scored on rag2026-0, by the ranking each document came from
  fig_ranks.pdf        column   development recall and precision at depth k, counting unlabelled documents as misses and over labelled documents only
  fig_stages.pdf       column   each ranking before and after reranking (precision at 10, two bounds)
  fig_grounded.pdf     column   coverage when only fully supported sentences count

    source ~/TREC/venv/bin/activate
    python paper/tex/figures/make_paper_figures.py          # CPU only; about a minute

Inputs: the official119 cache for rag2026-0, the scored submission file, and the analysis outputs
eval/analysis/team_ideas_2026_11_09.json and eval/analysis/nugget_dev22_*.json.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Agg can measure text for check_text; savefig still writes PDF
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
for _f in ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf", "LiberationSans-Italic.ttf"):
    font_manager.fontManager.addfont("/usr/share/fonts/truetype/liberation2/" + _f)

HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent                    # repos/trec-rag
sys.path.insert(0, str(ROOT))
from systems.rag.agentic_v2.harvest import read_pool
from systems.rag.agentic_v2.synthesize import _select_context

TOPIC = "rag2026-0"
D = ROOT / "data/rag/cache/agentic_v2/official119" / TOPIC
SCORED = ROOT / "submissions/bp_agentic_v2/r_output_trec_rag_2026.tsv"
ANALYSIS = ROOT / "eval/analysis"
PREFIX = "Represent this sentence for searching relevant passages: "   # tighten.py
COL, PAGE = 3.33, 7.0                                                   # acmart sigconf, inches

INK, INK2, POOL, MID, GRID = "#16191f", "#4B5563", "#dfe3e6", "#8f99a3", "#E5E7EB"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LIGHT1 = "#9ec5f4"
SHORT = {"Pipeline & Education Partnerships": "education",
         "Hiring & Promotion Reform": "hiring",
         "Reporting & Accountability Systems": "accountability",
         "Budget Tradeoffs & Metrics": "budget",
         "Avoiding Performative Pitfalls": "pitfalls"}

plt.rcParams.update({
    "font.family": "Liberation Sans", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9, "legend.title_fontsize": 9,
    "pdf.fonttype": 42, "axes.edgecolor": MID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def check_text(fig, name):
    """No text under 9 pt, and none outside the canvas (the figure is included unscaled)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.get_size_inches() * fig.dpi
    bad = []
    undrawn = set()             # tick labels whose tick lies outside the view limits are never drawn
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if not lo - 1e-9 <= tick.get_loc() <= hi + 1e-9:
                    undrawn.update({id(tick.label1), id(tick.label2)})
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in undrawn:
            continue
        if t.get_fontsize() < 9 - 1e-6:
            bad.append(f"{t.get_text()[:30]!r} at {t.get_fontsize():.1f} pt")
        b = t.get_window_extent(r)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > W + 1 or b.y1 > H + 1:
            bad.append(f"{t.get_text()[:30]!r} outside the canvas")
    assert not bad, f"{name}: " + "; ".join(bad)


def near(a, b, tol=0.0006):
    return abs(a - b) <= tol


def save(fig, name):
    check_text(fig, name)
    fig.savefig(HERE / name)            # exact size: no bbox_inches, so LaTeX can include it unscaled
    plt.close(fig)
    print("wrote", name)


def bare(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title, pad=2)


def dot(color, label, size=6, ring="white"):
    return plt.Line2D([], [], marker="o", ls="", markersize=size, markerfacecolor=color,
                      markeredgecolor=ring, label=label)


# ------------------------------------------------------------------ the document maps
def maps():
    pool = read_pool(D / "pool.jsonl")
    emb = np.load(D / "embeddings.npy")
    docids = json.loads((D / "emb_docids.json").read_text())
    bp = json.loads((D / "blueprint.json").read_text())
    rankings = json.loads((D / "facet_rankings.json").read_text())
    row = {d: i for i, d in enumerate(docids)}
    facets = bp["facets"]
    fids = [f["id"] for f in facets]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    queries = [q for f in facets for q in f["queries"]]
    q_emb = model.encode([PREFIX + q for q in queries], normalize_embeddings=True)

    # Candidates exactly as tighten.py builds them, checked against the cache.
    cands, k = {}, 0
    for f in facets:
        ids = set()
        for _ in f["queries"]:
            ids.update(np.argsort(-(emb @ q_emb[k]))[:150].tolist())
            k += 1
        cands[f["id"]] = ids
        assert {row[d] for d, _ in rankings[f["id"]]} <= ids, f"{f['id']}: kept set not inside candidates"
    bm25_top = {row[e["docid"]] for e in sorted(pool, key=lambda e: e["bm25_score"], reverse=True)[:200]}
    assert {row[d] for d, _ in rankings["_anchor"]} == bm25_top, "whole-question candidates do not match"

    # What the writer reads, and which ranking contributed each document (synthesize.py).
    context = [d for d, _ in _select_context(rankings, {e["docid"]: e["text"] for e in pool})]
    source, seen = {}, set()
    for d, _ in rankings["_anchor"][:10]:
        source[d] = "_anchor"; seen.add(d)
    for f in facets:
        for d, _ in rankings[f["id"]][:6]:
            if d not in seen:
                source[d] = f["id"]; seen.add(d)
    assert set(source) == set(context)

    # The scored ranking, and the ranking each document took its best score from
    # (scripts/build_variable_depth_runs.py best_ce: first ranking in file order reaching the maximum).
    scored = [l.split()[2] for l in open(SCORED) if l.split()[0] == TOPIC]
    best = {}
    for name, lst in rankings.items():
        for d, s in lst:
            if d not in best or s > best[d][0]:
                best[d] = (s, name)
    origin = {d: best[d][1] for d in scored}
    head = [d for d, _ in json.loads((D / "final_ranking.json").read_text())][: len(scored)]

    # numbers printed in the paper
    assert len(pool) == 1814 and len(cands[fids[0]]) == 225 and len(context) == 28
    assert len(scored) == 197 and Counter(origin.values())["_anchor"] == 10
    assert all(d in {x for x, _ in rankings["_anchor"]} for d in head)

    from sklearn.manifold import TSNE
    xy = TSNE(n_components=2, metric="cosine", init="pca", perplexity=30, random_state=0).fit_transform(
        np.vstack([emb, q_emb]))
    P = xy[: len(emb)]
    ring = dict(edgecolors="white", linewidths=0.5)
    color_of = {"_anchor": INK, **{fid: SERIES[i] for i, fid in enumerate(fids)}}
    name_of = {"_anchor": "whole topic", **{f["id"]: SHORT.get(f["name"], f["name"]) for f in facets}}
    order = ["_anchor"] + fids

    # Figure 2: reranking and context, page width
    fig = plt.figure(figsize=(PAGE, 2.75))
    axes = [fig.add_axes([0.005 + i * 0.333, 0.30, 0.322, 0.62]) for i in range(3)]
    for ax in axes:
        ax.scatter(P[:, 0], P[:, 1], s=2.0, c=POOL, linewidths=0, zorder=1, rasterized=True)
    f1 = fids[0]
    bare(axes[0], f"(a) one facet ({name_of[f1]})")
    c = sorted(cands[f1]); kept = [row[d] for d, _ in rankings[f1]]
    axes[0].scatter(P[c, 0], P[c, 1], s=5, c=LIGHT1, linewidths=0, zorder=2)
    axes[0].scatter(P[kept, 0], P[kept, 1], s=7, c=SERIES[0], linewidths=0, zorder=3)
    axes[0].legend(handles=[dot(LIGHT1, f"nearest by bge ({len(c)})", 5, LIGHT1),
                            dot(SERIES[0], f"kept by MiniLM ({len(kept)})", 5, SERIES[0])],
                   loc="upper left", bbox_to_anchor=(0.0, -0.02), handletextpad=0.2)
    bare(axes[1], "(b) the whole topic")
    b = sorted(bm25_top); top10 = [row[d] for d, _ in rankings["_anchor"][:10]]
    axes[1].scatter(P[b, 0], P[b, 1], s=5, c=MID, linewidths=0, zorder=2)
    axes[1].scatter(P[top10, 0], P[top10, 1], s=22, c=INK, zorder=3, **ring)
    axes[1].legend(handles=[dot(MID, f"BM25 top {len(b)}", 5, MID), dot(INK, "MiniLM top 10")],
                   loc="upper left", bbox_to_anchor=(0.0, -0.02), handletextpad=0.2)
    bare(axes[2], f"(c) what the writer reads ({len(context)})")
    for d, src in source.items():
        axes[2].scatter(P[row[d], 0], P[row[d], 1], s=24, c=color_of[src], zorder=4, **ring)
    n = Counter(source.values())
    # two columns at 9 pt are wider than the panel, so this legend starts under panel (b)'s empty right half
    fig.legend(handles=[dot(color_of[s], f"{name_of[s]} ({n.get(s, 0)})") for s in order],
               loc="upper left", bbox_to_anchor=(0.585, 0.29), ncol=2, handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig_rerank_map.pdf")

    # Figure 3: what was scored, column width
    fig = plt.figure(figsize=(COL, 2.75))
    axes = [fig.add_axes([0.01 + i * 0.50, 0.38, 0.47, 0.54]) for i in range(2)]
    for ax in axes:
        ax.scatter(P[:, 0], P[:, 1], s=1.6, c=POOL, linewidths=0, zorder=1, rasterized=True)
    bare(axes[0], f"(a) anchor-first, top {len(head)}")
    axes[0].scatter([P[row[d], 0] for d in head], [P[row[d], 1] for d in head], s=6, c=INK, linewidths=0, zorder=3)
    bare(axes[1], f"(b) scored, {len(scored)}")
    for src in order:
        pts = [row[d] for d in scored if origin[d] == src]
        if pts:
            axes[1].scatter(P[pts, 0], P[pts, 1], s=6, c=color_of[src], linewidths=0, zorder=3)
    on = Counter(origin.values())
    fig.legend(handles=[dot(color_of[s], f"{name_of[s]} ({on.get(s, 0)})", 5.5, color_of[s]) for s in order],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, handletextpad=0.2, columnspacing=0.8)
    save(fig, "fig_scored_map.pdf")
    return {"context": dict((name_of[s], v) for s, v in n.items()), "scored": dict((name_of[s], v) for s, v in on.items())}


# ------------------------------------------------------------------ development results at depth k
def ranks():
    r = json.loads((ANALYSIS / "team_ideas_2026_11_09.json").read_text())["1_ranks"]["v2"]
    ks = [5, 10, 20, 50, 100, 200, 500, 1000]
    wq, af, sc, bm = (r["whole question"], r["anchor-first (shipped)"],
                      r["best-score order (submitted)"], r["plain BM25"])
    # numbers printed in the paper
    assert near(af["R@1000"], 0.302, 0.0015) and near(wq["R@1000"], 0.250, 0.0015)
    assert near(sc["P@10"], 0.295, 0.0015) and near(sc["P'@10"], 0.927, 0.0015)
    # identical lists through rank 200 on 21 of 22 topics; the 22nd differs by a few documents
    assert all(abs(af[f"R@{k}"] - wq[f"R@{k}"]) < 0.001 and abs(af[f"P@{k}"] - wq[f"P@{k}"]) < 0.003
               for k in ks if k <= 200)

    fig = plt.figure(figsize=(COL, 3.9))
    a1 = fig.add_axes([0.14, 0.62, 0.83, 0.32])
    a2 = fig.add_axes([0.14, 0.13, 0.83, 0.32])
    a1.plot(ks, [wq[f"R@{k}"] for k in ks], color=MID, lw=1.6, ls="--", marker="o", ms=4, label="whole topic")
    a1.plot(ks, [af[f"R@{k}"] for k in ks], color=INK, lw=1.4, marker="s", ms=3.5, label="anchor-first")
    a1.plot(ks, [sc[f"R@{k}"] for k in ks], color=SERIES[0], lw=1.4, marker="^", ms=4, label="as scored")
    a1.set_title("(a) recall at depth k")
    a1.legend(loc="upper left", handlelength=1.8)
    a1.set_ylim(0, 0.32); a1.set_yticks([0, 0.1, 0.2, 0.3])
    # no shaded band: scoring labelled documents only is an estimate, not an upper bound
    for lst, color, mk, lab in ((af, INK, "s", "anchor-first"), (sc, SERIES[0], "^", "as scored")):
        lo = [lst[f"P@{k}"] for k in ks]; hi = [lst[f"P'@{k}"] for k in ks]
        a2.plot(ks, lo, color=color, lw=1.4, marker=mk, ms=3.5, label=f"{lab}")
        a2.plot(ks, hi, color=color, lw=1.0, ls=":", marker=mk, ms=3.5, mfc="white")
    a2.set_title("(b) precision at depth k")
    a2.text(1000, 0.90, "labelled documents only", ha="right", va="bottom", color=INK2)
    a2.text(1000, 0.06, "unlabelled counted as misses", ha="right", va="bottom", color=INK2)
    a2.set_ylim(0, 1.0)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    a2.set_xlabel("depth k")
    save(fig, "fig_ranks.pdf")


# ------------------------------------------------------------------ reranking at each stage
def stages():
    r = json.loads((ANALYSIS / "team_ideas_2026_11_09.json").read_text())["6_rerank_stage"]
    L, S = r["lists"], r["subtopic_lists"]
    rows = [("whole topic", None),
            ("BM25 order", L["anchor: BM25 order"]),
            ("MiniLM + BM25 (ours)", L["anchor: MiniLM + BM25 (shipped)"]),
            ("bge reranker + BM25", L["anchor: bge-reranker-v2-m3 + BM25 (R7)"]),
            ("each facet", None),
            ("dense order (bge)", S["sub-topic: dense only"]),
            ("MiniLM + dense (ours)", S["sub-topic: MiniLM + dense (shipped)"]),
            ("same, whole topic", S["sub-topic: MiniLM + dense, whole question"]),
            ("merged rankings", None),
            ("best score (as scored)", L["merged: best score over lists (submitted)"]),
            ("reranked, whole topic", L["merged: union re-ranked, MiniLM + BM25"])]
    # numbers printed in the paper
    assert near(L["anchor: BM25 order"]["P@10"], 0.695, 0.0015)
    assert near(L["anchor: MiniLM + BM25 (shipped)"]["P@10"], 0.850, 0.0015)
    assert near(S["sub-topic: MiniLM + dense (shipped)"]["P@10"], 0.272, 0.0015)
    assert near(S["sub-topic: MiniLM + dense, whole question"]["P@10"], 0.357, 0.0015)
    assert near(S["sub-topic: MiniLM + dense (shipped)"]["P'@10"], 0.857, 0.0015)
    assert near(S["sub-topic: MiniLM + dense, whole question"]["P'@10"], 0.919, 0.0015)

    fig = plt.figure(figsize=(COL, 3.45))
    ax = fig.add_axes([0.45, 0.235, 0.52, 0.745])
    y = len(rows) - 1
    labels, ys = [], []
    for name, m in rows:
        if m is None:
            ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontweight="bold", color=INK)
        else:
            lo, hi = m["P@10"], m["P'@10"]
            ax.plot([lo, hi], [y, y], color=MID, lw=1.6, zorder=1)
            ax.plot([lo], [y], "o", color=INK, ms=5, zorder=2)
            ax.plot([hi], [y], "o", color=INK, ms=5, mfc="white", zorder=2)
            labels.append(name); ys.append(y)
        y -= 1
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0.2, 1.0); ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("precision at 10")
    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=INK, ms=5, label="unlabelled as misses"),
                        plt.Line2D([], [], marker="o", ls="", color=INK, mfc="white", ms=5, label="labelled only")],
               loc="lower left", bbox_to_anchor=(0.02, 0.0), ncol=2, handletextpad=0.1, columnspacing=0.8)
    save(fig, "fig_stages.pdf")


# ------------------------------------------------------------------ grounded coverage
def grounded():
    runs = {}
    for f in ("nugget_dev22_existing.json", "nugget_dev22_arms.json", "nugget_dev22_multipass.json",
              "nugget_dev22_grounded.json"):
        runs.update(json.loads((ANALYSIS / f).read_text())["runs"])
    names = [("submitted", "bp_agentic_v2_dev22"), ("best passages", "bp_v2_evidence_dev22"),
             ("weights, reviews", "bp_v2_multipass_dev22")]   # the row names of Table 4
    full = {}
    for _, run in names:
        sents = [s for r in map(json.loads, open(ROOT / f"runs/rag/{run}.support_judgments.jsonl"))
                 for s in r["sentences"]]
        full[run] = sum(any(c["support"] == 2 for c in s["citations"]) for s in sents) / len(sents)
    a = {run: runs[run]["strict_all_score"] for _, run in names}
    g = {run: runs[f"{run}_grounded"]["strict_all_score"] for _, run in names}
    # numbers printed in the paper
    assert near(a["bp_agentic_v2_dev22"], 0.356) and near(g["bp_agentic_v2_dev22"], 0.305)
    assert near(a["bp_v2_evidence_dev22"], 0.488) and near(g["bp_v2_evidence_dev22"], 0.382)
    assert round(full["bp_agentic_v2_dev22"] * 100) == 83 and round(full["bp_v2_evidence_dev22"] * 100) == 65

    fig = plt.figure(figsize=(COL, 1.85))
    a1 = fig.add_axes([0.31, 0.27, 0.42, 0.58])
    a2 = fig.add_axes([0.77, 0.27, 0.20, 0.58])
    ys = [2, 1, 0]
    for (label, run), y in zip(names, ys):
        a1.annotate("", xy=(g[run], y), xytext=(a[run], y),
                    arrowprops=dict(arrowstyle="-|>", color=MID, lw=1.3, shrinkA=4, shrinkB=4))
        a1.plot([a[run]], [y], "o", color=INK, ms=5.5)
        a1.plot([g[run]], [y], "o", color=INK, ms=5.5, mfc="white")
        a2.barh(y, full[run] * 100, height=0.5, color=SERIES[0])
        a2.text(full[run] * 100 - 3, y, f"{full[run] * 100:.0f}%", ha="right", va="center", color="white")
    a1.set_yticks(ys); a1.set_yticklabels([n for n, _ in names])
    a2.set_yticks([])
    for ax in (a1, a2):
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_ylim(-0.6, 2.6)
    a1.set_xlim(0.27, 0.52); a1.set_xticks([0.3, 0.4, 0.5])
    a2.set_xlim(0, 100); a2.set_xticks([0, 50, 100])
    a1.set_title("(a) coverage", pad=3)
    a2.set_title("(b) backed", pad=3)
    a1.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=INK, ms=5.5, label="all sentences"),
                       plt.Line2D([], [], marker="o", ls="", color=INK, mfc="white", ms=5.5,
                                  label="fully supported only")],
              loc="lower left", bbox_to_anchor=(-0.80, -0.42), ncol=2, handletextpad=0.1, columnspacing=0.8)
    save(fig, "fig_grounded.pdf")
    return {"coverage": a, "grounded": g, "fully_supported_share": full}


if __name__ == "__main__":
    out = {"maps": maps()}
    ranks()
    stages()
    out["grounded"] = grounded()
    (HERE / "paper_figures.json").write_text(json.dumps(out, indent=2))
