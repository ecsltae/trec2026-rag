"""
A person's check of our support judge: 100 sentence-citation pairs, rated blind.

    python eval/analysis/human_support_check.py build    # after review_checks_2026_11_10.py support-shown
    python eval/analysis/human_support_check.py score ratings.csv

build  samples 100 pairs from the four development writers' answers (submitted, coverage prompt,
       best passages, multi-pass), 25 per writer, balanced over the judge's three verdicts where
       possible, with a fixed seed. Each pair is the sentence and the passage the writer was shown
       (the same text the judge read in the support-shown pass). Writes, under
       eval/analysis/human_support_check/:
         rate.html   self-contained page: one pair per card, full / partial / none, and a button
                     that downloads ratings.csv. Nothing about the writer or the judge is shown.
         key.json    pair id -> writer, topic, docid, the judge's verdict (do not open while rating)
score  compares ratings.csv with the judge: agreement, Cohen's kappa, and the confusion matrix,
       overall and for "full support" against the rest (the verdict the grounded coverage uses).
"""
import csv
import html
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
OUTDIR = Path(__file__).parent / "human_support_check"
RUNS = {"bp_agentic_v2_dev22": "head700", "bp_v2_coverage_dev22": "head700",
        "bp_v2_evidence_dev22": "best1000", "bp_v2_multipass_dev22": "best1000"}
NAMES = {0: "none", 1: "partial", 2: "full"}


def build(per_run=25, seed=13):
    from systems.rag.agentic_v2.harvest import read_pool
    from systems.rag.agentic_v2.synthesize import _best_window, _query_terms
    from systems.rag.agentic_v2.pipeline import load_topic_set
    topics = load_topic_set("dev22")
    dev = ROOT / "data/rag/cache/agentic_v2/dev22"
    rng = random.Random(seed)
    pairs = []
    for run, mode in RUNS.items():
        rows = [json.loads(l) for l in open(ROOT / "runs/rag" / f"{run}.support_shown.jsonl")]
        cands = [(r["topic_id"], s["text"], c["docid"], c["support"])
                 for r in rows for s in r["sentences"] for c in s["citations"]]
        by = {v: [x for x in cands if x[3] == v] for v in (0, 1, 2)}
        take, want = [], [per_run // 3, per_run // 3, per_run - 2 * (per_run // 3)]
        for v, n in zip((0, 1, 2), want):
            take += rng.sample(by[v], min(n, len(by[v])))
        rest = [x for x in cands if x not in take]
        take += rng.sample(rest, per_run - len(take))
        for t, sent, d, v in take:
            texts = {e["docid"]: e["text"] for e in read_pool(dev / t / "pool.jsonl")}
            bp = json.loads((dev / t / "blueprint.json").read_text())
            full = texts.get(d, "")
            shown = full[:700] if mode == "head700" else _best_window(
                full, _query_terms(topics[t].get("narrative", ""), bp["facets"]), 1000)
            pairs.append({"run": run, "topic": t, "docid": d, "judge": v, "sentence": sent, "passage": shown})
    rng.shuffle(pairs)
    OUTDIR.mkdir(exist_ok=True)
    key = {f"p{i:03d}": {k: p[k] for k in ("run", "topic", "docid", "judge")} for i, p in enumerate(pairs, 1)}
    (OUTDIR / "key.json").write_text(json.dumps(key, indent=1))
    cards = "\n".join(
        f'<section class="card" data-id="p{i:03d}"><h2>Pair {i} of {len(pairs)}</h2>'
        f'<p class="lab">Sentence</p><p class="sent">{html.escape(p["sentence"])}</p>'
        f'<p class="lab">Cited passage</p><p class="pass">{html.escape(p["passage"])}</p>'
        f'<fieldset><legend>Does the passage support the sentence?</legend>'
        + "".join(f'<label><input type="radio" name="p{i:03d}" value="{v}"> {v}</label>'
                  for v in ("full", "partial", "none"))
        + '</fieldset></section>' for i, p in enumerate(pairs, 1))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Support check</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;color:#16191f;background:#fff}}
.card{{border:1px solid #d0d5da;border-radius:8px;padding:1rem 1.25rem;margin:1rem 0}}.lab{{font-size:13px;color:#4b5563;margin:.6rem 0 .1rem;text-transform:uppercase;letter-spacing:.04em}}
.sent{{font-weight:600}}.pass{{white-space:pre-wrap;background:#f6f7f8;padding:.6rem;border-radius:6px}}fieldset{{border:0;padding:0;margin-top:.6rem}}
label{{margin-right:1.2rem}}#bar{{position:sticky;top:0;background:#fff;padding:.5rem 0;border-bottom:1px solid #d0d5da}}button{{font:inherit;padding:.4rem .9rem}}</style></head><body>
<div id="bar"><b>Support check</b> — rate each pair: <i>full</i> if the passage states everything the sentence claims,
<i>partial</i> if it supports some of it, <i>none</i> otherwise. <span id="n"></span> <button id="dl">Download ratings.csv</button></div>
{cards}
<script>
const ids=[...document.querySelectorAll('.card')].map(c=>c.dataset.id);
function count(){{document.getElementById('n').textContent=ids.filter(i=>document.querySelector('input[name='+i+']:checked')).length+' of '+ids.length+' rated';}}
document.addEventListener('change',count);count();
document.getElementById('dl').onclick=()=>{{const rows=['id,rating'].concat(ids.map(i=>{{const c=document.querySelector('input[name='+i+']:checked');return i+','+(c?c.value:'');}}));
const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([rows.join('\\n')],{{type:'text/csv'}}));a.download='ratings.csv';a.click();}};
</script></body></html>"""
    (OUTDIR / "rate.html").write_text(page)
    print(f"wrote {len(pairs)} pairs: {OUTDIR / 'rate.html'} (and key.json)")
    print("judge verdicts in the sample:", Counter(NAMES[p["judge"]] for p in pairs))


def kappa(a, b, labels):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def score(path):
    key = json.loads((OUTDIR / "key.json").read_text())
    rated = {r["id"]: r["rating"] for r in csv.DictReader(open(path)) if r["rating"]}
    ids = [i for i in key if i in rated]
    h = [rated[i] for i in ids]
    j = [NAMES[key[i]["judge"]] for i in ids]
    print(f"{len(ids)} pairs rated")
    print(f"three verdicts: agreement {sum(x == y for x, y in zip(h, j)) / len(ids):.2f}, "
          f"kappa {kappa(h, j, ['full', 'partial', 'none']):.2f}")
    hf, jf = [x == "full" for x in h], [x == "full" for x in j]
    print(f"full vs rest:   agreement {sum(x == y for x, y in zip(hf, jf)) / len(ids):.2f}, "
          f"kappa {kappa(hf, jf, [True, False]):.2f}")
    print("confusion (rows: person, columns: judge):")
    for r in ("full", "partial", "none"):
        print(f"  {r:8s}" + "".join(f"{sum(1 for x, y in zip(h, j) if x == r and y == c):6d}" for c in ("full", "partial", "none")))


if __name__ == "__main__":
    {"build": lambda: build(), "score": lambda: score(sys.argv[2])}[sys.argv[1]]()
