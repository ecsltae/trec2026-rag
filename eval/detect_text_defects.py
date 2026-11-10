#!/usr/bin/env python3
"""Detect STRUCTURAL TEXT defects in RAG answer files — the class the support metric cannot see.

Why this exists
---------------
The citation-support metric scores (statement, citation) pairs. It is blind to whether the
statement is readable English. Three independent audits of our already-validated submissions
found defects it rubber-stamped, including sentences whose entire text is a bare period "." that
our judge awarded FULL SUPPORT. NIST evaluates by pooling plus system-vs-system battles read by a
human or a strong LLM, so this class of defect costs credibility exactly where it counts while
being invisible to every number we have been optimising.

Defect classes detected:
  BARE_PUNCT      text has no alphanumeric content at all (e.g. ".")
  TOO_SHORT       fewer than MIN_WORDS content words
  FACET_ID_LEAK   text begins with an internal blueprint facet id (f1:, **f2** (...):, ...)
  HEAD_FRAGMENT   starts lowercase / mid-clause, or starts with a known case-name tail — the
                  signature of a sentence splitter breaking on "v." in "Roe v. Wade"
  NO_TERMINAL     no terminal punctuation
  MARKDOWN_ROW    a markdown table row rendered as an answer sentence
  DANGLING_TAIL   begins with a capitalised fragment that cannot start a sentence

KNOWN LIMITATION — do not treat a clean report as proof the answers are clean.
An adversarial audit found 13 defects this file missed. Two of them are STILL undetectable here:
"Roosevelt leads to a world under total German and Japanese domination." (a split on "Franklin D.")
and "Constitution to refuse medical treatment." (a split on "U. S.") both start with a capital, end
with a period, and are of ordinary length — they are only identifiable by READING the sentence and
noticing the subject is missing. Regex cannot close this class. Budget a human/LLM read of a sample
of answers before any submission; this tool catches the mechanical classes only.

Read-only. No GPU, no model calls, no network.

Usage:
    python eval/detect_text_defects.py --run runs/rag/x.jsonl [more.jsonl ...]
    python eval/detect_text_defects.py --submission submissions/bp_agentic_v2
    python eval/detect_text_defects.py --submission submissions/* --json out.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

MIN_WORDS = 5

FACET_ID = re.compile(r"^\**\s*f\d+\**\s*(\([^)]*\))?\s*[:.]\s*", re.IGNORECASE)
# An adversarial audit found 7 leaked facet ids that this file MISSED because FACET_ID above is
# anchored to ^. Internal ids leak mid-sentence too ("... as requested in f4.", "... leaving facets
# f3, f4, and f5 unsupported ..."), so scan the whole string as well.
FACET_ANYWHERE = re.compile(r"\bf\d+\b")
# Likewise, CASE_TAIL below is a hardcoded allowlist and therefore structurally blind to any case
# name not on it. An UNPAIRED asterisk is the splitter artifact's real signature and is name-
# agnostic: "State of Montana* trial ...", "Hendricks*, recognizing ...".
UNPAIRED_STAR = re.compile(r"^[^*]*\*[^*]*$")
MARKDOWN_ROW = re.compile(r"^\s*\|.*\|")
TERMINAL = re.compile(r"[.!?\"')\]]\s*$")
# A sentence that opens with "<Word>* " or with a bare proper-noun tail is the classic
# split-on-"v." artifact: "Roe v. Wade" -> "...Roe v." + "Wade* ...".
# The negative lookahead is essential: a CORRECTLY formed "Roe v. Wade established ..." opens with
# a word from this list but is not a fragment, so only flag it when " v." does NOT follow.
CASE_TAIL = re.compile(
    r"^(Wade|Glucksberg|Klein|Director|Board|Ferguson|Madison|Doe|Roe)\b[*]?(?!\s+v\.?\s)")
LOWER_START = re.compile(r"^[a-z]")


# --- headless-sentence rules, added after a human read-sweep found 5 more fragments -------------
# The read sweep found sentences that start mid-clause because the tighten/ground step dropped the
# leading NP. They are grammatical-looking and of ordinary length, so nothing above catches them:
#   "Afrifa, who were army officers and senior police officials."
#   "Eisenhower's air commanders, who initially favored attritional attacks..."
#   "Departments of Education, Justice, and Homeland Security as a critical component..."
# The two rules below were tuned by MEASURING on all 4,664 shipped sentences: together they catch
# 11/11 known cases at ONE false positive (0.02%). Two other rules were tried and REJECTED as
# unshippable — a finite-verb inventory fired on 20.4% of legitimate sentences and a gerund rule on
# 2.3%; without a POS tagger (none installed offline) that class cannot be closed. So this is a
# REVIEW GATE, not an auto-fixer, and it provably cannot catch fragments like
#   "Navy Band Commodores at 7 p.m., serving as a perfect prelude to the holiday."
# which need real parsing.
_STOP = (r"(?:The|A|An|This|That|These|Those|Its|It|They|Their|Some|Many|Most|Such|Other|Both|All|"
         r"Each|One|Two|Several|Recent|Current|Unlike|Like|Despite|Although|While|Because|Since|"
         r"After|Before|During|Following|Given|With|Without|Beyond|Among|Between|Under|Within|"
         r"For|In|On|At|By|From|As|If|When|Where|Whereas|Though|However|Consequently|Furthermore|"
         r"Moreover|Additionally|Conversely|Similarly|Therefore|Thus|Meanwhile|Overall|Ultimately)")
REL_FRAG = re.compile(
    rf"^(?!{_STOP}\b)[A-Z][\w'’-]*(?:\s+[\w'’.-]+){{0,2}},\s+(?:who|which|that|whose|whom)\b")
BARE_INST = re.compile(r"^(?:Departments?|Bureau|Office|Agency|Commission|Committee|Task Force|"
                       r"Ninth|Eighth|Seventh|Sixth|Fifth|Fourth|Third|Second|Preventive|Circuit)"
                       r"(?![-\w])")


def content_words(t: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", t))


def inspect(text: str) -> list[str]:
    t = (text or "").strip()
    flags = []
    if content_words(t) == 0:
        flags.append("BARE_PUNCT")
        return flags  # nothing else meaningful to say
    if MARKDOWN_ROW.match(t):
        flags.append("MARKDOWN_ROW")
    if FACET_ID.match(t):
        flags.append("FACET_ID_LEAK")
    elif FACET_ANYWHERE.search(t):
        flags.append("FACET_ID_MIDSENTENCE")
    if UNPAIRED_STAR.match(t):
        flags.append("UNPAIRED_ASTERISK")
    if REL_FRAG.match(t):
        flags.append("REL_FRAG_SUSPECT")
    if BARE_INST.match(t):
        flags.append("BARE_INST_SUSPECT")
    if CASE_TAIL.match(t):
        flags.append("HEAD_FRAGMENT")
    if LOWER_START.match(t):
        flags.append("HEAD_FRAGMENT")
    if content_words(t) < MIN_WORDS:
        flags.append("TOO_SHORT")
    if not TERMINAL.search(t):
        flags.append("NO_TERMINAL")
    return flags


def scan(path: Path) -> dict:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    hits, n_sent = [], 0
    for r in rows:
        meta = r.get("metadata", {})
        tid = str(meta.get("topic_id") or meta.get("narrative_id") or "?")
        for i, s in enumerate(r.get("answer", [])):
            n_sent += 1
            txt = s.get("text", "")
            fl = inspect(txt)
            if fl:
                hits.append({"topic_id": tid, "sent_idx": i, "flags": fl,
                             "citations": s.get("citations", []),
                             "text": txt[:220]})
    return {"path": str(path), "records": len(rows), "sentences": n_sent, "defects": hits}


def report(res: dict) -> int:
    from collections import Counter
    d = res["defects"]
    print("=" * 78)
    print(f"  {res['path']}")
    print(f"  {res['records']} records · {res['sentences']} sentences · "
          f"{len(d)} defective sentences in {len({h['topic_id'] for h in d})} topics")
    print("=" * 78)
    if not d:
        print("  CLEAN — no structural text defects.")
        return 0
    for cls, n in Counter(f for h in d for f in h["flags"]).most_common():
        print(f"    {cls:<15} {n}")
    print()
    for h in sorted(d, key=lambda x: (x["topic_id"], x["sent_idx"])):
        print(f"  {h['topic_id']:>12} s{h['sent_idx']:<3} {','.join(h['flags']):<28} "
              f"cites={h['citations']}")
        print(f"               {h['text']!r}")
    return len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="*", default=[])
    ap.add_argument("--submission", nargs="*", default=[])
    ap.add_argument("--json", help="write machine-readable results here")
    a = ap.parse_args()

    targets = [Path(p) for p in a.run]
    for sd in a.submission:
        p = Path(sd) / "rag_output_trec_rag_2026.jsonl"
        if p.exists():
            targets.append(p)
        else:
            print(f"WARN: no rag_output in {sd}", file=sys.stderr)
    if not targets:
        sys.exit("nothing to scan; pass --run and/or --submission")

    out, total = [], 0
    for t in targets:
        res = scan(t)
        total += report(res)
        out.append(res)
    print(f"\nTOTAL defective sentences across {len(targets)} file(s): {total}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
