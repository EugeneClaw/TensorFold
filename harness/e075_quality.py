#!/usr/bin/env python3
"""E-075: E-040 six-task quality suite, parameterized for TF(:8010)+spec and oMLX(:8008) arms.

Usage: e075_quality.py <port> <model_id> <arm_tag> <outfile>
Task content identical to repo/scripts/e040_quality.py (E-053 lineage); deterministic scorers.
"""
import json
import random
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
MODEL = sys.argv[2]
ARM = sys.argv[3]
OUT = sys.argv[4]

FILLER = ("the quick brown fox jumps over lazy dog program memory kernel attention "
          "mechanism transformer layer weights quantization bandwidth latency throughput "
          "measurement silicon metal gpu cpu unified apple studio cluster spark prefill "
          "decode tokens per second benchmark engineering optimisation profiling evidence "
          "hypothesis experiment system architecture implementation performance analysis results").split()

FACTS = [
    ("Marabel Quinstone keeps the saffron keycard in drawer 42.", ["marabel", "quinstone", "saffron", "42"]),
    ("Project ZEPHYRION-9 went live on the fourth of June.", ["zephyrion", "fourth of june"]),
    ("Old Tomas feeds exactly 17 gulls every dawn.", ["tomas", "17"]),
    ("Copper kettle inventory: 1,742 units in the north shed.", ["1,742", "1742"]),
    ("Dr. Yuki Tananaka patented the amber centrifuge in 2019.", ["tananaka", "amber"]),
    ("The night train to Veldsted departs at 11 minutes past midnight.", ["veldsted"]),
    ("Sergeant Pikeson's callsign is BADGER-FIVE.", ["badger"]),
    ("The greenhouse passphrase is 'verdant otter twilight'.", ["verdant otter twilight"]),
    ("Manifest 88-C lists 306 crates of walnut timber.", ["88-c", "306"]),
    ("Emergency blood supply is in fridge B7.", ["b7"]),
    ("Engineer Rhoda Vail wears a titanium ring on her left thumb.", ["rhoda", "titanium"]),
    ("The waterwheel was replaced in 1907 by the Ashford family.", ["1907", "ashford"]),
    ("Three green flares means 'harbour closed'.", ["three green", "green flares"]),
    ("The vault combination begins 4-4-1.", ["4-4-1", "441"]),
    ("Lamp oil is replenished every 11 days.", ["11 days"]),
    ("The ferry 'Salt Maiden' carries at most 63 passengers.", ["salt maiden", "63"]),
    ("Apprentice Fenwick owes the baker 2 shillings.", ["fenwick", "shillings"]),
    ("The Greyfall Ridge map is in the purple ledger's back cover.", ["greyfall", "purple ledger"]),
    ("Harbour bells ring 9 times when the tide turns at dusk.", ["9 times", "nine times"]),
    ("The hermit of Thornwick Island plays a cracked violin on Sundays.", ["thornwick", "violin"]),
]


def gen_filler(target_chars, salt):
    rng = random.Random(salt)
    words = []
    while sum(len(w) + 1 for w in words) < target_chars:
        n = rng.randint(3, 9)
        words.append(" ".join(rng.choice(FILLER) for _ in range(n)) + ".")
    return " ".join(words)


def build_context(target_chars, salt, payloads):
    rng = random.Random(salt ^ 0x5EED)
    parts = []
    interval = max(1, target_chars // (len(payloads) + 1))
    chars = 0
    for i, p in enumerate(payloads):
        f = gen_filler(interval, salt + i)
        parts.append(f)
        parts.append(p)
        chars += len(f) + len(p)
    while chars < target_chars:
        f = gen_filler(2000, salt + 997 + chars)
        parts.append(f)
        chars += len(f)
    return " ".join(parts)


def task_recall(salt):
    picks = [FACTS[(salt + 7 * i) % len(FACTS)] for i in range(8)]
    payload = [p for p, _ in picks]
    ctx = build_context(14000, salt, payload)
    want = [(p, keys) for (p, keys), i in zip(picks, range(8)) if i % 2 == 0]
    q = ("Answer strictly from the notes. For each of these, give the exact fact: "
         + "; ".join(k[0].split()[0:2][0] for p, k in want) + ".")
    q = ("Answer strictly from the notes, one line per item, quoting exactly: "
         + " | ".join(str(i + 1) + ". " + want[i][0].split()[-1] for i in range(len(want))))
    def score(text):
        t = text.lower()
        hits = sum(1 for p, keys in want if all(any(kk in t for kk in keys) for kk in [keys[0]]))
        return hits / len(want), f"{hits}/{len(want)} facts"
    return ctx, q, score


def task_constraints(salt):
    payload = [FACTS[(salt + 3 * i) % len(FACTS)][0] for i in range(6)]
    ctx = build_context(12000, salt, payload)
    q = ("From the notes only: which single item number (drawer, count or time) appears with "
         "'kettle'? Answer with just the number.")
    def score(text):
        return (1.0 if "1742" in text.replace(",", "") else 0.0), text.strip()[:40]
    return ctx, q, score


def task_summary(salt):
    payload = [FACTS[(salt + 5 * i) % len(FACTS)][0] for i in range(10)]
    ctx = build_context(16000, salt, payload)
    q = "Summarise the notes in exactly three sentences, each mentioning one concrete fact."
    def score(text):
        sents = [s for s in text.replace("!\n", ".\n").replace("?\n", ".\n").split(". ") if len(s) > 20]
        digits = sum(1 for s in sents if any(c.isdigit() for c in s))
        ok = min(len(sents), 3) >= 3 and digits >= 2
        return (1.0 if ok else 0.5 if len(sents) >= 2 else 0.0), f"{len(sents)} sents, {digits} w/ facts"
    return ctx, q, score


def task_toolcall(salt):
    payload = [FACTS[(salt + 11 * i) % len(FACTS)][0] for i in range(5)]
    ctx = build_context(10000, salt, payload)
    q = ("Emit one JSON object per note containing the word 'inventory' or a number over 1000, "
         'format: {"item": "<subject>", "value": "<number>"}. Only the JSON lines.')
    def score(text):
        lines = [ln for ln in text.splitlines() if ln.strip().startswith("{")]
        good = 0
        for ln in lines:
            try:
                o = json.loads(ln)
                if "item" in o and "value" in o:
                    good += 1
            except Exception:
                pass
        return (1.0 if good >= 2 else 0.5 if good == 1 else 0.0), f"{good} valid JSON lines"
    return ctx, q, score


def task_codeedit(salt):
    code = ("def process_batch(items, threshold):\n"
            "    out = []\n"
            "    for item in items:\n"
            "        if item.score > threshold:\n"
            "            out.append(item)\n"
            "    return out\n")
    ctx = ("Here is a function:\n\n" + code +
           "\n\nNotes to consider: " + build_context(9000, salt, [FACTS[salt % len(FACTS)][0]]) +
           "\n\nRewrite the function: rename it to filter_strong_items and keep the threshold comparison. "
           "Reply with only the rewritten function.")
    def score(text):
        s = 0.0
        if "def filter_strong_items" in text:
            s += 0.4
        if "return" in text:
            s += 0.2
        if "score > threshold" in text or "threshold" in text:
            s += 0.4
        return s, f"rename+keep={s:.1f}"
    return ctx, "Rewrite the function as instructed.", score


def task_arith(salt):
    payload = [FACTS[(salt + 13 * i) % len(FACTS)][0] for i in range(6)]
    ctx = build_context(13000, salt, payload)
    q = ("Compute 1240 + 380 - 155 + 970 and answer with the number only.")
    def score(text):
        import re
        m = re.findall(r"\d[\d,]*", text.replace(",", ""))
        ok = bool(m) and int(m[0]) == 2435
        return (1.0 if ok else 0.0), f"got {m[0] if m else None} (want 2435)"
    return ctx, q, score


TASKS = [task_recall, task_constraints, task_summary, task_toolcall, task_codeedit, task_arith]
NAMES = ["recall", "constraints", "summary", "toolcall", "codeedit", "arith"]


def answer_of(text: str) -> str:
    """Score what the model ANSWERED, not what it thought: keep only the post-</think> part."""
    return text.split("</think>")[-1] if "</think>" in text else text


def run(ctx, q):
    body = {"model": MODEL, "messages": [{"role": "user", "content": ctx + "\n\n" + q}],
            "max_tokens": 2048, "temperature": 0, "stream": True}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft, chunks, ctok, ptok = None, [], 0, 0
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                o = json.loads(line[6:])
                u = o.get("usage")
                if u:
                    ctok = u.get("completion_tokens") or ctok
                    ptok = u.get("prompt_tokens") or ptok
                ch = o.get("choices", [])
                if ch and (ch[0].get("delta", {}) or {}).get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    chunks.append(ch[0]["delta"]["content"])
            except Exception:
                pass
    wall = time.perf_counter() - t0
    return ttft, wall, ptok, ctok, "".join(chunks)


results = []
salt = 2026095000 + (0 if ARM.startswith("tf") else 555)
for name, builder in zip(NAMES, TASKS):
    for rep in range(2):
        salt += 1
        ctx, q, score = builder(salt)
        ttft, wall, ptok, ctok, text = run(ctx, q)
        sc, detail = score(answer_of(text))
        row = {"task": name, "rep": rep, "arm": ARM, "ttft_s": round(ttft or 0, 2),
               "wall_s": round(wall, 2), "ptok": ptok, "ctok": ctok,
               "score": round(sc, 3), "detail": detail, "text_head": text[:200]}
        results.append(row)
        print(f"[{name} r{rep} {ARM}] score={sc:.2f} ({detail}) ttft={ttft or 0:.1f}s ctok={ctok} "
              f"head={text[:60]!r}", flush=True)

agg = {}
for name in NAMES:
    scores = [r["score"] for r in results if r["task"] == name]
    agg[name] = round(sum(scores) / len(scores), 3)
total = round(sum(agg.values()) / len(agg), 3)
print(f"AGGREGATE [{ARM}]: {total}  per-task: {agg}", flush=True)
json.dump({"arm": ARM, "aggregate": total, "per_task": agg, "rows": results},
          open(OUT, "w"), indent=1)
print(f"WROTE {OUT}")
