#!/usr/bin/env python3
"""E-072b: TensorFold prefill/TTFT — same arms as e072_prefill.py, with per-request draft flag.
Runs DRAFTED (their default policy) and SERIAL ("draft": false) in one server session.
Usage: e072_prefill_tf.py <outfile_prefix> <nreps>
"""
import json, random, sys, time, urllib.request

BASE = "http://127.0.0.1:8010/v1"
MODEL = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"
OUT_PREFIX = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3

random.seed()
WORDS = ("harbor lantern gull tide kelp diesel anchor rope fog bell chart compass rum barrel "
         "spar hull mast quay crate net herring gale beacon cliff foam drone cable winch "
         "engine boiler copper rivet sail wind swell breeze current buoy whistle").split()

def make_words(n_words, tag):
    body = [random.choice(WORDS) for _ in range(n_words)]
    body.append(f"nonce-{tag}-{random.randint(100000,999999)}")
    return " ".join(body)

def ask(prompt, max_tokens=1, draft=True):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "draft": draft}).encode()
    r = urllib.request.Request(BASE + "/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.loads(resp.read())
    wall = time.time() - t0
    u = d.get("usage", {})
    ptoks = u.get("prompt_tokens", 0)
    ctoks = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    fresh = ptoks - ctoks
    return {"wall": round(wall, 3), "prompt_tokens": ptoks, "cached": ctoks,
            "fresh_toks": fresh, "prefill_tps": round(fresh / wall, 1) if wall > 0 else None}

def run_suite(tag, draft):
    results = {"drafted": draft, "cold": {}, "multiturn": {}}
    ask("Say OK", 4, draft)  # warm
    for target in (1000, 4000, 9000, 16000, 20000):
        rows = []
        for rep in range(N):
            p = make_words(int(target * 1.35), f"t{target}r{rep}")
            m = ask(p, 1, draft)
            m["target"] = target
            rows.append(m)
            print(f"[{tag}] cold {target} r{rep}: {m}", flush=True)
        results["cold"][str(target)] = rows
    sysmsg = make_words(int(8000 * 1.35), "sys-" + tag)
    turns = []
    prev = None
    for t in range(3):
        if t == 0:
            prompt = sysmsg + "\n\nAcknowledge the briefing with one sentence."
        else:
            prompt = prev + f"\n\nFollow-up {t}: " + make_words(300, f"turn{t}") + "\nAnswer in one sentence."
        m = ask(prompt, 32, draft)
        m["turn"] = t
        turns.append(m)
        prev = prompt
        print(f"[{tag}] turn {t}: {m}", flush=True)
    json.dump(results, open(f"{OUT_PREFIX}_{tag}.json", "w"), indent=1)
    print(f"WROTE {OUT_PREFIX}_{tag}.json", flush=True)

run_suite("drafted", True)
run_suite("serial", False)
