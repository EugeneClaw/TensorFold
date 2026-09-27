#!/usr/bin/env python3
"""E-075 soak fixpoint: replay the MISS configs from /tmp/e075_soak.json at max_tokens=512.
Prompt reconstruction is deterministic (content depends only on i and class)."""
import json, time, random, urllib.request

import os
BASE = os.environ.get("SOAK_BASE", "http://127.0.0.1:8008/v1/chat/completions")
FILLER = ("harbor ledger meridian lantern compass granite thistle fossil amber copper drizzle "
          "fabric garnet hollow ingress jade kernel lumen masonry nominal orchard prism quarry "
          "ribbon saffron tundra umber velvet willow zephyr anchor beacon cipher delta ember "
          "flint glade helium iris jasper karst lagoon masonry opal pumice quartz reed slate").split()

def fillers(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out.append(rng.choice(FILLER))
        if len(out) % 11 == 0:
            out[-1] += "."
    return " ".join(out)

def run(prompt, max_tokens):
    body = json.dumps({"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return time.time() - t0, d["choices"][0]["message"]["content"]

def main():
    rows = json.load(open("/tmp/e075_soak.json"))
    misses = [r["i"] for r in rows if not r.get("ok") and r.get("class") in (2, 3)]
    print(f"replaying {len(misses)} MISS configs at 512 tokens")
    still = 0
    for i in misses:
        rng = random.Random(i)
        cls = i % 4
        suffix = "-M" if cls == 2 else "-L"
        code = f"{rng.randrange(10**6):06d}{suffix}"
        if cls == 2:
            pre_n, post_n = rng.randrange(1400, 5600), rng.randrange(1400, 5600)
        else:
            pre_n, post_n = rng.randrange(3000, 9000), rng.randrange(3000, 9000)
        prompt = fillers(pre_n, i) + f" The access phrase is {code}. " + fillers(post_n, i + 500 if cls == 2 else i + 900)
        wall, text = run(prompt, 512)
        ok = code in text
        still += 0 if ok else 1
        print(f"[{i:02d} c{cls}] ptok~{pre_n+post_n} wall={wall:.1f}s {'OK' if ok else 'REAL-MISS'} "
              f"finished_think={'</think>' in text} len={len(text)}")
        time.sleep(0.4)
    print(f"GATED-REPLAY: {len(misses)-still}/{len(misses)} PASS")

if __name__ == "__main__":
    main()
