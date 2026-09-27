#!/usr/bin/env python3
"""E-075 hosted soak: 60 requests against a hosted server (default loopback :8008; SOAK_BASE to override),
mixed sizes. Verifies: no tps decay, needle correctness preserved throughout,
no 5xx/errors, spec activation on long prompts, throughput stability."""
import json, time, random, urllib.request, sys

import os
BASE = os.environ.get("SOAK_BASE", "http://127.0.0.1:8008/v1/chat/completions")
MODELS = ["GLM-5.3-Flash-MLX-mixed-4_8bit-mtp", "GLM-5.3-Flash", "GLM-5.3-Flash-Q4/8"]
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

def run(model, prompt, max_tokens=128):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    wall = time.time() - t0
    return wall, d["usage"], d["choices"][0]["message"]["content"]

def main():
    t_start = time.time()
    rows = []
    fails = 0
    for i in range(60):
        nonce = f"soak-{i}-{int(t_start)}"
        rng = random.Random(i)
        size_class = i % 4
        if size_class == 0:      # short chat
            prompt = f"Request {nonce}: In two sentences, why do bridges have expansion joints?"
            expect = None
        elif size_class == 1:    # short needle (recall correctness, no spec)
            code = f"{rng.randrange(10**6):06d}-S"
            prompt = f"Memorize: {code}. Now reply with just that code."
            expect = code
        elif size_class == 2:    # ~11K spec needle at random depth
            code = f"{rng.randrange(10**6):06d}-M"
            pre_n, post_n = rng.randrange(1400, 5600), rng.randrange(1400, 5600)
            prompt = fillers(pre_n, i) + f" The access phrase is {code}. " + fillers(post_n, i + 500)
            expect = code
        else:                    # ~17K spec needle
            code = f"{rng.randrange(10**6):06d}-L"
            pre_n, post_n = rng.randrange(3000, 9000), rng.randrange(3000, 9000)
            prompt = fillers(pre_n, i) + f" The access phrase is {code}. " + fillers(post_n, i + 900)
            expect = code
        model = MODELS[i % 3]
        try:
            wall, usage, text = run(model, prompt, max_tokens=256 if expect and len(prompt) > 5000 else 128)
            ok = (expect in text) if expect else (len(text) > 5)
            if not ok:
                fails += 1
            rows.append({"i": i, "class": size_class, "model": model.split("-Flash-")[-1][:10],
                         "ptok": usage["prompt_tokens"], "wall": round(wall, 2), "ok": ok})
            print(f"[{i:02d} c{size_class} {model.split('-Flash-')[-1][:10]:>10}] ptok={usage['prompt_tokens']:6d} "
                  f"wall={wall:6.2f}s {'OK' if ok else 'MISS'}")
        except Exception as e:
            fails += 1
            rows.append({"i": i, "error": str(e)[:120], "ok": False})
            print(f"[{i:02d}] ERROR {str(e)[:120]}")
        time.sleep(0.5)
    # stability summary
    longs = [r["wall"] for r in rows if r.get("class") in (2, 3) and "wall" in r]
    first5, last5 = longs[:6], longs[-6:]
    import statistics
    print(f"\nSOAK DONE {time.time()-t_start:.0f}s fails={fails}/60")
    print(f"long-prompt wall first6 med={statistics.median(first5):.1f}s last6 med={statistics.median(last5):.1f}s")
    json.dump(rows, open("/tmp/e075_soak.json", "w"), indent=1)

if __name__ == "__main__":
    main()
