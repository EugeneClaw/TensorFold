#!/usr/bin/env python3
"""E-075 keep-sweep on 5 confirmed-miss configs: does raising keep recover retrieval?
Run once per keep value (server restarted with TF_SPEC_KEEP=N between runs)."""
import json, os, time, random, urllib.request, sys

KEEP = sys.argv[1]
BASE = os.environ.get("SOAK_BASE", "http://127.0.0.1:8008/v1/chat/completions")
FILLER = ("harbor ledger meridian lantern compass granite thistle fossil amber copper drizzle "
          "fabric garnet hollow ingress jade kernel lumen masonry nominal orchard prism quarry "
          "ribbon saffron tundra umber velvet willow zephyr anchor beacon cipher delta ember "
          "flint glade helium iris jasper karst lagoon masonry opal pumice quartz reed slate").split()
CONFIGS = [3, 7, 26, 34, 51]

def fillers(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out.append(rng.choice(FILLER))
        if len(out) % 11 == 0:
            out[-1] += "."
    return " ".join(out)

def run(prompt, max_tokens=512):
    body = json.dumps({"model": "GLM-5.3-Flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return time.time() - t0, d["choices"][0]["message"]["content"]

def main():
    okn = 0
    for i in CONFIGS:
        rng = random.Random(i)
        cls = i % 4
        suffix = "-M" if cls == 2 else "-L"
        code = f"{rng.randrange(10**6):06d}{suffix}"
        if cls == 2:
            pre_n, post_n = rng.randrange(1400, 5600), rng.randrange(1400, 5600)
            post_seed = i + 500
        else:
            pre_n, post_n = rng.randrange(3000, 9000), rng.randrange(3000, 9000)
            post_seed = i + 900
        prompt = fillers(pre_n, i) + f" The access phrase is {code}. " + fillers(post_n, post_seed)
        wall, text = run(prompt)
        ok = code in text
        okn += ok
        print(f"[{i:02d} keep={KEEP}] wall={wall:.1f}s {'OK' if ok else 'MISS'} finished_think={'</think>' in text}")
        time.sleep(0.4)
    print(f"KEEP={KEEP}: {okn}/{len(CONFIGS)} PASS")

if __name__ == "__main__":
    main()
