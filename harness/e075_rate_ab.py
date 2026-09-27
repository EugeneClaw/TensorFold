#!/usr/bin/env python3
"""E-075 rate A/B: 20 fresh word-salad needle configs per keep setting.
Usage: e075_rate_ab.py <tag>  (SOAK_BASE env overrides endpoint)"""
import json, os, time, random, urllib.request, sys

TAG = sys.argv[1]
BASE = os.environ.get("SOAK_BASE", "http://127.0.0.1:8008/v1/chat/completions")
FILLER = ("harbor ledger meridian lantern compass granite thistle fossil amber copper drizzle "
          "fabric garnet hollow ingress jade kernel lumen masonry nominal orchard prism quarry "
          "ribbon saffron tundra umber velvet willow zephyr anchor beacon cipher delta ember "
          "flint glade helium iris jasper karst lagoon masonry opal pumice quartz reed slate "
          "tundra vellum wicket xenon yarrow zither anvil basalt cobalt dune ember firth").split()

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
    okn, walls = 0, []
    for i in range(100, 120):
        rng = random.Random(i)
        code = f"{rng.randrange(10**6):06d}-R"
        pre_n = rng.randrange(2500, 8000)
        post_n = rng.randrange(2500, 8000)
        prompt = fillers(pre_n, i) + f" The access phrase is {code}. " + fillers(post_n, i + 77)
        wall, text = run(prompt)
        ok = code in text
        okn += ok
        walls.append(wall)
        print(f"[{i} {TAG}] ~{(pre_n+post_n)//1}w wall={wall:.1f}s {'OK' if ok else 'MISS'}")
        time.sleep(0.3)
    print(f"RATE {TAG}: {okn}/20 PASS  med_wall={sorted(walls)[10]:.1f}s")

if __name__ == "__main__":
    main()
