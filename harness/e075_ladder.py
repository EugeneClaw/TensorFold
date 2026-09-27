#!/usr/bin/env python3
"""E-075 depth ladder: verbatim-code needle at sizes 6K/12K/24K/30K x depths 25/50/75%.
Runs against server on :8010 (spec per server env). PASS = exact code in response."""
import json, time, hashlib, random, urllib.request

BASE = "http://127.0.0.1:8010/v1/chat/completions"
MODEL = "76add2a341a1cd90ad0e86bb69839ea9c35827c6"
FILLER_WORDS = ("harbor ledger meridian lantern compass granite thistle harbor fossil amber "
                "copper drizzle fabric garnet hollow ingress jade kernel lumen masonry nominal "
                "orchard prism quarry ribbon saffron tundra umber velvet willow zephyr anchor "
                "beacon cipher delta ember flint glade helium iris jasper karst lagoon").split()

def fillers(n_words, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n_words:
        out.append(rng.choice(FILLER_WORDS))
        if len(out) % 9 == 0:
            out[-1] += "."
    return " ".join(out)

def build_unique(code, target_tokens, nonce):
    # calibrated: requested n -> ~1.785n actual tokens on GLM tokenizer
    chars = int(target_tokens * 4 / 1.785)
    n = max(200, chars // 7)
    parts = [fillers(n, seed=hash((nonce, "a")) & 0xFFFF)]
    parts.append(f" RECORD-CODE {code}. Do not lose this code. ")
    parts.append(fillers(n, seed=hash((nonce, "b")) & 0xFFFF))
    return "".join(parts)

def run(prompt, max_tokens=64):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0, "stream": False}).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    return time.time() - t0, d["usage"]["prompt_tokens"], d["choices"][0]["message"]["content"]

def main():
    nonce = f"{time.time():.3f}"
    results = []
    for size in (6000, 12000, 24000, 30000):
        code = f"{hash((nonce, size)) & 0xFFFFFF:06d}-{size}"
        for depth in (0.25, 0.5, 0.75):
            pos = int(size * depth)
            pre = build_unique(code, pos, (nonce, size, "pre"))
            post = build_unique(code, size - pos, (nonce, size, "post"))
            prompt = pre + f" The record code is {code}. " + post
            wall, ptok, text = run(prompt)
            ok = code in text
            results.append({"size": size, "depth": depth, "ptok": ptok, "wall_s": round(wall, 2),
                            "pass": ok, "head": text[:80]})
            print(f"[{size:5d} tok @ {int(depth*100):3d}%] ptok={ptok} wall={wall:5.1f}s "
                  f"{'PASS' if ok else 'FAIL'} head={text[:60]!r}")
            time.sleep(0.3)
    npass = sum(r["pass"] for r in results)
    print(f"LADDER: {npass}/{len(results)} PASS")
    json.dump({"nonce": nonce, "results": results}, open("/tmp/e075_ladder.json", "w"), indent=1)

if __name__ == "__main__":
    main()
