"""E-079 claim-1 probe: zero-latent K/V through the quantized wk/wv of layer 11 MLA.

Question (Minimax review #1): do gap keys contribute a bias term, or exactly zero?
`mx.quantized_matmul(lat, W, scales, biases)` — MLX `biases` are quantization
zero-points applied to dequantize W, NOT an additive output bias. A zero input row
must produce exactly zero output. If true, prior-window gap keys have logit 0 and
value 0 — they sit in the softmax with weight exp(0)=1 each, but contribute NOTHING
to the value mixture (v=0). The effect is pure softmax-mass dilution, magnitude
measured below; NOT "bias-only K/V" as the review claims.
"""
import json
from pathlib import Path
import mlx.core as mx
from safetensors import safe_open

SNAP = Path("/Users/studio/.cache/huggingface/hub/models--Vontra--GLM-5.3-Flash-MLX-4bit-MTP/snapshots/76add2a341a1cd90ad0e86bb69839ea9c35827c6")
idx = json.load(open(SNAP / "model.safetensors.index.json"))
wm = idx["weight_map"]
L = "model.language_model.layers.11.self_attn."
need = [L + "kv_a_proj_with_mqa.weight", L + "kv_a_proj_with_mqa.scales", L + "kv_a_proj_with_mqa.biases",
        L + "kv_a_layernorm.weight"]
shards = {}
t = {}
import numpy as np

for k in need:
    s = wm[k]
    if s not in shards:
        shards[s] = safe_open(str(SNAP / s), framework="np")
    try:
        arr = shards[s].get_tensor(k)
    except TypeError:
        # numpy can't view bf16 — read raw file bytes and fix up uint16 -> float32
        import numpy as _np
        import json as _json
        import struct
        with open(str(SNAP / s), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = _json.loads(f.read(n))
            header.pop("__metadata__", {})
            ent = header[k]
            assert ent["dtype"] == "BF16", ent["dtype"]
            f.seek(8 + n + ent["data_offsets"][0])
            raw = f.read(ent["data_offsets"][1] - ent["data_offsets"][0])
        u16 = _np.frombuffer(raw, dtype=_np.uint16)
        u32 = (u16.astype(_np.uint32) << 16)
        arr = u32.view(_np.float32).reshape(ent["shape"])
    t[k] = arr
wq = mx.array(np.asarray(t[L + "kv_a_proj_with_mqa.weight"]).view(np.uint32))
sc = mx.array(np.asarray(t[L + "kv_a_proj_with_mqa.scales"], dtype=np.float32))
bi = mx.array(np.asarray(t[L + "kv_a_proj_with_mqa.biases"], dtype=np.float32))
ln_w = mx.array(np.asarray(t[L + "kv_a_layernorm.weight"], dtype=np.float32))
print("kv_a W:", wq.shape, wq.dtype, "| scales:", sc.shape, "| biases:", bi.shape)

D = wq.shape[1] * 8  # packed uint32 (d_out, d_in/8): expanded weight is (4096, 512) -> x needs d_in=4096
lat0 = mx.zeros((1, D), mx.bfloat16)
z = mx.quantized_matmul(lat0, wq, sc, bi, transpose=True, group_size=64, bits=4)
mx.eval(z)
print("zero-latent kv_a out max-abs:", float(mx.max(mx.abs(z)).item()))
# 2. zero latent through kv_norm first (as spec_feed's scatter path effectively does:
#    scattered zeros then rms_norm would be NaN — but spec_feed scatters AFTER norm,
#    so gap latents are literal zeros in cache)
# real-magnitude comparison:
mx.random.seed(3)
latr = mx.fast.rms_norm(mx.random.normal((1, D)).astype(mx.bfloat16), ln_w[:1], 1e-6) if False else mx.random.normal((1, D)).astype(mx.bfloat16)
r = mx.quantized_matmul(latr, wq, sc, bi, transpose=True, group_size=64, bits=4)
mx.eval(r)
rm = float(mx.max(mx.abs(r)).item())
print("typical kv_a out max-abs:", rm)
# 3. softmax dilution quantification: n_gap zero-logit keys among n_real real keys
import math
for n_gap, n_real in ((25000, 38000), (90000, 38000)):
    # logit scale: typical q.k for this arch, assume 1.0 sigma ~ 3.0 nats peak
    # zero keys contribute exp(0) each; real keys exp(l) with |l| up to ~20
    # fraction = n_gap / (n_gap + sum exp(l_i)·avg)
    # honest bound: even with avg real logit just 2 nats above 0:
    frac_optimistic = n_gap / (n_gap + n_real * math.exp(2.0))
    frac_conservative = n_gap / (n_gap + n_real * math.exp(0.5))
    print(f"n_gap={n_gap}: mass fraction on gap keys between "
          f"{frac_optimistic:.2%} (real logits +2 nats) and {frac_conservative:.2%} (real +0.5 nats)")
