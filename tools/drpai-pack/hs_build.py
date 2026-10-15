#!/usr/bin/env python3
"""Build a drpai backend package for any HF causal LM from shape templates, without the DRP-AI Translator,
quantizer or docker.  Same recipe and package layout as build_model.py + compile_model.sh.
usage: hs_build.py <hf model id> <out dir> <calibration text file>
The package holds only packed weights + scales + CPU sidecars; shape templates live separately (see README.md)."""
import sys, os, json, struct, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hotswap import quantize_weight
from drpai_pack import pack_weight
from transformers import AutoModelForCausalLM, AutoTokenizer
mid, out, calib_txt = sys.argv[1], sys.argv[2], sys.argv[3]
KI, KO, G, OB, PF, CHUNK = 16, 4, 8.0, 0.5, 256, 32768
t0 = time.time()
tok = AutoTokenizer.from_pretrained(mid)
m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.float32).eval()
ids = tok(open(calib_txt, encoding="utf-8", errors="ignore").read()[:60000], return_tensors="pt").input_ids[0][:1024]
names = {}
for i, l in enumerate(m.model.layers):
    for n, sub in (("attn_q", l.self_attn.q_proj), ("attn_k", l.self_attn.k_proj), ("attn_v", l.self_attn.v_proj), ("attn_output", l.self_attn.o_proj),
                   ("ffn_gate", l.mlp.gate_proj), ("ffn_up", l.mlp.up_proj), ("ffn_down", l.mlp.down_proj)):
        names[f"blk.{i}.{n}.weight"] = sub
names["output.weight"] = m.lm_head
# fused objects: q|k|v share the input, gate|up share the input -> one DRP job each (the backend splits the output rows)
FUSED = {"attn_qkv": ("attn_q", "attn_k", "attn_v"), "ffn_gateup": ("ffn_gate", "ffn_up")} if os.environ.get("HS_FUSED") else {}   # measured: no gain, DRP time is pure weight bandwidth
store = {k: [] for k in names}
hooks = [sub.register_forward_hook(lambda mod, inp, o, k=k: store[k].append(inp[0].detach().reshape(-1, inp[0].shape[-1]).clone())) for k, sub in names.items()]
with torch.no_grad():
    for i in range(0, 1024, 256): m(ids[i:i + 256][None])
for h in hooks: h.remove()
print(f"calibration done {time.time() - t0:.0f}s", flush=True)
def q(x, a_s): return np.clip(np.round(x / a_s), -127, 127) * a_s
LMH_KO, LMH_YM = 16, 2.0                     # lm_head: more CPU-side rows and 2x output headroom (clipped logits cost ~17% ppl on hardware)
def emit(name, W, X, ko=KO, ym=1.0):
    N, K = W.shape
    in_idx = np.argsort(-np.abs(X).max(0))[:KI]; out_idx = np.argsort(-np.abs(X @ W.T).max(0))[:ko]
    Wd = W.copy(); Wd[:, in_idx] = 0; Wd[out_idx, :] = 0
    Xd = X.copy(); Xd[:, in_idx] = 0
    mo = np.abs(Xd @ Wd.T).max(0); mo[out_idx] = 1; mo = mo ** OB * mo.max() ** (1 - OB)
    r = (1.0 / np.maximum(mo, 1e-6)).astype(np.float32); Wd = Wd * r[:, None]
    a_s = float(np.abs(Xd).max() / 127)
    h = q(Xd, a_s); rows = np.concatenate([h, q((Xd - h) * G, a_s)])
    out_s = float(np.abs(rows @ Wd.T).max() / 127) * ym
    Wq, ws = quantize_weight(Wd)
    d = f"{out}/{name}"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/weight.bin", "wb").write(pack_weight(Wq, ws, a_s, out_s))      # pool image, shared by all M templates
    open(f"{d}/scales.bin", "wb").write(struct.pack("<ff", a_s, out_s))
    with open(f"{d}/sidecar.bin", "wb") as f:
        f.write(b"DRPL" + struct.pack("<IIIIIff", 1, K, N, KI, ko, G, a_s))
        f.write(in_idx.astype(np.int32).tobytes()); f.write(out_idx.astype(np.int32).tobytes())
        f.write(np.ascontiguousarray(W[:, in_idx]).astype(np.float32).tobytes()); f.write(np.ascontiguousarray(W[out_idx, :]).astype(np.float32).tobytes())
        f.write(r.tobytes())
    return {"name": name, "K": K, "N": N}
manifest = []
for i in range(len(m.model.layers)):
    for fname, parts in FUSED.items():
        W = np.concatenate([names[f"blk.{i}.{p}.weight"].weight.detach().numpy().astype(np.float32) for p in parts])
        X = torch.cat(store[f"blk.{i}.{parts[0]}.weight"]).numpy()
        manifest.append(emit(f"blk.{i}.{fname}.weight", W, X)); manifest[-1]["parts"] = [names[f"blk.{i}.{p}.weight"].weight.shape[0] for p in parts]
        open(f"{out}/blk.{i}.{fname}.weight/parts.txt", "w").write("".join(f"{p} {names[f'blk.{i}.{p}.weight'].weight.shape[0]}\n" for p in parts))
        print(f"blk.{i}.{fname}.weight", W.shape, flush=True)
for k, sub in names.items():
    if any(k.endswith(f".{p}.weight") for parts in FUSED.values() for p in parts): continue
    W = sub.weight.detach().numpy().astype(np.float32); X = torch.cat(store[k]).numpy()
    if k == "output.weight":
        for j in range(0, W.shape[0], CHUNK):
            manifest.append(emit(f"output.weight.c{j // CHUNK}", W[j:j + CHUNK], X, LMH_KO, LMH_YM)); manifest[-1]["row0"] = j
    else:
        manifest.append(emit(k, W, X))
json.dump({"model": mid, "ki": KI, "ko": KO, "g": G, "ob": OB, "pf": PF, "chunk": CHUNK, "builder": "hotswap-pool", "objects": manifest},
          open(f"{out}/manifest.json", "w"), indent=1)
print(f"package {out}: {len(manifest)} weights, {time.time() - t0:.0f}s total")
