#!/usr/bin/env python3
"""Build a drpai package from a GGUF (weights, in the exact row order llama.cpp expects) plus an HF model for the
calibration activations.  usage: hs_build_gguf.py <model.gguf> <hf model id> <out dir> <calibration text> [--fp16-only]
Supported: Qwen3 (attn_q/k/v/output, ffn_*, output) and Qwen3.5 (adds attn_qkv, attn_gate, ssm_out).
Tensors with no calibration hook or N < 256 (ssm_alpha/beta) are left to the CPU."""
import sys, os, re, struct, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
from hotswap import quantize_weight
from drpai_pack import pack_weight
from gguf import GGUFReader
from gguf.quants import dequantize
from transformers import AutoModelForCausalLM, AutoTokenizer
gguf_path, mid, out, calib_txt = sys.argv[1:5]
KI, KO, G, OB, CHUNK = 16, 4, 8.0, 0.5, 32768
LMH_KO, LMH_YM = 16, 2.0
t0 = time.time()
# --- GGUF tensor -> HF module whose *input* is this tensor's input
HOOKS = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.q_proj", "attn_v": "self_attn.q_proj", "attn_output": "self_attn.o_proj",
         "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.gate_proj", "ffn_down": "mlp.down_proj",
         "attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_qkv", "ssm_out": "linear_attn.out_proj"}
r = GGUFReader(gguf_path)
tensors = {}
for t in r.tensors:
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", t.name)
    if (m and m.group(2) in HOOKS) or t.name in ("output.weight", "token_embd.weight"):
        if len(t.shape) != 2: continue
        tensors[t.name] = t
tok = AutoTokenizer.from_pretrained(mid)
model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
layers = model.model.layers if hasattr(model.model, "layers") else model.model.language_model.layers
ids = tok(open(calib_txt, encoding="utf-8", errors="ignore").read()[:60000], return_tensors="pt").input_ids[0][:1024]
store = {}
def hook(key):
    def f(mod, inp, outp): store.setdefault(key, []).append(inp[0].detach().reshape(-1, inp[0].shape[-1]).clone())
    return f
handles = []
for name in tensors:
    m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", name)
    if not m: continue
    i, kind = int(m.group(1)), m.group(2)
    mod = layers[i]
    for a in HOOKS[kind].split("."):
        mod = getattr(mod, a, None)
        if mod is None: break
    key = f"{i}.{HOOKS[kind]}"
    if mod is not None and key not in store: store[key] = []; handles.append(mod.register_forward_hook(hook(key)))
store["lm_head"] = []; handles.append(model.lm_head.register_forward_hook(hook("lm_head")))
with torch.no_grad():
    for i in range(0, len(ids), 256): model(ids[i:i + 256][None])
for h in handles: h.remove()
print(f"calibration done {time.time() - t0:.0f}s, {len(store)} hook points", flush=True)
def q(x, a_s): return np.clip(np.round(x / a_s), -127, 127) * a_s
def emit(name, W, X, ko=KO, ym=1.0):
    N, K = W.shape
    in_idx = np.argsort(-np.abs(X).max(0))[:KI]; out_idx = np.argsort(-np.abs(X @ W.T).max(0))[:ko]
    Wd = W.copy(); Wd[:, in_idx] = 0; Wd[out_idx, :] = 0
    Xd = X.copy(); Xd[:, in_idx] = 0
    mo = np.abs(Xd @ Wd.T).max(0); mo[out_idx] = 1; mo = mo ** OB * mo.max() ** (1 - OB)
    rr = (1.0 / np.maximum(mo, 1e-6)).astype(np.float32); Wd = Wd * rr[:, None]
    a_s = float(np.abs(Xd).max() / 127)
    h = q(Xd, a_s); rows = np.concatenate([h, q((Xd - h) * G, a_s)])
    out_s = float(np.abs(rows @ Wd.T).max() / 127) * ym
    Wq, ws = quantize_weight(Wd)
    d = f"{out}/{name}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/weight.bin", "wb").write(pack_weight(Wq, ws, a_s, out_s))
    open(f"{d}/scales.bin", "wb").write(struct.pack("<ff", a_s, out_s))
    with open(f"{d}/sidecar.bin", "wb") as f:
        f.write(b"DRPL" + struct.pack("<IIIIIff", 1, K, N, KI, ko, G, a_s))
        f.write(in_idx.astype(np.int32).tobytes()); f.write(out_idx.astype(np.int32).tobytes())
        f.write(np.ascontiguousarray(W[:, in_idx]).astype(np.float32).tobytes()); f.write(np.ascontiguousarray(W[out_idx, :]).astype(np.float32).tobytes())
        f.write(rr.tobytes())
n = 0
for name, t in tensors.items():
    W = dequantize(t.data, t.tensor_type).reshape(int(t.shape[1]), int(t.shape[0])).astype(np.float32)   # (N, K)
    if name == "token_embd.weight":
        if "output.weight" in tensors: continue
        X = torch.cat(store["lm_head"]).numpy(); gname = "output.weight"
    elif name == "output.weight":
        X = torch.cat(store["lm_head"]).numpy(); gname = name
    else:
        m = re.match(r"blk\.(\d+)\.(\w+)\.weight$", name); key = f"{m.group(1)}.{HOOKS[m.group(2)]}"
        if key not in store or not store[key]: print("no calibration for", name, "-> CPU"); continue
        X = torch.cat(store[key]).numpy(); gname = name
    if W.shape[0] < 256: print("skip", name, W.shape, "-> CPU"); continue
    if gname == "output.weight":
        for j in range(0, W.shape[0], CHUNK): emit(f"output.weight.c{j // CHUNK}", W[j:j + CHUNK], X, LMH_KO, LMH_YM)
    else: emit(gname, W, X)
    n += 1; print(gname, W.shape, flush=True)
print(f"package {out}: {n} weights, {time.time() - t0:.0f}s total")
