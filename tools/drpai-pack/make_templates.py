#!/usr/bin/env python3
"""Emit the ONNX + calibration inputs for shape templates: one t_<K>x<N>_m<M>.onnx per (K, N) and M in (2, 256).
usage: make_templates.py <template dir> KxN [KxN ...]      e.g. 1024x2048 1024x1024 2048x1024 1024x3072 3072x1024 1024x32768 1024x20864
The weights are random; only the shape matters (the backend swaps the weights in at load time)."""
import sys, os, numpy as np, torch
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
class Lin(torch.nn.Module):
    def __init__(s_, w): super().__init__(); s_.w = torch.nn.Parameter(w)
    def forward(s_, x): return x @ s_.w.T
rng = np.random.default_rng(7)
for kn in sys.argv[2:]:
    K, N = map(int, kn.split("x"))
    W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32) * (2.0 ** rng.uniform(-3, 3, N)).astype(np.float32)[:, None]
    for M in (2, 256):
        torch.onnx.export(Lin(torch.from_numpy(W)), torch.zeros(M, K), f"{out}/t_{K}x{N}_m{M}.onnx", opset_version=13,
                          input_names=["x"], output_names=["y"], do_constant_folding=True, dynamo=False)
        cal = rng.standard_normal((16, M, K)).astype(np.float32); cal[0, 0, :2] = (4, -4)
        np.save(f"{out}/calib_{K}_m{M}.npy", cal)
    print(f"t_{K}x{N}_m{{2,256}}")
