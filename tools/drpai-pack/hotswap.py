#!/usr/bin/env python3
"""Build a DRP-AI object for a new weight matrix by patching a shape template (no Translator, no quantizer).
Template dir (from tpl/build_templates.sh): deploy.{so,json,params}, weight.bin, drp_param.bin, drp_param_info.txt."""
import os, re, shutil, numpy as np
from drpai_pack import pack_weight, fp16_bits

class Template:
    def __init__(self, d):
        self.d = d
        self.so = open(f"{d}/deploy.so", "rb").read()
        w = open(f"{d}/weight.bin", "rb").read(); p = open(f"{d}/drp_param.bin", "rb").read()
        self.w_off, self.w_len = self.so.find(w), len(w)
        self.p_off, self.p_len = self.so.find(p), len(p)
        assert self.w_off > 0 and self.so.count(w) == 1 and self.p_off > 0, "template blobs not found once in deploy.so"
        info = open(f"{d}/drp_param_info.txt").read()
        lay = re.findall(r"OFFSET_ADD:(\d+),layer_name:([^,]+),drp_lib:(\w+)", info)
        sc = [int(x) for x in re.findall(r"Param:SCALE,\s+Value:\d+,\s+offset:(\d+)", info)]
        q = [int(o) for o, n, lib in lay if lib == "quantize"]
        assert len(q) == 2 and len(sc) == 2, "expected one input and one output quantize layer"
        self.in_scale_off, self.out_scale_off = q[0] + sc[0], q[1] + sc[1]

    def build(self, Wq, ws, in_s, out_s, out_dir):
        """Wq int8 (K,N) in ONNX transpose_kernel2 layout, ws (N,) per-channel scales."""
        wbin = pack_weight(Wq, ws, in_s, out_s)
        assert len(wbin) == self.w_len, f"weight.bin size {len(wbin)} != template {self.w_len}"
        so = bytearray(self.so)
        so[self.w_off:self.w_off + self.w_len] = wbin
        for off, s in ((self.in_scale_off, in_s), (self.out_scale_off, out_s)):
            so[self.p_off + off:self.p_off + off + 2] = fp16_bits(s).to_bytes(2, "little")
        os.makedirs(out_dir, exist_ok=True)
        open(f"{out_dir}/deploy.so", "wb").write(so)
        for f in ("deploy.json", "deploy.params"): shutil.copy(f"{self.d}/{f}", out_dir)

def quantize_weight(W):
    """W float (N,K) -> (Wq int8 (K,N), ws (N,)): symmetric per-output-channel 8-bit."""
    ws = (np.abs(W).max(1) / 127.0).astype(np.float32)
    ws[ws == 0] = np.median(ws[ws > 0])                   # all-zero rows (moved to the CPU): any sane scale
    Wq = np.clip(np.round(W / ws[:, None]), -127, 127).astype(np.int8)
    return np.ascontiguousarray(Wq.T), ws
