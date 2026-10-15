#!/usr/bin/env python3
"""Re-implementation of the DRP-AI Translator i8 v1.12 weight packer for a dense int8 MatMul (sparse off),
recovered from compiled examples.  pack_weight() returns the bytes of weight.bin; fp16 scales go into drp_param."""
import numpy as np

SFT_MAX = 32

def expr_scale_hw(v, bit_mul=15):
    """float multiplier -> (mant, shift): mant = floor(v * 2^shift) with 2^(bit_mul-1) <= mant < 2^bit_mul and
    shift <= 32, then strip trailing zero bits (Translator ExprScaleHW.from_float + _adjust_unnecessary_shift)."""
    v = float(np.float32(v))
    e = int(np.floor(np.log2(v)))
    sh = (bit_mul - 1) - e
    mant = int(np.floor(v * 2.0 ** sh))
    if mant >= 1 << bit_mul: mant >>= 1; sh -= 1
    while mant and mant % 2 == 0: mant >>= 1; sh -= 1
    if sh > SFT_MAX: sh = SFT_MAX; mant = int(np.floor(v * 2.0 ** sh))
    if sh < 0: mant <<= -sh; sh = 0
    return mant, sh

def split_shift(mant, sh):
    """total right shift -> (rsft_alu a10, rsft_add32 a11): a10 <= 16, a10 <= bits(mant)+7, a11 <= 16."""
    a10 = min(16, mant.bit_length() + 7, sh)
    a10 = max(a10, sh - 16)
    return a10, sh - a10

OCH_TILE = 256
ICH_SLICE = 4096                                           # ponytail: seen K<=3072 -> 1 slice, K=6144 -> 2; exact limit unverified

def pack_weight(Wq, ws, in_s, out_s):
    """Wq int8 (K, N) as in the quantized ONNX; ws (N,) per-channel weight scales.  The output channels are cut into
    tiles of 256; weight.bin = all tile bodies, then all tile PPU arrays, then all tile tails."""
    K, N = Wq.shape
    parts = [_pack_tile(Wq[:, t:t + OCH_TILE], ws[t:t + OCH_TILE], in_s, out_s) for t in range(0, N, OCH_TILE)]
    n_kslice = -(-K // ICH_SLICE)                          # K > ICH_SLICE: one pass-through tail record per extra slice
    if n_kslice > 1:
        z = np.zeros(64, np.uint16); z[8] = 1
        parts = [(b, a, z.tobytes() * (n_kslice - 1) + t) for b, a, t in parts]
    return b"".join(p[0] for p in parts) + b"".join(p[1] for p in parts) + b"".join(p[2] for p in parts)

def _pack_tile(Wq, ws, in_s, out_s):
    K, N = Wq.shape
    body = Wq.reshape(K // 2, 2, N).transpose(0, 2, 1).astype(np.int8).tobytes()      # [k-pair][n][2]
    M = (np.float32(in_s) * ws.astype(np.float32) / np.float32(out_s)).astype(np.float32)
    arr = np.zeros((12, N), np.uint16)
    for n in range(N):
        mant, sh = expr_scale_hw(M[n])
        a10, a11 = split_shift(mant, sh)
        while a11 == 0 and mant.bit_length() < 15:          # a11=0 puts 2^15 in the low half of the rounding word, which
            mant <<= 1; sh += 1                               # the PPU reads as signed (-32768); same value, a11 >= 1
            a10, a11 = split_shift(mant, sh)
        r = 1 << (15 + a11)
        arr[0, n] = (r >> 16) & 0xff; arr[1, n] = (r >> 24) & 0xff; arr[2, n] = r & 0xff; arr[3, n] = (r >> 8) & 0xff
        arr[4, n] = mant & 0xff; arr[5, n] = mant >> 8
        v = 1 << (a10 - 1) if a10 else 0                   # a6/a7 = 16-bit 2^(a10-1), low/high byte
        arr[6, n] = v & 0xff; arr[7, n] = v >> 8; arr[10, n] = a10; arr[11, n] = a11
    tail = np.zeros(64, np.uint16)
    tail[0] = tail[1] = arr[11, 0]; tail[2] = tail[3] = arr[10, 0]; tail[8] = 1
    return body, arr.tobytes(), tail.tobytes()

def fp16_bits(x): return int(np.array(x, np.float16).view(np.uint16))
