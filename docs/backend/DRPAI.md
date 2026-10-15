# llama.cpp on the Renesas RZ/V2H DRP-AI

This backend runs every linear layer of a model on the DRP-AI3 of the RZ/V2H (tested on the Kakip board,
Ubuntu 26.04, Renesas AI SDK v6.00 runtime).  Attention, norms, RoPE, activations and the embedding lookup stay
on the Cortex-A55 cores.  Weights live in DRP-AI memory as int8 and are loaded from a *model package* built by
`tools/drpai-pack/hs_build.py`; the DRP objects themselves are *shape templates* compiled once per matrix shape
with the Renesas DRP-AI Translator, so changing models needs no Translator run.

Not affiliated with or endorsed by Renesas.

## Results

Qwen3-0.6B, Q8_0 GGUF for the CPU side, Kakip (4 x A55 at ~550 MHz effective, DRP-AI area 2 GB).

| | CPU only | DRP-AI |
|---|---|---|
| decode | 7.0 tok/s | 5.6 tok/s |
| prefill (149 tokens) | 28 tok/s | 34 tok/s |
| perplexity (pg11, 4 x 512) | 19.18 | 19.36 |
| CPU load during decode | 4 cores | glue only |

DRP time is pure weight streaming (~5 GB/s int8): 575 MB per token for 0.6B gives a ~8.7 tok/s ceiling.

## Requirements

- Kernel with the `drpai` driver and a DRP-AI reserved region large enough for the templates plus the packed
  weights (`DRP-AI@1C0000000`, 2 GB, in the Kakip device tree).
- The DRP-AI TVM runtime libraries from the AI SDK (`libmera2_runtime.so`, `libmera2_plan_io.so`,
  `libdrp_tvm_rt.so`, `libdrp_rt.so`, `libacl_rt.so`, `libarm_compute*.so`, `libtvm_runtime.so`) in the library
  path, and a checkout of https://github.com/renesas-rz/rzv_drp-ai_tvm for the headers at build time.
- Shape templates and a model package, see `tools/drpai-pack/README.md`.

## Build

    cmake -B build -DGGML_DRPAI=ON -DGGML_DRPAI_TVM_DIR=/path/to/rzv_drp-ai_tvm -DGGML_DRPAI_LIB_DIR=/usr/local/lib
    cmake --build build --target llama-cli llama-completion llama-perplexity

## Run

    GGML_DRPAI_MODEL=~/models/qwen3-0.6b GGML_DRPAI_TEMPLATES=~/drpai-tpl \
      build/bin/llama-cli -m qwen3-0.6b-q8_0.gguf -c 2048 -st -p "Introduce yourself. /no_think"

Environment: `GGML_DRPAI_THREADS` (CPU glue threads, default 4), `GGML_DRPAI_PROF=1` (per-phase times at exit),
`GGML_DRPAI_SKIP=substr,...` (keep matching weights on the CPU), `GGML_DRPAI_NO_RAW=1` (use the runtime's `Run()`
for every object instead of the direct driver path used for prefill).

## How it works

- `drpai_pack.py` reproduces the Translator's dense int8 MatMul weight layout and per-channel requantization
  encoding byte for byte (verified on 32 shapes), so a template compiled for one weight can run any weight.
- At load the backend writes each packed weight into a pool at the top of the DRP-AI area and, before a run,
  repoints the template's AI-MAC parameter commands at the pool and rewrites the activation scales.
- Accuracy: the DRP-AI quantizes activations with static per-tensor int8 scales.  The package keeps 16 outlier
  input columns and 4 output rows per layer on the CPU in fp32, equalizes output channels into the weights, and
  feeds each token as two int8 rows (value and residual x 8) so the DRP effectively sees an 11-bit input.
