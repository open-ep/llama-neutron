# llama.cpp on the NXP i.MX 95 Neutron NPU

This fork offloads llama.cpp's 4-bit matrix multiplies to the eIQ Neutron NPU in the NXP i.MX 95.
It hooks in as a ggml **CPU extra buffer type** (the same mechanism AMX and KleidiAI use): every
`Q4_0` `MUL_MAT` the NPU can handle runs there, and everything else stays on the Cortex-A55 cores.
Weights are repacked once into the NPU's int4 format and cached on disk.

Not affiliated with or endorsed by NXP.

## Results

Qwen3, Q4_0, 585-token prompt, 32 generated tokens, 6 threads, i.MX 95 @ 1.8 GHz.

| Model | Prefill CPU | Prefill NPU | | Decode CPU | Decode NPU | |
|---|---|---|---|---|---|---|
| Qwen3-1.7B | 26.48 tok/s | 47.11 | **1.78×** | 4.95 | 3.96 | 0.80× |
| Qwen3-4B | 10.90 | 19.64 | **1.80×** | 2.25 | 1.83 | 0.81× |
| Qwen3-8B | 6.13 | 13.12 | **2.14×** | 1.55 | 1.15 | 0.74× |

The NPU helps prefill (long input), not decode: decode is memory-bandwidth bound and six A55
cores are faster there. Good fits are long-input / short-output jobs such as classification,
triage and summarisation.

## Requirements

- An NXP i.MX 95 SoC with the eIQ Neutron NPU
- NXP's Linux BSP (tested with lf-6.18.20, NPU firmware 3.1.1), which provides the Neutron kernel
  driver (`/dev/neutron0`), `libNeutronDriver.so` and `NeutronDriver.h`
- Enough CMA reserved for the NPU: about 7 GB for an 8B Q4_0 model, plus RAM for the system
- Models in **Q4_0** GGUF. Other quantisations still run, but on the CPU

## Build

Cross-compile with an NXP Yocto SDK or a recipe sysroot that contains the Neutron headers:

```bash
export IMX95_SYSROOT=/path/to/target/sysroot
export IMX95_CROSS=/path/to/bin/aarch64-poky-linux-

cmake -B build-imx95 -DCMAKE_TOOLCHAIN_FILE=imx95-toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_CPU_NEUTRON=ON
cmake --build build-imx95 -j
```

Copy everything in `build-imx95/bin/` (executables and `.so` files) to the board.

## Run

```bash
export LD_LIBRARY_PATH=/path/to/bin

# 1. pack the weights once (can take tens of minutes for an 8B model; later runs load the cache)
./neutron-pack -t 6 Qwen3-8B-Q4_0.gguf

# 2. run on the NPU
NEUTRON_DIRECT=1 ./llama-completion -m Qwen3-8B-Q4_0.gguf -p "Hello" -n 64 -t 6 -no-cnv

# compare against the CPU
NEUTRON_DISABLE=1 ./llama-completion -m Qwen3-8B-Q4_0.gguf -p "Hello" -n 64 -t 6 -no-cnv
```

| Variable | Effect |
|---|---|
| `NEUTRON_DIRECT=1` | Talk to `/dev/neutron0` directly with several memory regions (recommended). Required once the packed model exceeds the vendor library's 3.5 GiB single-buffer limit |
| `NEUTRON_DISABLE=1` | Do not use the NPU |
| `NEUTRON_PREALLOC_REGIONS=N` | Reserve N NPU regions before the model is mapped |
| `NEUTRON_CACHE_DIR=<dir>` | Packed-weight cache (default `~/.cache/ggml-neutron`) |
| `NEUTRON_VERBOSE=1`, `NEUTRON_PROF=1` | Show offloaded tensors / per-stage timing |

## Also in this fork

- `tools/laya` — `laya-score`, runs [Laya](https://github.com/NandhaKishorM/laya) decision models
  (encoder + decision head) end to end. Converted models:
  https://huggingface.co/wigcheng5566/laya-neutron-gguf
- `tools/neutron-pack` — offline weight packer

## Limitations

- Only 2D `Q4_0` `MUL_MAT` is offloaded; MoE (`MUL_MAT_ID`) and other ops stay on the CPU
- Some matrix widths (K) hit tiling bugs in NPU firmware 3.1.1; they are split into verified
  chunks automatically. A different firmware may need the list re-checked
- First-time packing is slow. It uses the packer in NXP's `libNeutronDriver.so`, which is
  aarch64-only, so pack on the board or another aarch64 Linux machine (`tools/neutron-pack`).
  The NPU itself is not needed for packing

## License

MIT, like llama.cpp. Code adapted from NXP's ONNX Runtime Neutron execution provider is also MIT;
see `ggml/src/ggml-cpu/neutron/NOTICE`.
