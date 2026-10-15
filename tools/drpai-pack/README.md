# drpai-pack: model packages and shape templates for the DRP-AI backend

The backend (`ggml/src/ggml-drpai`) runs every linear layer of a model as a DRP object.  A DRP object is normally
produced per weight matrix by the Renesas DRP-AI Translator; here the Translator is run **once per matrix shape**
to make a *template*, and the weights of any model are packed by `drpai_pack.py` and swapped into the template at
load time.  Changing models therefore needs no Translator run.

## Shape templates (once per shape, needs the Renesas AI SDK docker image with DRP-AI TVM + Translator)

    python3 make_templates.py tpl 1024x2048 1024x1024 2048x1024 1024x3072 3072x1024 1024x32768 1024x20864   # Qwen3-0.6B
    ./build_templates.sh tpl                 # ~5 min per shape, SPARSE_ENABLE=false, compile_layer.py in this dir
    python3 verify_tpl.py tpl                # every template must print IDENTICAL (packer == Translator)
    python3 tpl_meta.py tpl                  # writes tpl.txt (offsets the backend needs) into each template
    # ship tpl/t_*/{deploy.so,deploy.json,deploy.params,tpl.txt} to the board, e.g. ~/drpai-tpl

Shapes are (K, N) of the GGUF weight (`ne[0] x ne[1]`); lm_head is split into chunks of 32768 rows plus the
remainder (Qwen3 vocab 151936 -> 4 x 32768 + 20864).

## Model package (per model, seconds, no Translator)

    python3 hs_build.py Qwen/Qwen3-0.6B pkg/qwen3-0.6b calibration.txt     # HF model id or local path, ~1 min
    # ship pkg/qwen3-0.6b to the board, e.g. ~/models/qwen3-0.6b

The recipe per linear: per-channel 8-bit weights, static per-tensor int8 activations calibrated on 1024 tokens of
the calibration text, 16 outlier input columns / 4 output rows computed on the CPU in fp32, partial output-channel
equalization (b = 0.5), and a two-row (hi/lo, g = 8) input so the DRP sees ~11-bit input.  lm_head chunks get
16 CPU rows and 2x output headroom (LMH_KO / LMH_YM in hs_build.py).

## Run

    GGML_DRPAI_MODEL=~/models/qwen3-0.6b GGML_DRPAI_TEMPLATES=~/drpai-tpl \
      llama-cli -m qwen3-0.6b-q8_0.gguf -c 2048 -st -p "Hello /no_think"

The GGUF supplies tokenizer, norms and embeddings; any weight that has a package entry runs on the DRP-AI.
