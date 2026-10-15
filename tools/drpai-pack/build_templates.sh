#!/bin/bash
# Compile every t_<K>x<N>_m<M>.onnx in a template directory with the DRP-AI Translator (one docker container per
# template so the Translator's output dir is never shared).  usage: build_templates.sh <template dir> [docker image]
T=$(cd "$1" && pwd); IMG=${2:-drp-ai_tvm_v2h_27}; W=$(cd "$(dirname "$0")" && pwd)
one() { f=$1; tag=${f%.onnx}; M=${tag##*_m}; KN=${tag#t_}; KN=${KN%_m*}; K=${KN%x*}
  [ -f $T/$tag/deploy.so ] && return 0
  docker run --rm -v $T:/x -v $W:/work $IMG bash -c "
export TRANSLATOR=\"\$(find /opt -name python_api | head -1)/../../\"; export SDK=\"\$(find /opt -name sysroots | head -1)/../\"; export PRODUCT=V2H; export SPARSE_ENABLE=false
cp -r /drp-ai_tvm/tutorials /tmp/t && cd /tmp/t && cp /work/compile_layer.py .
python3 compile_layer.py /x/$f -o /tmp/out -r /tmp/rec -t \$SDK -d \$TRANSLATOR -c \$QUANTIZER -s $M,$K -i x --images /x/calib_${K}_m$M.npy -p '--opts reduce_range False' > /x/$tag.log 2>&1
mkdir -p /x/$tag && cp /tmp/out/sub_0000__CPU_DRP_TVM/deploy.* /x/$tag/ 2>/dev/null
d=\$(dirname \$(find /tmp/t/temp -name weight.bin | head -1)); cp \$d/weight.bin \$d/drp_param.bin \$d/drp_param_info.txt \$d/addr_map.txt /x/$tag/ 2>/dev/null
cp \$(find /tmp/t/temp -name model_quantization.onnx | head -1) /x/$tag/ 2>/dev/null
chown -R $(id -u):$(id -g) /x/$tag /x/$tag.log"
  [ -f $T/$tag/deploy.so ] && echo "OK $tag" || echo "FAIL $tag"; }
export -f one; export T W IMG
ls $T/t_*.onnx | xargs -n1 basename | xargs -P 6 -I{} bash -c 'one {}'
