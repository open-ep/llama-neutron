#!/usr/bin/env python3
"""For every finished template: repack its own quantized tensors with drpai_pack and compare to the Translator weight.bin."""
import onnx, numpy as np, sys, glob, os
from onnx import numpy_helper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drpai_pack import pack_weight
from hotswap import Template
root = sys.argv[1] if len(sys.argv) > 1 else 'tpl'
for d in sorted(glob.glob(os.path.join(root, 't_*'))):
    if not os.path.isdir(d) or not os.path.exists(f'{d}/weight.bin'): continue
    m = onnx.load(f'{d}/model_quantization.onnx'); I = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    wq = [k for k in I if k.endswith('_quantized')][0]; base = wq[:-len('_quantized')]
    ins = [v for k, v in I.items() if k.endswith('_i0_scale')][0]; outs = I['matmul_1_scale']
    ref = open(f'{d}/weight.bin', 'rb').read(); mine = pack_weight(I[wq], I[base + '_scale'], ins, outs)
    t = Template(d)
    ok = ref == mine
    msg = ''
    if not ok:
        r = np.frombuffer(ref, np.uint8); mm = np.frombuffer(mine, np.uint8)
        if len(r) == len(mm): dd = np.nonzero(r != mm)[0]; msg = f'{len(dd)} diff bytes, first {hex(dd[0])}'
        else: msg = f'len {len(r)} vs {len(mm)}'
    print(f'{os.path.basename(d):22s} Wq{I[wq].shape} weight.bin {len(ref):9d}  {"IDENTICAL" if ok else "MISMATCH " + msg}  in_sc@{t.in_scale_off} out_sc@{t.out_scale_off}')
