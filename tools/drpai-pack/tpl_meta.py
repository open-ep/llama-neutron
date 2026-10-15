#!/usr/bin/env python3
"""Write tpl.txt into every template dir: one line the backend reads without parsing Translator files:
K N M w_off w_len apc_off apc_len dp_off dp_len in_sc_off out_sc_off   (all byte offsets relative to the object base)."""
import glob, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hotswap import Template
root = sys.argv[1] if len(sys.argv) > 1 else "tpl"
for d in sorted(glob.glob(f"{root}/t_*")):
    if not os.path.isdir(d) or not os.path.exists(f"{d}/deploy.so"): continue
    K, N, M = map(int, re.match(r".*t_(\d+)x(\d+)_m(\d+)$", d).groups())
    am = {l.split()[0]: (int(l.split()[1], 16), int(l.split()[2], 16)) for l in open(f"{d}/addr_map.txt") if l.strip()}
    t = Template(d)
    dp_len = (am["drp_param"][1] + 63) // 64 * 64
    apc_len = (am["aimac_param_cmd"][1] + 63) // 64 * 64
    regs = ["data_in", "drp_desc", "drp_config", "drp_param", "aimac_desc", "weight", "data_out", "aimac_cmd", "aimac_param_desc", "aimac_param_cmd"]   # DRPAI_INDEX_* order
    line = f"{K} {N} {M} {am['weight'][0]} {am['weight'][1]} {am['aimac_param_cmd'][0]} {apc_len} {am['drp_param'][0]} {dp_len} {t.in_scale_off} {t.out_scale_off} " + " ".join(f"{am[r][0]} {am[r][1]}" for r in regs) + "\n"
    open(f"{d}/tpl.txt", "w").write(line); print(os.path.basename(d), line.strip())
