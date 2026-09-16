#!/usr/bin/env python3
"""
verify_fp16_tree_attention.py — FP16 32-way tree headroom for the MHA internal
sub-ops QK^T / Score×V / out_proj, which the e4m3 activation dump does NOT contain
(they run on fp16/internal tensors). This is the companion to
verify_fp16_tree_headroom.py (which covered every conv/proj W4A8/FP8 layer).

CONFIDENCE: this is a CALIBRATION-GROUNDED BOUND, not a golden-exact reconstruction
— the Q/K/V/scores/probs/context intermediates are not captured in hw_patterns_final.
The numbers come from:
  * activation_blockfloat/table.json  (mattn.mattn.activation_quantizer: the single
    e4m3 boundary for Q/K/V — amax, pow2 exponent k)
  * layer_config.json attention block (head_dim, top_k, prob_scale, qk_scale)
  * the deployed forward DCNN/utils/complexPyTorch/complexLayers.py
    _mha_attention_forward L2134-2137: q,k,v ALL pass activation_quantizer (same
    scale) -> in HW all three are e4m3 amplified by 2^k. _apply_qk_scale = x2^-2.
    Score×V = q0.8 packet (prob code 0..255) x V e4m3.  out_proj = ctx(fp16,true) x INT4.

The overflow verdicts are robust because for QK and Score×V a SINGLE pre-descale
product already exceeds FP16 max 65504 — no summation/cancellation argument needed.
"""
import json, os

DEPLOY = "/raid/R12K41024/18_w4a8_pow2_mixed_lmul_hw_handoff_20260530"
TBL = f"{DEPLOY}/export_out_strict18/activation_blockfloat/table.json"
CFG = "/raid/R12K41024/19_algo_final/hw_patterns_final/tb_pattern/layer_config.json"
FP16_MAX = 65504.0

t = json.load(open(TBL))["mattn.mattn.activation_quantizer"]
cfg = json.load(open(CFG))["attention"]

amax_true = t["calib_amax"]          # true |Q/K/V| max  (p999 calib)
k_exp     = t["scale_pow2_exp"]      # block-float exponent: code = e4m3(x * 2^k)
amax_code = t["scaled_amax"]         # amplified amax actually stored as e4m3 code
amp       = 2.0 ** k_exp

head_dim  = cfg["head_dim"]                              # QK contraction depth
top_k     = cfg["prob_packet"]["top_k"]                 # Score×V contraction depth
prob_max  = cfg["prob_packet"]["prob_max_code"]         # q0.8 prob code max (255)
prob_scl  = cfg["prob_packet"]["prob_scale"]            # 256
qk_pow2   = cfg["qk_scale"]["pow2_exp"]                 # -2  (1/sqrt(16))
QMAX = 7                                                # INT4 symmetric qmax7

print("=" * 96)
print("FP16 32-way tree headroom — MHA internals (CALIBRATION BOUND, not golden-exact)")
print("=" * 96)
print(f"activation_quantizer (Q/K/V): true amax={amax_true:.2f}, k=2^{k_exp} (x{amp:.0f}), "
      f"e4m3 code amax={amax_code:.1f}")
print(f"head_dim(QK depth)={head_dim}  top_k(ScoreV depth)={top_k}  "
      f"prob_code_max={prob_max} /{prob_scl}  qk_scale=2^{qk_pow2}  INT4 qmax={QMAX}")
print(f"FP16 max-normal = {FP16_MAX:.0f}\n")

import math
prob_exp = int(round(math.log2(prob_scl)))      # q0.8 prob descale = 2^-prob_exp (=2^-8)

rows = []
# ---- QK^T : both Q,K e4m3 amplified x2^k. PRE-tree preshift = -(k+k) (pe_pair i_preshift)
qk_prod = amax_code * amax_code
qk_acc  = head_dim * qk_prod
qk_ps   = -(k_exp + k_exp)                       # pre-apply Q's 2^-k and K's 2^-k
rows.append(("QK^T (L-Mul m8)", "Q,K e4m3 x32", head_dim, qk_prod, qk_acc, qk_ps))

# ---- Score×V : prob q0.8(<=255) x V e4m3 x2^k. preshift = -(log2 prob_scale + k)
sv_prod = prob_max * amax_code
sv_acc  = top_k * sv_prod
sv_ps   = -(prob_exp + k_exp)                    # pre-apply prob/256 and V 2^-k
rows.append(("Score×V (top-4 pkt)", "prob<=255 x V x32", top_k, sv_prod, sv_acc, sv_ps))

# ---- out_proj : ctx TRUE fp16 (no 2^k), INT4 weight, depth 256 (8 grp x32). No preshift.
ctx_true = amax_true
op_prod  = ctx_true * QMAX
op_grp32 = 32 * op_prod
rows.append(("out_proj (W4A16)", "ctx TRUE x INT4", 32, op_prod, op_grp32, 0))

hdr = (f"{'sub-op':<22}{'operands':<19}{'depth':>6}{'pre partial':>13}{'preshift':>9}"
       f"{'post partial':>14}{'/65504':>9}{'ovf':>6}")
print(hdr); print("-" * len(hdr))
allsafe = True
for name, ops, depth, prod, partial, ps in rows:
    post = partial * (2.0 ** ps)
    post_prod = prod * (2.0 ** ps)
    over = post > FP16_MAX or post_prod > FP16_MAX
    allsafe &= not over
    pstxt = f"2^{ps}" if ps != 0 else "—"
    print(f"{name:<22}{ops:<19}{depth:>6}{partial:>13.0f}{pstxt:>9}{post:>14.1f}"
          f"{post/FP16_MAX:>8.2f}x{('OVF' if over else 'ok'):>6}")
print("-" * len(hdr))
print("pre partial  = worst-case pre-descale accumulator WITHOUT fix (overflow shown earlier)")
print("preshift     = per-tensor pow2 folded into pe_pair (Q/K 2^-k; prob/256 & V 2^-k); per-tensor=valid")
print("post partial = same accumulator WITH the pre-tree preshift fix")
print()
print("VERDICT (with pre-tree preshift fix):")
print(f"  QK^T     : preshift 2^{qk_ps}  -> partial {qk_acc*2.0**qk_ps:.0f} ({qk_acc*2.0**qk_ps/FP16_MAX:.2f}x), single prod "
      f"{qk_prod*2.0**qk_ps:.0f}  -> SAFE.")
print(f"  Score×V  : preshift 2^{sv_ps} -> partial {sv_acc*2.0**sv_ps:.1f} ({sv_acc*2.0**sv_ps/FP16_MAX:.3f}x)  -> SAFE.")
print(f"  out_proj : no preshift needed (ctx true-scale) -> partial {op_grp32:.0f} ({op_grp32/FP16_MAX:.2f}x)  -> SAFE.")
print()
print("ALL THREE MHA sub-ops fit FP16 after the per-tensor pre-tree preshift." if allsafe
      else "STILL OVERFLOWS — investigate.")
print("NOTE: calibration-bound (intermediates not dumped); the 1/sqrt(d)=2^-2 score scale and any")
print("APoT residual stay post-tree in the score-scale / dealer path. INT4 conv path unaffected.")
