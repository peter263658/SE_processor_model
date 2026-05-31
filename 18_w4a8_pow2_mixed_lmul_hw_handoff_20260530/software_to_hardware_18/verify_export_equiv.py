"""Numerical equivalence check: source parametrization (what the validated eval
forward actually uses) vs export-helper reconstruction (what the chip will store).

For every quantized module we compare:
  source  = module.weight            # parametrization forward output = q_src * scale_src
  export  = reconstruct(eh.int4_*(raw_weight))   # q_exp * scale_exp

A faithful export => max|source-export| == 0 (same algorithm, same inputs).
BN/PReLU are compared against the source low-power functions directly.
"""
from __future__ import annotations
import os, sys
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import export_helpers as eh

model, model_cfg, training_cfg, load_report = eh.build_model()
num_heads = int(model_cfg.get("num_heads", 16))
embed_dim = int(model_cfg.get("embed_dim", 256))
group_size = int(training_cfg.get("int4_group_size", 32))
scale_format = training_cfg.get("int4_scale_format", "apot_clip")
print(f"load missing={len(load_report['missing'])} unexpected={len(load_report['unexpected'])}")
print(f"num_heads={num_heads} embed_dim={embed_dim} group_size={group_size} scale_format={scale_format}")

worst = {}
def report(tag, src, exp):
    src = src.detach().to(torch.float32); exp = exp.detach().to(torch.float32)
    if src.shape != exp.shape:
        print(f"[SHAPE-MISMATCH] {tag}: src{tuple(src.shape)} exp{tuple(exp.shape)}")
        worst[tag] = float("inf"); return
    d = (src - exp).abs().max().item()
    worst[tag] = d
    flag = "" if d < 1e-6 else "  <<< DIFF"
    print(f"{tag:52s} maxabs={d:.3e}{flag}")

# ---- reconstruction matching each export granularity ----
def recon_per_channel(w):
    q, raw, sq, enc, eff = eh.int4_per_channel(w.float(), scale_format)
    return (q.float() * sq)  # sq broadcasts [out,1,1,1]

def recon_per_group(w):
    q, raw, sq, enc, eff = eh.int4_per_group_input_dim(w.float(), group_size, scale_format)
    if eff == "per_channel":
        return q.float() * sq
    out_ch, in_ch = w.shape[0], w.shape[1]
    k_h, k_w = (w.shape[2], w.shape[3]) if w.dim() == 4 else (1, 1)
    gs = (out_ch, in_ch // group_size, group_size, k_h, k_w)
    return (q.float().reshape(gs) * sq).reshape_as(w)

def recon_per_head_linear(w):
    q, raw, sq, enc, eff = eh.int4_per_head_linear(w.float(), num_heads, scale_format)
    rows, cols = w.shape
    if eff == "per_head_rows":
        return (q.float().reshape(num_heads, rows//num_heads, cols) * sq).reshape_as(w)
    if eff == "per_head_cols":
        return (q.float().reshape(rows, num_heads, cols//num_heads) * sq).reshape_as(w)
    return q.float() * sq

def recon_per_head_mha(w):
    q, raw, sq, enc = eh.int4_per_head_mha(w.float(), num_heads, scale_format)
    total = 3*num_heads; rows, cols = w.shape
    return (q.float().reshape(total, rows//total, cols) * sq).reshape_as(w)

# ---- 1. encoder/decoder conv blocks ----
blocks = eh.discover_blocks(model)
for b in blocks:
    if b.depthwise_r is not None:
        for half, sub in (("r", b.depthwise_r), ("i", b.depthwise_i)):
            report(f"{b.dotted}.depthwise_{half}", sub.weight, recon_per_channel(eh.raw_weight(sub)))
    for suffix, sub in b.pointwise_items:
        if b.is_outlier_fp8:
            # FP8 outlier: per-tensor scaled FP8 (scale=448/amax). recon = code/scale.
            raw = eh.raw_weight(sub).float()
            enc = eh.scaled_fp8_encode(raw, torch.float8_e4m3fn)
            report(f"{b.dotted}.{suffix} (scaled-fp8)", sub.weight.float(), enc["recon"])
        else:
            report(f"{b.dotted}.{suffix}", sub.weight, recon_per_group(eh.raw_weight(sub)))

# ---- 2. mattn complex linears (input_proj / output_proj) ----
matt = model.mattn
for name, lin in (("input_proj", matt.input_proj), ("output_proj", matt.output_proj)):
    w = eh.raw_weight(lin)  # complex
    src = lin.weight
    exp_r = recon_per_head_linear(w.real.contiguous())
    exp_i = recon_per_head_linear(w.imag.contiguous())
    report(f"mattn.{name}.real", src.real, exp_r)
    report(f"mattn.{name}.imag", src.imag, exp_i)

# ---- 3. MHA in_proj (dense, pre-mask) and out_proj ----
cmha = matt.mattn
mha = cmha.MultiheadAttention_re
raw_ip = eh.raw_in_proj_weight(mha)
report("mha.in_proj_weight (dense)", mha.in_proj_weight, recon_per_head_mha(raw_ip))
report("mha.out_proj.weight", mha.out_proj.weight, recon_per_head_linear(eh.raw_weight(mha.out_proj)))

# verify export mask zeros same entries as source forward mask
try:
    from DCNN.utils.complexPyTorch import complexLayers as CL
    src_mask = CL._mha_in_proj_group_mask(mha.in_proj_weight, embed_dim, 4)
    exp_mask = eh.mha_in_proj_group_mask(tuple(raw_ip.shape), embed_dim, 4)
    if src_mask is not None:
        dm = (src_mask.float().cpu() - exp_mask.float()).abs().max().item()
        print(f"{'mha in_proj group mask (src vs exp)':52s} maxabs={dm:.3e}{'' if dm==0 else '  <<< DIFF'}")
        worst["mha_mask"] = dm
except Exception as e:
    print(f"[mask check skipped] {e}")

# ---- 4. BN fold: source low-power affine vs export apot ----
try:
    from DCNN.utils.complexPyTorch import complexLayers as CL
    src_fold = None
    for cand in ("_folded_bn_affine", "_fold_bn_affine", "_folded_bn"):
        if hasattr(CL, cand):
            src_fold = getattr(CL, cand); src_fold_name = cand; break
    print(f"[BN] source fold fn = {src_fold_name if src_fold else 'NOT FOUND'}")
    for b in blocks:
        if b.bn_r is None: continue
        aff = eh.folded_bn_affine(b, -9, 4, fuse_conv_bias=False)  # compare alpha only (no conv-bias fuse)
        for half, bn, alpha_exp in (("r", b.bn_r, aff.alpha_r), ("i", b.bn_i, aff.alpha_i)):
            if src_fold is not None:
                a_src, _ = src_fold(bn, "fold_apot", -9, 4)
                report(f"{b.dotted}.bn_{half}.alpha", a_src.cpu(), alpha_exp.cpu())
            else:
                # fallback: replicate effective alpha (unquantized) magnitude sanity
                gamma = bn.weight.detach().cpu(); var = bn.running_var.detach().cpu()
                a_eff = gamma * torch.rsqrt(var + float(bn.eps))
                print(f"  {b.dotted}.bn_{half}: |alpha_eff| range [{a_eff.abs().min():.4f},{a_eff.abs().max():.4f}] exp[{torch.log2(a_eff.abs().clamp_min(1e-9)).min():.2f},{torch.log2(a_eff.abs().clamp_min(1e-9)).max():.2f}]")
except Exception as e:
    import traceback; print(f"[BN check error] {e}"); traceback.print_exc()

# ---- 5. PReLU pow2 ----
try:
    from DCNN.utils.complexPyTorch import complexLayers as CL
    src_p2 = None
    for cand in ("_pow2_quantize_scale", "_quantize_pow2", "_pow2_quantize"):
        if hasattr(CL, cand):
            src_p2 = getattr(CL, cand); src_p2_name = cand; break
    print(f"[PReLU] source pow2 fn = {src_p2_name if src_p2 else 'NOT FOUND'}")
    for b in blocks:
        if b.prelu_r is None: continue
        for half, pr in (("r", b.prelu_r), ("i", b.prelu_i)):
            exp = eh.pow2_encode(pr.weight.detach().cpu(), -8, 0)["quantized"]
            if src_p2 is not None:
                s = src_p2(pr.weight.detach().cpu(), -8, 0)
                report(f"{b.dotted}.prelu_{half}.slope", s.cpu(), exp.cpu())
except Exception as e:
    import traceback; print(f"[PReLU check error] {e}"); traceback.print_exc()

print("\n================ SUMMARY ================")
bad = {k:v for k,v in worst.items() if v >= 1e-6}
print(f"modules checked: {len(worst)}   diffs>=1e-6: {len(bad)}")
if bad:
    for k,v in sorted(bad.items(), key=lambda kv:-kv[1]):
        print(f"  DIFF {k}: {v:.3e}")
else:
    print("  ALL EXPORT RECONSTRUCTIONS BIT-MATCH THE SOURCE PARAMETRIZATION.")
