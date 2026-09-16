#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mac_profiler.py — per-layer real-MAC / duty profiler for the BCCTN zero-multiplier core.

Why this exists
---------------
The duty number in HANDOFF_HW_ZERO_MULTIPLIER_CORE.md (§9.4 "duty ≈ 0.81") mixes
two MAC conventions (complex = ×4 in §8, ×3 in §9.5) and is attention-dominated,
so it is hard to tell whether the literal 4-branch complex cost of every conv /
deconv is fully inside it. This script counts MAC/frame exactly from the
architecture, separating:

  * encoder / decoder complex conv   -> genuinely complex, apply_complex = 4 real
                                        branches  (fr·xr, fi·xi, fr·xi, fi·xr)
  * MHA OUTER complex projections     -> genuinely complex, 4 real branches
    (input_proj / output_proj)
  * MHA INNER (tied ComplexMHA)       -> 2 REAL passes (a=M(x.real), b=M(x.imag)),
    in_proj / QK / Score×V / out_proj    NOT complex -> no ×4, just ×2 passes

It reports every block under both the ×4 (literal apply_complex) and ×3 (Gauss /
Karatsuba 3-mult) conventions, for the DENSE baseline and the low-power DEPLOY
(qkv block-diagonal + decoder grouped pointwise + 50% skip + top-4 Score×V), and
computes the duty against a configurable (lanes, clock).

MAC definition (honest hardware count = every weight×activation product done):
  * forward Conv2d (freq-only, stride S): output-driven
        MAC = Cout * (Cin/groups) * K * Fout
  * ConvTranspose2d (freq-only, stride S): INPUT-driven (a stride-2 transpose does
    NOT evaluate all K taps at every output position)
        MAC = Cout * (Cin/groups) * K * Fin       (depthwise: Cin/groups = 1)
  * Linear / pointwise(1x1): Cout * (Cin/groups) * (spatial)

All counts are "complex-MAC base" (one complex multiply = 1), then multiplied by
the complex factor (3 or 4) for genuinely-complex ops, or by the pass count
(2 tied / 4 untied) for the real inner-MHA ops.

No torch / checkpoint needed: MAC counts depend only on integer dims + group
counts. `--config FILE` pulls those dims + deploy knobs from a Hydra config.yaml;
`--checkpoint FILE` (optional, needs torch) cross-checks the channel dims against
the trained state_dict.
"""

import argparse
import sys


# --------------------------------------------------------------------------- #
# Architecture spec
# --------------------------------------------------------------------------- #
class Arch:
    """Derive per-stage dims from the config the same way model.py does."""

    def __init__(self, kernel_num, kernel_size, fft_len, win_inc,
                 embed_dim, num_heads, attn_win, binaural, tied_mha, sr=16000):
        self.K = int(kernel_size)
        self.fft_len = int(fft_len)
        self.win_inc = int(win_inc)
        self.sr = int(sr)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.attn_win = int(attn_win)
        self.binaural = bool(binaural)
        self.tied_mha = bool(tied_mha)

        # model.py: self.kernel_num = [2] + kernel_num ; per-path channels = //2
        kn = [2] + list(kernel_num)
        self.c = [k // 2 for k in kn]              # per-path channel at each node
        self.n_stages = len(kn) - 1                # encoder/decoder stage count

        # Encoder input freq. STFT(fft_len) then stride-2 in freq n_stages times.
        # model.py base_flat uses fft_len // 2**(len(kn)) for the final freq, so
        # F0 = fft_len // 2  ->  final = fft_len // 2**(n_stages+1).
        self.F0 = self.fft_len // 2
        self.final_freq = self.fft_len // (2 ** (self.n_stages + 1))
        if self.final_freq < 1:
            self.final_freq = 1

        # bottleneck dims (cross-check against configured embed_dim)
        self.flat_per_ear = self.c[-1] * self.final_freq
        self.flat_in = self.flat_per_ear * (2 if self.binaural else 1)

        self.frame_rate = self.sr / self.win_inc   # frames / second

    # ---- encoder ----
    def encoder_stages(self):
        """yield dict(stage, Cin, Cout, Fin, Fout) per-ear, per-frame."""
        out = []
        for i in range(self.n_stages):
            Fin = self.F0 // (2 ** i)
            Fout = Fin // 2
            out.append(dict(stage=i, Cin=self.c[i], Cout=self.c[i + 1],
                            Fin=Fin, Fout=Fout))
        return out

    # ---- decoder ----
    def decoder_stages(self):
        """yield dict(stage, Cin_x, Cout, Cin_skip_full, Fin, Fout)."""
        out = []
        for s in range(self.n_stages):
            idx = self.n_stages - s            # model.py idx runs n..1
            Cin_x = self.c[idx]
            Cout = self.c[idx - 1]
            Cin_skip_full = self.c[idx]        # mirror enc out = same as Cin_x here
            Fin = self.final_freq * (2 ** s)
            Fout = Fin * 2
            out.append(dict(stage=s, idx=idx, Cin_x=Cin_x, Cout=Cout,
                            Cin_skip_full=Cin_skip_full, Fin=Fin, Fout=Fout))
        return out


# --------------------------------------------------------------------------- #
# Group-count helper (mirrors complexLayers._valid_group_count)
# --------------------------------------------------------------------------- #
def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def valid_groups(requested, in_f, out_f):
    requested = max(1, int(requested))
    max_g = _gcd(int(in_f), int(out_f))
    for g in range(min(requested, max_g), 0, -1):
        if max_g % g == 0:
            return g
    return 1


def skip_keep(full, ratio, groups):
    """Group-aware top-ratio keep count (mirrors model._skip_keep_count)."""
    if ratio >= 1.0:
        return full
    g = valid_groups(groups, full, full)        # group sizing for the keep
    gsize = full // g
    keep_per = max(1, min(gsize, int(round(gsize * ratio))))
    return keep_per * g


# --------------------------------------------------------------------------- #
# MAC counting
# --------------------------------------------------------------------------- #
class Knobs:
    """Deploy switches that change MAC (grouping / skip / top-k / pass count)."""
    def __init__(self, name, qkv_groups, out_proj_groups,
                 in_outproj_groups, dec_mode, groups_x, groups_skip,
                 skip_ratio, top_k, passes):
        self.name = name
        self.qkv_groups = qkv_groups
        self.out_proj_groups = out_proj_groups
        self.in_outproj_groups = in_outproj_groups   # outer input_proj/output_proj
        self.dec_mode = dec_mode                      # "off" or "grouped"
        self.groups_x = groups_x
        self.groups_skip = groups_skip
        self.skip_ratio = skip_ratio
        self.top_k = top_k                            # None -> full window
        self.passes = passes                          # 2 tied / 4 untied


def dense_knobs(passes):
    return Knobs("dense-baseline", qkv_groups=1, out_proj_groups=1,
                 in_outproj_groups=1, dec_mode="off", groups_x=1, groups_skip=1,
                 skip_ratio=1.0, top_k=None, passes=passes)


def deploy_knobs(passes):
    return Knobs("low-power-deploy", qkv_groups=4, out_proj_groups=1,
                 in_outproj_groups=1, dec_mode="grouped", groups_x=4,
                 groups_skip=2, skip_ratio=0.5, top_k=4, passes=passes)


def encoder_macs(arch):
    """per-ear complex-MAC base, per block. (encoder has no deploy knobs)."""
    rows = []
    for st in arch.encoder_stages():
        Cin, Cout, Fout = st["Cin"], st["Cout"], st["Fout"]
        dw = Cin * arch.K * Fout                    # depthwise: groups=Cin, Cin/g=1
        pw = Cout * Cin * Fout                       # 1x1 dense
        rows.append((f"enc{st['stage']}.dw", dw))
        rows.append((f"enc{st['stage']}.pw", pw))
    return rows


def decoder_macs(arch, kn):
    """per-ear complex-MAC base, per block, under knobs kn."""
    rows = []
    for st in arch.decoder_stages():
        Cin_x, Cout, Fin = st["Cin_x"], st["Cout"], st["Fin"]
        skip_full = st["Cin_skip_full"]
        dwT = Cout * arch.K * Fin                    # transpose depthwise, INPUT-driven
        if kn.dec_mode == "off":
            # cat([x, skip]) -> single dense pointwise, full skip
            pw = Cout * (Cin_x + skip_full) * Fin
            rows.append((f"dec{st['stage']}.pw(cat)", pw))
        else:
            gx = valid_groups(kn.groups_x, Cin_x, Cout)
            skip_keep_ch = skip_keep(skip_full, kn.skip_ratio, kn.groups_skip)
            gs = valid_groups(kn.groups_skip, skip_keep_ch, Cout)
            pw_x = Cout * (Cin_x // gx) * Fin
            pw_sk = Cout * (skip_keep_ch // gs) * Fin
            rows.append((f"dec{st['stage']}.pw_x(g{gx})", pw_x))
            rows.append((f"dec{st['stage']}.pw_skip(g{gs},keep{skip_keep_ch})", pw_sk))
        rows.append((f"dec{st['stage']}.dwT", dwT))
    return rows


def mha_macs(arch, kn):
    """
    Returns (outer_complex_base, inner_real_total, breakdown_rows).
      outer_complex_base : complex-MAC base of input_proj + output_proj (×cx later)
      inner_real_total   : REAL MAC of the tied passes (passes already folded in)
    """
    E = arch.embed_dim
    Ein = arch.flat_in
    H = arch.num_heads
    D = arch.head_dim
    keys = arch.attn_win + 1                          # causal window incl. self
    topk = keys if kn.top_k is None else min(kn.top_k, keys)

    # ---- outer complex projections (genuinely complex) ----
    g_io = valid_groups(kn.in_outproj_groups, Ein, E)
    in_proj = E * (Ein // g_io)
    out_proj = Ein * (E // g_io)
    outer_base = in_proj + out_proj

    # ---- inner tied MHA, per real pass ----
    g_qkv = valid_groups(kn.qkv_groups, E, E)
    g_o = valid_groups(kn.out_proj_groups, E, E)
    qkv = 3 * E * (E // g_qkv)                        # Q,K,V block-diagonal
    qk = H * keys * D                                # per query frame, windowed
    sv = H * topk * D                               # top-k Score×V
    oproj = E * (E // g_o)
    pass_base = qkv + qk + sv + oproj
    inner_total = kn.passes * pass_base

    rows = [
        ("mha.input_proj  [complex]", in_proj),
        ("mha.output_proj [complex]", out_proj),
        (f"  inner.in_proj QKV (g{g_qkv}) x{kn.passes}", kn.passes * qkv),
        (f"  inner.QK^T (win{keys}) x{kn.passes}", kn.passes * qk),
        (f"  inner.Score×V (top{topk}) x{kn.passes}", kn.passes * sv),
        (f"  inner.out_proj (g{g_o}) x{kn.passes}", kn.passes * oproj),
    ]
    return outer_base, inner_total, rows


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def fmt(n):
    return f"{n:,}"


def profile(arch, kn, cx):
    """Total real-MAC/frame and category subtotals for one knob set + cx."""
    enc = encoder_macs(arch)
    dec = decoder_macs(arch, kn)
    outer_base, inner_total, _ = mha_macs(arch, kn)

    enc_base = sum(v for _, v in enc)
    dec_base = sum(v for _, v in dec)

    enc_real = cx * enc_base * 2                      # ×2 ears
    dec_real = cx * dec_base * 2
    outer_real = cx * outer_base                      # MHA joint (×1)
    inner_real = inner_total                          # already real, passes folded

    total = enc_real + dec_real + outer_real + inner_real
    return dict(enc=enc_real, dec=dec_real, mha_outer=outer_real,
                mha_inner=inner_real, total=total)


def print_block_table(arch, kn):
    print(f"\n  per-block complex-MAC base (1 complex MAC = 1; ×cx / ×passes applied later)")
    print(f"  {'block':40s} {'base':>12s}")
    print("  " + "-" * 54)
    enc = encoder_macs(arch)
    dec = decoder_macs(arch, kn)
    _, _, mrows = mha_macs(arch, kn)
    for name, v in enc:
        print(f"  {name:40s} {fmt(v):>12s}")
    e = sum(v for _, v in enc)
    print(f"  {'ENCODER per-ear base':40s} {fmt(e):>12s}")
    print("  " + "-" * 54)
    for name, v in dec:
        print(f"  {name:40s} {fmt(v):>12s}")
    d = sum(v for _, v in dec)
    print(f"  {'DECODER per-ear base':40s} {fmt(d):>12s}")
    print("  " + "-" * 54)
    for name, v in mrows:
        print(f"  {name:40s} {fmt(v):>12s}")


def duty_table(arch, totals, scenarios):
    print(f"\n  duty @ frame_rate {arch.frame_rate:.0f} fps "
          f"(hop {arch.win_inc} @ {arch.sr} Hz):")
    header = f"  {'profile / cx':28s}" + "".join(
        f"{f'{l}L@{c/1e6:g}MHz':>16s}" for l, c in scenarios)
    print(header)
    print("  " + "-" * (28 + 16 * len(scenarios)))
    for label, total in totals:
        cells = ""
        for lanes, clock in scenarios:
            budget = lanes * clock / arch.frame_rate
            cells += f"{total/budget:>15.3f} "
        print(f"  {label:28s}{cells}")
    # budget reference (real-MAC slots available per frame for each scenario)
    print(f"  {'(slots/frame budget)':28s}" +
          "".join(f"{fmt(int(l*c/arch.frame_rate)):>16s}" for l, c in scenarios))


def run_report(arch, scenarios, cx_list, profiles):
    print("=" * 70)
    print("BCCTN zero-multiplier core — MAC / duty profiler")
    print("=" * 70)
    print(f"kernel_num(per-path) = {arch.c}")
    print(f"stages={arch.n_stages}  F0={arch.F0} -> final_freq={arch.final_freq}")
    print(f"embed_dim={arch.embed_dim} heads={arch.num_heads} head_dim={arch.head_dim} "
          f"attn_win={arch.attn_win} keys={arch.attn_win+1}")
    print(f"flatten/ear={arch.flat_per_ear}  bottleneck_in={arch.flat_in}  "
          f"tied_mha={arch.tied_mha} (passes={2 if arch.tied_mha else 4})")
    print(f"binaural={arch.binaural} -> encoder/decoder counted ×2 ears; MHA joint ×1")

    totals_for_duty = []
    for kn in profiles:
        print("\n" + "=" * 70)
        print(f"PROFILE: {kn.name}   "
              f"[qkv_groups={kn.qkv_groups}, decoder={kn.dec_mode}, "
              f"groups_x={kn.groups_x}, groups_skip={kn.groups_skip}, "
              f"skip_ratio={kn.skip_ratio}, top_k={kn.top_k}, passes={kn.passes}]")
        print("=" * 70)
        print_block_table(arch, kn)
        print(f"\n  category real-MAC / frame:")
        print(f"  {'category':22s} {'×3 (Gauss)':>16s} {'×4 (literal)':>16s}")
        print("  " + "-" * 56)
        p3 = profile(arch, kn, 3)
        p4 = profile(arch, kn, 4)
        for key, lab in [("enc", "encoder (×2 ears)"),
                         ("dec", "decoder (×2 ears)"),
                         ("mha_outer", "MHA outer proj"),
                         ("mha_inner", "MHA inner (real)")]:
            print(f"  {lab:22s} {fmt(p3[key]):>16s} {fmt(p4[key]):>16s}")
        print("  " + "-" * 56)
        print(f"  {'TOTAL':22s} {fmt(p3['total']):>16s} {fmt(p4['total']):>16s}")
        for cx in cx_list:
            tot = profile(arch, kn, cx)["total"]
            totals_for_duty.append((f"{kn.name} cx={cx}", tot))

    duty_table(arch, totals_for_duty, scenarios)
    print("\nnote: MHA inner is REAL (tied = 2 passes), so it carries the ×passes")
    print("      factor but NOT ×cx. Only conv/deconv + outer MHA proj are complex.")
    print("      QK / Score×V scale (2^-k) ride the L-Mul exponent path (not MAC).")


# --------------------------------------------------------------------------- #
# Optional config / checkpoint ingestion
# --------------------------------------------------------------------------- #
def load_config(path):
    try:
        import yaml
    except ImportError:
        print(f"[warn] pyyaml not available; ignoring --config {path}", file=sys.stderr)
        return {}
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("model", cfg) if isinstance(cfg, dict) else {}


def arch_from_config(m, sr):
    return Arch(
        kernel_num=m.get("kernel_num", [16, 32, 64, 128, 256, 256]),
        kernel_size=m.get("kernel_size", 5),
        fft_len=m.get("fft_len", 128),
        win_inc=m.get("win_inc", 80),
        embed_dim=m.get("embed_dim", 256),
        num_heads=m.get("num_heads", 16),
        attn_win=m.get("attn_win", 31),
        binaural=m.get("binaural", True),
        tied_mha=m.get("tied_mha", True),
        sr=sr,
    )


def crosscheck_checkpoint(path, arch):
    try:
        import torch
    except ImportError:
        print(f"[warn] torch not available; skipping --checkpoint cross-check",
              file=sys.stderr)
        return
    print("\n" + "=" * 70)
    print(f"CHECKPOINT cross-check: {path}")
    print("=" * 70)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    # strip lightning prefixes
    clean = {}
    for k, v in sd.items():
        kk = k
        for pre in ("model.", "module."):
            if kk.startswith(pre):
                kk = kk[len(pre):]
        clean[kk] = v
    # report a few key conv / proj shapes vs analytic
    interesting = [k for k in clean
                   if any(t in k for t in ("pointwise_r", "depthwise_r",
                                           "in_proj_weight", "out_proj.weight",
                                           "input_proj.weight", "output_proj.weight"))
                   and k.endswith("weight")]
    print(f"found {len(interesting)} conv/proj weight tensors")
    grouped = [k for k in clean if "pointwise_r_x" in k or "skip_indices" in k]
    print(f"decoder split/grouped markers present: {len(grouped)} "
          f"({'GROUPED checkpoint' if grouped else 'dense (off) checkpoint'})")
    for k in sorted(interesting)[:24]:
        print(f"  {k:55s} {tuple(clean[k].shape)}")
    print("\nnote: qkv_groups / out_proj_groups are a RUNTIME weight mask "
          "(in_proj_weight stays dense [3E,E]); they are NOT recoverable from the\n"
          "      checkpoint — take them from the config / CLI.")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="Hydra config.yaml to read dims + knobs from")
    ap.add_argument("--checkpoint", help="state_dict to cross-check channel dims")
    ap.add_argument("--sr", type=int, default=16000, help="sample rate (frame rate = sr/win_inc)")
    ap.add_argument("--lanes", type=int, action="append", default=[],
                    help="extra duty scenario lane count (pair with --clock)")
    ap.add_argument("--clock", type=float, action="append", default=[],
                    help="extra duty scenario clock Hz (pair with --lanes)")
    ap.add_argument("--cx", default="3,4", help="complex conventions to tabulate for duty")
    ap.add_argument("--profile", choices=["both", "dense", "deploy"], default="both")
    args = ap.parse_args()

    if args.config:
        arch = arch_from_config(load_config(args.config), args.sr)
    else:
        arch = arch_from_config({}, args.sr)        # built-in 19_algo_final defaults

    # sanity: configured embed should equal derived bottleneck input
    if arch.flat_in != arch.embed_dim:
        print(f"[note] bottleneck_in({arch.flat_in}) != embed_dim({arch.embed_dim}); "
              f"input_proj maps {arch.flat_in}->{arch.embed_dim}", file=sys.stderr)

    scenarios = [(128, 2.5e6), (64, 5.0e6)]
    for l, c in zip(args.lanes, args.clock):
        scenarios.append((l, c))

    cx_list = [int(x) for x in args.cx.split(",") if x.strip()]
    passes = 2 if arch.tied_mha else 4
    profs = {"dense": dense_knobs(passes), "deploy": deploy_knobs(passes)}
    if args.profile == "both":
        chosen = [profs["dense"], profs["deploy"]]
    else:
        chosen = [profs[args.profile]]

    run_report(arch, scenarios, cx_list, chosen)

    if args.checkpoint:
        crosscheck_checkpoint(args.checkpoint, arch)


if __name__ == "__main__":
    main()
