#!/usr/bin/env python3
"""
BCCTN MAC Utilization & Pipeline Bottleneck Profiler
=====================================================
Per-layer MAC utilization for the 256-MAC ASIC target.

Columns:
  layer_name | type | params | MACs | compute_cycles | weight_dma_cycles
           | act_dma_cycles | utilization(%) | bottleneck | util_w4a8(%)

Usage:
    cd 01_utilization_analysis
    python utilization_profiler.py
"""

import sys, os, math, csv, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict

from DCNN.models.binaural_attention_model import BinauralAttentionDCNN

# ─── Hardware Constants ──────────────────────────────────────────────────────
NUM_MACS         = 256     # PE array size
BUS_WIDTH_BYTES  = 8       # 64-bit bus
WEIGHT_PREC_FP8  = 1       # fp8 = 1 byte/weight
WEIGHT_PREC_W4   = 0.5     # W4 = 0.5 byte/weight  (4-bit)
ACT_PREC         = 1       # fp8 = 1 byte/activation
CLOCK_FREQ_MHZ   = 200
WBUF_SIZE_KB     = 64
ABUF_SIZE_KB     = 32

# ─── Model Config ────────────────────────────────────────────────────────────
MODEL_CFG = dict(
    win_len=128, win_inc=80, fft_len=128, win_type='hann',
    kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256],
    masking_mode='E', use_clstm=False, use_depthwise=True,
    rnn_layers=2, rnn_units=128,
    embed_dim=512, num_heads=32, attn_win=31,
    binaural=True,
)

CKPT_PATH = os.path.join(os.path.dirname(__file__),
                         'DCNN', 'Checkpoints', 'Trained_model.ckpt')

# 1-second stereo audio → T_frames
SR         = 16000
DURATION   = 1.0          # seconds
N_SAMPLES  = int(SR * DURATION)
# T = ceil((N_SAMPLES - win_len) / win_inc) + 1
T_FRAMES   = math.ceil((N_SAMPLES - MODEL_CFG['win_len']) / MODEL_CFG['win_inc']) + 1
F_BINS     = MODEL_CFG['fft_len'] // 2 + 1   # 65 for fft_len=128

KERNEL_NUM_FULL = [2] + MODEL_CFG['kernel_num']   # [2,16,32,64,128,256,256]
KERNEL_SIZE     = MODEL_CFG['kernel_size']         # 5
EMBED_DIM       = MODEL_CFG['embed_dim']           # 512
NUM_HEADS       = MODEL_CFG['num_heads']           # 32
ATTN_WIN        = MODEL_CFG['attn_win']            # 31


# ─── Utility ─────────────────────────────────────────────────────────────────

def _cycles_from_bytes(nbytes, prec=WEIGHT_PREC_FP8):
    """DMA cycles = bytes / BUS_WIDTH_BYTES (1 transfer per cycle)."""
    return math.ceil(nbytes * prec / BUS_WIDTH_BYTES)


def _utilization(compute_cyc, weight_dma_cyc, act_dma_cyc):
    """MAC utilization = compute / (compute + bottleneck_dma)."""
    bottleneck_dma = max(weight_dma_cyc, act_dma_cyc)
    total = compute_cyc + bottleneck_dma
    if total == 0:
        return 0.0, 'n/a'
    util = compute_cyc / total * 100.0
    bottleneck = 'compute' if compute_cyc >= bottleneck_dma else 'dma'
    return util, bottleneck


# ─── Layer analysis helpers ──────────────────────────────────────────────────

def analyze_depthwise_conv(layer_name, C_in, C_out, kH, kW, H_in, W_in, stride,
                            is_transpose=False):
    """
    Depthwise conv: groups = C_in  →  C_out = C_in after depthwise.
    Only kH*kW MACs per output element, no cross-channel reduction.
    For complex: ×4 real passes.
    """
    H_out = H_in // stride if not is_transpose else H_in * stride
    W_out = W_in           # stride only in freq axis

    # real MACs per complex depthwise conv (4 real passes)
    macs_per_ch = kH * kW * H_out * W_out    # per channel, per real pass
    total_macs  = 4 * C_in * macs_per_ch     # ×4 for complex, ×C_in channels

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    # Weight bytes: depthwise has C_in * kH * kW weights (real part only ×2 for complex)
    weight_count = 2 * C_in * kH * kW        # ×2: re + im kernels
    weight_dma   = _cycles_from_bytes(weight_count)

    # Activation bytes (fp8): input + output
    act_in_bytes  = C_in  * H_in  * W_in
    act_out_bytes = C_in  * H_out * W_out    # C_out=C_in for depthwise
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    # Effective utilization note: only kH*kW MACs per channel tile
    # With NUM_MACS=256 and kH=5, kW=1 → ~5/256 = 2% spatial utilization
    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    params = 2 * C_in * kH * kW   # re+im

    return dict(
        layer_name=layer_name,
        type='depthwise_conv',
        params=params,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'kH={kH},kW={kW} → effective_PE_util={kH*kW/NUM_MACS*100:.1f}%',
    )


def analyze_pointwise_conv(layer_name, C_in, C_out, H, W):
    """
    Pointwise (1×1) conv: full Cin×Cout reduction → maps well to 256 MACs.
    Complex: 4 real passes.
    """
    # MACs: B=1, each output element needs C_in MACs
    total_macs = 4 * C_out * H * W * C_in   # ×4 complex

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    weight_count = 2 * C_in * C_out          # re+im kernels
    weight_dma   = _cycles_from_bytes(weight_count)

    act_in_bytes  = C_in  * H * W
    act_out_bytes = C_out * H * W
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    params = 2 * C_in * C_out

    return dict(
        layer_name=layer_name,
        type='pointwise_conv',
        params=params,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'Cin={C_in}→Cout={C_out}',
    )


def analyze_linear_proj(layer_name, in_feat, out_feat, T, is_complex=True):
    """
    Linear projection: T time-steps, (in_feat × out_feat) matmul.
    Complex: 4 real passes.
    """
    scale   = 4 if is_complex else 1
    total_macs = scale * T * in_feat * out_feat

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    weight_count = (2 if is_complex else 1) * in_feat * out_feat
    weight_dma   = _cycles_from_bytes(weight_count)

    act_in_bytes  = (2 if is_complex else 1) * T * in_feat
    act_out_bytes = (2 if is_complex else 1) * T * out_feat
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    params = (2 if is_complex else 1) * in_feat * out_feat

    return dict(
        layer_name=layer_name,
        type='linear',
        params=params,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'T={T},{in_feat}→{out_feat}',
    )


def analyze_mha_qkv_proj(layer_name, embed_dim, T, num_heads):
    """
    MHA in-projection: combined QKV = 3 × (embed_dim × embed_dim) matmuls.
    PyTorch packs them as in_proj: (3*E, E). Complex: 4 real passes.
    """
    total_macs = 4 * T * 3 * embed_dim * embed_dim

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    weight_count = 2 * 3 * embed_dim * embed_dim    # re+im, Wq+Wk+Wv
    weight_dma   = _cycles_from_bytes(weight_count)

    act_in_bytes  = 2 * T * embed_dim
    act_out_bytes = 2 * T * 3 * embed_dim
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    params = 2 * 3 * embed_dim * embed_dim

    return dict(
        layer_name=layer_name,
        type='mha_qkv_proj',
        params=params,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'3×(E²), E={embed_dim}',
    )


def analyze_mha_out_proj(layer_name, embed_dim, T):
    """MHA output projection: (embed_dim × embed_dim). Complex: 4 real passes."""
    total_macs = 4 * T * embed_dim * embed_dim

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    weight_count = 2 * embed_dim * embed_dim
    weight_dma   = _cycles_from_bytes(weight_count)

    act_in_bytes  = 2 * T * embed_dim
    act_out_bytes = 2 * T * embed_dim
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    params = 2 * embed_dim * embed_dim

    return dict(
        layer_name=layer_name,
        type='mha_out_proj',
        params=params,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'E²={embed_dim}×{embed_dim}',
    )


def analyze_qkt_matmul(layer_name, embed_dim, T, num_heads, win_size):
    """
    QK^T: For each head, Q[T, d_head] × K[T, d_head]^T → [T, T] (windowed → [T, W]).
    With sliding window W=31, each row only attends to W keys.
    Complex: 4 real passes (re/re, re/im, im/re, im/im).
    """
    d_head     = embed_dim // num_heads
    # windowed: T * win_size * d_head per head, ×num_heads, ×2 MHA instances (re+im)
    # ×4 for complex arithmetic
    total_macs = 4 * 2 * num_heads * T * win_size * d_head

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    # No weight DMA (Q,K are activations)
    weight_dma   = 0
    act_in_bytes  = 2 * 2 * T * embed_dim   # Q+K, re+im, ×2 MHA
    act_out_bytes = 2 * num_heads * T * win_size  # attention scores re+im, ×2 MHA
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    return dict(
        layer_name=layer_name,
        type='qkt_matmul',
        params=0,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'T={T},W={win_size},d_head={d_head}',
    )


def analyze_score_v_matmul(layer_name, embed_dim, T, num_heads, win_size):
    """
    Score × V: attention_weights[T, W] × V[T, d_head] → [T, d_head].
    Complex: 4 real passes, ×2 MHA instances.
    """
    d_head     = embed_dim // num_heads
    total_macs = 4 * 2 * num_heads * T * win_size * d_head

    compute_cyc = math.ceil(total_macs / NUM_MACS)

    weight_dma   = 0   # V is activation
    act_in_bytes  = 2 * 2 * (num_heads * T * win_size + T * embed_dim)  # scores + V
    act_out_bytes = 2 * 2 * T * embed_dim
    act_dma       = _cycles_from_bytes(act_in_bytes + act_out_bytes)

    util, bottleneck = _utilization(compute_cyc, weight_dma, act_dma)

    return dict(
        layer_name=layer_name,
        type='score_v_matmul',
        params=0,
        MACs=total_macs,
        compute_cycles=compute_cyc,
        weight_dma_cycles=weight_dma,
        act_dma_cycles=act_dma,
        utilization=round(util, 2),
        bottleneck=bottleneck,
        note=f'T={T},W={win_size},d_head={d_head}',
    )


def analyze_softmax(layer_name, T, num_heads, win_size):
    """
    Softmax is non-MAC. Estimate LUT cycles = exp + normalization.
    ~8 LUT ops per element: exp(x), accumulate, divide.
    ×2 for complex (re+im MHA). ×2 for binaural (2 passes).
    Treat as 0 MACs but non-zero LUT cycles.
    """
    lut_ops      = 4 * 2 * num_heads * T * win_size * 8   # 8 LUT ops per element
    lut_cyc_est  = math.ceil(lut_ops / NUM_MACS)          # if LUT unit parallelism = MACs

    return dict(
        layer_name=layer_name,
        type='softmax_lut',
        params=0,
        MACs=0,
        compute_cycles=0,
        weight_dma_cycles=0,
        act_dma_cycles=0,
        utilization=0.0,
        bottleneck='lut',
        note=f'~{lut_cyc_est} estimated LUT cycles (non-MAC)',
    )


# ─── W4A8 utilization helper ─────────────────────────────────────────────────

def w4_utilization(row):
    """
    Re-compute utilization with W4 (0.5 byte/weight) instead of FP8.
    Only weight_dma changes; act_dma and compute_cycles stay the same.
    """
    if row['type'] in ('qkt_matmul', 'score_v_matmul', 'softmax_lut'):
        return row['utilization']   # no weight DMA for these
    # Re-derive weight bytes from params
    # params = (re+im)*weights, so weight elements = params
    # FP8 weight_dma = ceil(params * 1 / 8)
    # W4  weight_dma = ceil(params * 0.5 / 8)
    w4_dma = math.ceil(row['params'] * WEIGHT_PREC_W4 / BUS_WIDTH_BYTES)
    util, _ = _utilization(row['compute_cycles'], w4_dma, row['act_dma_cycles'])
    return round(util, 2)


# ─── Main analysis ───────────────────────────────────────────────────────────

def build_layer_table():
    """
    Enumerate all layers analytically based on model architecture.
    BinauralAttentionDCNN: two shared encoders/decoders (L+R) + 1 joint attention.
    """
    rows = []
    kn = KERNEL_NUM_FULL   # [2,16,32,64,128,256,256]

    # ── Frequency axis after each encoder stage ──────────────────────────────
    # Input STFT: F_BINS=65, T=T_FRAMES
    # Each encoder stride=(2,1) halves frequency axis
    freq_sizes = [F_BINS]
    for _ in range(len(kn) - 1):
        freq_sizes.append(math.ceil(freq_sizes[-1] / 2))
    # freq_sizes[i] = freq after stage i  (0=input, 1=after enc[0], ...)

    # ── ENCODER (×2 binaural, but L and R share same weights in binaural model) ──
    # Architecture: ComplexDepthwiseSeparableConv2d per stage
    #   depthwise: Cin, kH=5, kW=1, stride=(2,1)
    #   pointwise: Cin → Cout, 1×1
    for i in range(len(kn) - 1):
        C_in  = kn[i]  // 2   # complex channels (real part only count)
        C_out = kn[i+1] // 2
        H_in  = freq_sizes[i]
        W_in  = T_FRAMES
        H_out = freq_sizes[i+1]

        # Binaural: each layer runs ×2 (L + R)
        # We report per-channel-pass and annotate ×2 in name
        for ch in ('L', 'R'):
            # Depthwise
            row = analyze_depthwise_conv(
                layer_name=f'Enc{i+1}.{ch}.depthwise',
                C_in=C_in, C_out=C_in,
                kH=KERNEL_SIZE, kW=1,
                H_in=H_in, W_in=W_in,
                stride=2,
            )
            rows.append(row)

            # Pointwise
            row = analyze_pointwise_conv(
                layer_name=f'Enc{i+1}.{ch}.pointwise',
                C_in=C_in, C_out=C_out,
                H=H_out, W=W_in,
            )
            rows.append(row)

    # ── ATTENTION BLOCK ──────────────────────────────────────────────────────
    # Input to mattn: concat(enc_out_L, enc_out_R) → C = 2×kn[-1]//2 = kn[-1]
    # Flatten: input_size = kn[-1] * freq_sizes[-1] = 256 * 2 = 512
    # (binaural: base_flat = (kn[-1]//2) * freq_sizes[-1]; flatten_size = base_flat * 2)
    base_flat    = (kn[-1] // 2) * freq_sizes[-1]
    flatten_size = base_flat * 2   # binaural
    # T after bottleneck = T_FRAMES (no temporal downsampling)
    T_attn = T_FRAMES

    # input_proj: flatten_size → EMBED_DIM
    rows.append(analyze_linear_proj(
        'Attn.input_proj', flatten_size, EMBED_DIM, T_attn, is_complex=True))

    # MHA QKV projection (Wq, Wk, Wv packed as in_proj)
    rows.append(analyze_mha_qkv_proj(
        'Attn.MHA.qkv_proj', EMBED_DIM, T_attn, NUM_HEADS))

    # QK^T (windowed)
    rows.append(analyze_qkt_matmul(
        'Attn.MHA.QKT', EMBED_DIM, T_attn, NUM_HEADS, ATTN_WIN))

    # Softmax
    rows.append(analyze_softmax(
        'Attn.MHA.softmax', T_attn, NUM_HEADS, ATTN_WIN))

    # Score × V
    rows.append(analyze_score_v_matmul(
        'Attn.MHA.scoreV', EMBED_DIM, T_attn, NUM_HEADS, ATTN_WIN))

    # out_proj
    rows.append(analyze_mha_out_proj(
        'Attn.MHA.out_proj', EMBED_DIM, T_attn))

    # output_proj: EMBED_DIM → flatten_size
    rows.append(analyze_linear_proj(
        'Attn.output_proj', EMBED_DIM, flatten_size, T_attn, is_complex=True))

    # ── DECODER (×2 binaural L+R) ───────────────────────────────────────────
    # Skip-connections double input channels: Cin_dec = 2 × kn[i]
    # ComplexDepthwiseSeparableConvTranspose2d: pointwise first then depthwise-T
    for i in range(len(kn) - 1, 0, -1):
        dec_idx = len(kn) - 1 - i   # 0..5

        # After skip concat: input channels = 2 × kn[i] (concat with encoder skip)
        C_in_dec  = kn[i]           # ×2 already reflected in kernel_num for decoder
        C_out_dec = kn[i-1] // 2
        H_in_dec  = freq_sizes[i]
        W_in_dec  = T_FRAMES
        H_out_dec = freq_sizes[i-1]

        for ch in ('L', 'R'):
            # Pointwise first: C_in_dec → C_out_dec
            row = analyze_pointwise_conv(
                layer_name=f'Dec{dec_idx+1}.{ch}.pointwise',
                C_in=C_in_dec, C_out=C_out_dec,
                H=H_in_dec, W=W_in_dec,
            )
            rows.append(row)

            # Depthwise transposed: C_out_dec, stride=2 (upsample)
            row = analyze_depthwise_conv(
                layer_name=f'Dec{dec_idx+1}.{ch}.depthwise_T',
                C_in=C_out_dec, C_out=C_out_dec,
                kH=KERNEL_SIZE, kW=1,
                H_in=H_in_dec, W_in=W_in_dec,
                stride=2,
                is_transpose=True,
            )
            rows.append(row)

    return rows


def add_w4_column(rows):
    for row in rows:
        row['util_w4a8'] = w4_utilization(row)
    return rows


# ─── Load checkpoint (optional – for param count validation) ─────────────────

def load_model_params():
    """Try to load checkpoint and return actual param counts per layer."""
    if not os.path.exists(CKPT_PATH):
        print(f"  [warn] checkpoint not found: {CKPT_PATH}")
        return None
    print(f"  Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        # strip Lightning prefix
        sd = {k.replace('model.', '', 1): v for k, v in sd.items()}
    else:
        sd = ckpt
    return sd


# ─── Output ─────────────────────────────────────────────────────────────────

FIELDNAMES = [
    'layer_name', 'type', 'params', 'MACs',
    'compute_cycles', 'weight_dma_cycles', 'act_dma_cycles',
    'utilization(%)', 'bottleneck', 'util_w4a8(%)', 'note',
]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            w.writerow({
                'layer_name':        r['layer_name'],
                'type':              r['type'],
                'params':            r['params'],
                'MACs':              r['MACs'],
                'compute_cycles':    r['compute_cycles'],
                'weight_dma_cycles': r['weight_dma_cycles'],
                'act_dma_cycles':    r['act_dma_cycles'],
                'utilization(%)':    r['utilization'],
                'bottleneck':        r['bottleneck'],
                'util_w4a8(%)':      r['util_w4a8'],
                'note':              r.get('note', ''),
            })
    print(f"  CSV  → {path}")


def print_table(rows):
    hdr = (f"{'layer_name':<35} {'type':<18} {'params':>8} {'MACs':>12} "
           f"{'cmp_cyc':>9} {'wdma_cyc':>9} {'adma_cyc':>9} "
           f"{'util%':>6} {'bottleneck':<10} {'w4util%':>8}")
    sep = '─' * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in rows:
        print(f"{r['layer_name']:<35} {r['type']:<18} {r['params']:>8,} "
              f"{r['MACs']:>12,} {r['compute_cycles']:>9,} "
              f"{r['weight_dma_cycles']:>9,} {r['act_dma_cycles']:>9,} "
              f"{r['utilization']:>6.1f} {r['bottleneck']:<10} "
              f"{r['util_w4a8']:>8.1f}")
    print(sep)


def write_summary(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    total_compute = sum(r['compute_cycles'] for r in rows)
    total_wdma    = sum(r['weight_dma_cycles'] for r in rows)
    total_adma    = sum(r['act_dma_cycles'] for r in rows)
    total_macs    = sum(r['MACs'] for r in rows)

    # Sorted by utilization ascending (worst first)
    sorted_rows = sorted(
        [r for r in rows if r['utilization'] > 0],
        key=lambda r: r['utilization']
    )
    top3_worst = sorted_rows[:3]
    top3_best  = sorted_rows[-3:][::-1]

    # DMA vs compute breakdown
    total_dma = total_wdma + total_adma

    # Group sums
    enc_macs  = sum(r['MACs'] for r in rows if r['layer_name'].startswith('Enc'))
    dec_macs  = sum(r['MACs'] for r in rows if r['layer_name'].startswith('Dec'))
    attn_macs = sum(r['MACs'] for r in rows if r['layer_name'].startswith('Attn'))

    # W4A8 average
    utils_fp8 = [r['utilization']  for r in rows if r['utilization'] > 0]
    utils_w4  = [r['util_w4a8']    for r in rows if r['utilization'] > 0]

    lines = [
        "=" * 72,
        "BCCTN 256-MAC Utilization Analysis — Key Findings",
        f"Model config: fft_len=128, embed_dim={EMBED_DIM}, num_heads={NUM_HEADS}",
        f"Input: {DURATION}s stereo @ {SR}Hz → T={T_FRAMES} frames, F={F_BINS} bins",
        "=" * 72,
        "",
        "── Total Cycle Budget ─────────────────────────────────────────────",
        f"  Total MACs            : {total_macs:,}",
        f"  Compute cycles        : {total_compute:,}",
        f"  Weight DMA cycles     : {total_wdma:,}",
        f"  Activation DMA cycles : {total_adma:,}",
        f"  DMA : Compute ratio   : {(total_dma / total_compute):.2f}:1"
            if total_compute else "  DMA:Compute = N/A",
        "",
        "── MAC Distribution ───────────────────────────────────────────────",
        f"  Encoder MACs  : {enc_macs:>15,}  ({enc_macs/total_macs*100:.1f}%)" if total_macs else "",
        f"  Decoder MACs  : {dec_macs:>15,}  ({dec_macs/total_macs*100:.1f}%)" if total_macs else "",
        f"  Attention MACs: {attn_macs:>15,}  ({attn_macs/total_macs*100:.1f}%)" if total_macs else "",
        "",
        "── Q1: Lowest MAC Utilization Layers (worst 3) ────────────────────",
    ]
    for r in top3_worst:
        lines.append(f"  {r['layer_name']:<38} util={r['utilization']:5.1f}%  "
                     f"type={r['type']}  bottleneck={r['bottleneck']}")
        lines.append(f"    compute={r['compute_cycles']:,}  wdma={r['weight_dma_cycles']:,}  "
                     f"adma={r['act_dma_cycles']:,}  {r.get('note','')}")
    lines += [
        "",
        "── Q2: DMA vs Compute Fraction ─────────────────────────────────────",
        f"  Compute fraction : {total_compute/(total_compute+total_dma)*100:.1f}%",
        f"  DMA fraction     : {total_dma/(total_compute+total_dma)*100:.1f}%",
        f"  → {'DMA-bound' if total_dma > total_compute else 'Compute-bound'} overall",
        "",
        "── Q3: Depthwise Conv Effective Utilization ────────────────────────",
        f"  kernel_size={KERNEL_SIZE} × kW=1 → only {KERNEL_SIZE}/256 = {KERNEL_SIZE/NUM_MACS*100:.1f}% of MACs active",
        "  per spatial tile. Depthwise layers are severely PE-underutilized.",
        "",
        "── Q4: MHA Bottleneck (DMA vs Compute) ─────────────────────────────",
    ]
    attn_rows = [r for r in rows if r['layer_name'].startswith('Attn')]
    for r in attn_rows:
        if r['MACs'] > 0:
            lines.append(f"  {r['layer_name']:<38} util={r['utilization']:5.1f}%  "
                         f"bottleneck={r['bottleneck']}")
    lines += [
        "",
        "── Q5: FP8 vs W4A8 Utilization Impact ──────────────────────────────",
        f"  FP8  avg utilization (non-zero layers): {np.mean(utils_fp8):.1f}%",
        f"  W4A8 avg utilization (non-zero layers): {np.mean(utils_w4):.1f}%",
        f"  W4 DMA halved → utilization improvement: "
        f"+{np.mean(utils_w4)-np.mean(utils_fp8):.1f}pp avg",
        "",
        "── Top-3 Highest Utilization Layers ────────────────────────────────",
    ]
    for r in top3_best:
        lines.append(f"  {r['layer_name']:<38} util={r['utilization']:5.1f}%  "
                     f"type={r['type']}")
    lines += ["", "=" * 72]

    txt = "\n".join(lines)
    with open(path, 'w') as f:
        f.write(txt)
    print(f"  Summary → {path}")
    print("\n" + txt)


def write_plot(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Only rows with non-zero MACs
    plot_rows = [r for r in rows if r['MACs'] > 0]

    names     = [r['layer_name'] for r in plot_rows]
    util_fp8  = [r['utilization'] for r in plot_rows]
    util_w4   = [r['util_w4a8']   for r in plot_rows]

    x = np.arange(len(names))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(max(16, len(names) * 0.55), 12))

    # ── Top: utilization bar chart ──
    ax = axes[0]
    bars1 = ax.bar(x - w/2, util_fp8, w, label='FP8 (fp8 weights)',
                   color='steelblue', alpha=0.85)
    bars2 = ax.bar(x + w/2, util_w4,  w, label='W4A8',
                   color='coral', alpha=0.85)
    ax.set_xlabel('Layer')
    ax.set_ylabel('MAC Utilization (%)')
    ax.set_title('BCCTN Per-Layer MAC Utilization (256-MAC ASIC)\n'
                 'FP8 vs W4A8 — lower = DMA-bottlenecked')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.legend()
    ax.axhline(50, color='gray', linestyle='--', linewidth=0.8, label='50% threshold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)

    # ── Bottom: cycle breakdown stacked bar ──
    ax2 = axes[1]
    comp  = np.array([r['compute_cycles']    for r in plot_rows], dtype=float)
    wdma  = np.array([r['weight_dma_cycles'] for r in plot_rows], dtype=float)
    adma  = np.array([r['act_dma_cycles']    for r in plot_rows], dtype=float)
    # Normalize to 100%
    total = comp + wdma + adma
    total = np.where(total == 0, 1, total)
    ax2.bar(x, comp/total*100,  label='Compute',      color='steelblue', alpha=0.85)
    ax2.bar(x, wdma/total*100,  bottom=comp/total*100,
            label='Weight DMA', color='orange', alpha=0.85)
    ax2.bar(x, adma/total*100,
            bottom=(comp+wdma)/total*100,
            label='Act DMA',    color='tomato', alpha=0.85)
    ax2.set_xlabel('Layer')
    ax2.set_ylabel('Cycle Fraction (%)')
    ax2.set_title('Cycle Breakdown: Compute vs DMA per Layer')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot  → {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BCCTN MAC Utilization Profiler")
    print(f"  NUM_MACS={NUM_MACS}, BUS_WIDTH={BUS_WIDTH_BYTES}B, "
          f"FP8 weights, T={T_FRAMES} frames")
    print("=" * 60)

    # (Optional) load checkpoint to verify existence
    load_model_params()

    # Build analytical layer table
    print("\n[1/4] Building analytical layer table...")
    rows = build_layer_table()
    rows = add_w4_column(rows)
    print(f"  {len(rows)} layers enumerated")

    # Print table
    print_table(rows)

    results_dir = os.path.join(os.path.dirname(__file__), 'results')

    # Write CSV
    print("\n[2/4] Writing CSV...")
    write_csv(rows, os.path.join(results_dir, 'utilization_table.csv'))

    # Write summary
    print("\n[3/4] Writing summary...")
    write_summary(rows, os.path.join(results_dir, 'utilization_summary.txt'))

    # Write plot
    print("\n[4/4] Generating plot...")
    write_plot(rows, os.path.join(results_dir, 'utilization_plot.png'))

    print("\nDone.")


if __name__ == '__main__':
    main()
