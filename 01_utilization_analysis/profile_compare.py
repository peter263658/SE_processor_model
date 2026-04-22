#!/usr/bin/env python3
"""
Compare per-layer FLOPs (MACs), parameters, and wall-clock time
between embed_dim=256 (num_heads=16) and embed_dim=512 (num_heads=32).

Counts MACs at the real-arithmetic level:
  - Each real Conv2d/ConvTranspose2d hook fires naturally 4x per complex layer
    (apply_complex calls fr/fi twice each → 4 real conv calls total)
  - Complex nn.Linear (dtype=complex64) is multiplied by 4 in the hook
  - Each nn.MultiheadAttention hook fires 4x per ComplexMultiheadAttention
    (2 calls in _forward_real + 2 calls in _forward_imaginary)

Usage (run from this directory):
    python profile_compare.py
    python profile_compare.py --cpu-only
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

from DCNN.models.binaural_attention_model import BinauralAttentionDCNN

# ─── Config ────────────────────────────────────────────────────────────────────

SR      = 16000
DUR     = 2.0
N_WARM  = 5
N_RUNS  = 20

BASE_CFG = dict(
    win_len=128, win_inc=80, fft_len=128, win_type='hann',
    kernel_size=5, kernel_num=[16, 32, 64, 128, 256, 256],
    masking_mode='E', use_clstm=False, use_depthwise=True,
    rnn_layers=2, rnn_units=128, attn_win=31, binaural=True,
)

VARIANTS = {
    'dim_256': {**BASE_CFG, 'embed_dim': 256, 'num_heads': 16},
    'dim_512': {**BASE_CFG, 'embed_dim': 512, 'num_heads': 32},
}

GROUPS = ['Encoder', 'Decoder', 'Attn.in_proj', 'Attn.MHA', 'Attn.out_proj']

# ─── FLOPs counter ─────────────────────────────────────────────────────────────

class FlopCounter:
    """
    Registers hooks on leaf primitive modules and accumulates MACs.
    Skips nn.Linear that live inside nn.MultiheadAttention to avoid
    double-counting the MHA output projection.
    """
    def __init__(self):
        self.macs     = defaultdict(int)   # module_id → total MACs
        self.handles  = []

    def register(self, model):
        # Build set of module-ids that are children of any MHA
        # (so we don't double-count MHA.out_proj via the linear hook)
        inside_mha = set()
        for m in model.modules():
            if isinstance(m, nn.MultiheadAttention):
                for child in m.modules():
                    inside_mha.add(id(child))

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(m.register_forward_hook(self._conv2d(m)))
            elif isinstance(m, nn.ConvTranspose2d):
                self.handles.append(m.register_forward_hook(self._convT2d(m)))
            elif isinstance(m, nn.Linear) and id(m) not in inside_mha:
                self.handles.append(m.register_forward_hook(self._linear(m)))
            elif isinstance(m, nn.MultiheadAttention):
                self.handles.append(m.register_forward_hook(self._mha(m)))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    def _conv2d(self, _m):
        def hook(mod, inp, out):
            x = inp[0]
            B, Ci, _, _ = x.shape
            _, Co, Ho, Wo = out.shape
            kH, kW = mod.kernel_size
            macs = B * Co * Ho * Wo * (Ci // mod.groups) * kH * kW
            self.macs[id(mod)] += macs
        return hook

    def _convT2d(self, _m):
        def hook(mod, inp, out):
            x = inp[0]
            B, Ci, Hi, Wi = x.shape
            _, Co, _, _ = out.shape
            kH, kW = mod.kernel_size
            # transposed conv: input spatial × output channels per group × kernel
            macs = B * Ci * Hi * Wi * (Co // mod.groups) * kH * kW
            self.macs[id(mod)] += macs
        return hook

    def _linear(self, _m):
        def hook(mod, inp, _out):
            x = inp[0]
            batch_seq = x.numel() // x.shape[-1]
            # complex64 linear → 4× real MACs (each complex mul = 4 real muls)
            scale = 4 if torch.is_complex(x) else 1
            macs = batch_seq * mod.in_features * mod.out_features * scale
            self.macs[id(mod)] += macs
        return hook

    def _mha(self, _m):
        def hook(mod, inp, _out):
            q = inp[0]
            B, T, _ = (q.shape if mod.batch_first else (q.shape[1], q.shape[0], q.shape[2]))
            E = mod.embed_dim
            W = 31  # sliding window size (attn_win)
            # Per real MHA:
            #   QKV projections : 3 × B×T×E×E
            #   Attention scores : B×T×W×E  (windowed)
            #   Weighted values  : B×T×W×E
            #   Output projection: B×T×E×E
            # Total per real MHA: B*T*(4*E^2 + 2*W*E)
            macs = B * T * (4 * E * E + 2 * W * E)
            self.macs[id(mod)] += macs
        return hook


def build_id_map(model):
    """Map each primitive module id → high-level group label."""
    groups = {
        'Encoder':       model.encoder,
        'Decoder':       model.decoder,
        'Attn.in_proj':  model.mattn.input_proj,
        'Attn.MHA':      model.mattn.mattn,
        'Attn.out_proj': model.mattn.output_proj,
    }
    id_map = {}
    for label, mod in groups.items():
        for m in mod.modules():
            id_map[id(m)] = label
    return id_map


def sum_flops_by_group(counter, id_map):
    group_macs = defaultdict(int)
    for mid, macs in counter.macs.items():
        label = id_map.get(mid, 'Other')
        group_macs[label] += macs
    return dict(group_macs)


# ─── Param counter ─────────────────────────────────────────────────────────────

def count_params(model):
    def p(mod): return sum(x.numel() for x in mod.parameters())
    attn = model.mattn
    return {
        'Encoder':       p(model.encoder),
        'Decoder':       p(model.decoder),
        'Attn.in_proj':  p(attn.input_proj),
        'Attn.MHA':      p(attn.mattn),
        'Attn.out_proj': p(attn.output_proj),
        'Total':         p(model),
    }


# ─── Timing ────────────────────────────────────────────────────────────────────

class ModuleTimer:
    """Wall-clock timing for named sub-modules using CUDA events or perf_counter."""
    def __init__(self, device):
        self.device = device
        self.times   = defaultdict(list)
        self.handles = []

    def register(self, named):
        for name, mod in named.items():
            self.handles.append(mod.register_forward_pre_hook(self._pre(name)))
            self.handles.append(mod.register_forward_hook(self._post(name)))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    def _pre(self, name):
        def hook(mod, inp):
            if self.device.type == 'cuda':
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                mod._t_start = e
            else:
                mod._t_start = time.perf_counter()
        return hook

    def _post(self, name):
        def hook(mod, inp, out):
            if self.device.type == 'cuda':
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                torch.cuda.synchronize()
                self.times[name].append(mod._t_start.elapsed_time(end))
            else:
                self.times[name].append((time.perf_counter() - mod._t_start) * 1000)
        return hook

    def avg_ms(self, name):
        t = self.times.get(name, [])
        return np.mean(t) if t else 0.0

    def std_ms(self, name):
        t = self.times.get(name, [])
        return np.std(t) if t else 0.0


# ─── Helpers ───────────────────────────────────────────────────────────────────

def fmt_macs(n):
    if n >= 1e9: return f'{n/1e9:7.2f} G'
    if n >= 1e6: return f'{n/1e6:7.2f} M'
    if n >= 1e3: return f'{n/1e3:7.1f} K'
    return f'{n:10d}'

def fmt_params(n):
    if n >= 1e6: return f'{n/1e6:7.3f} M'
    if n >= 1e3: return f'{n/1e3:7.1f} K'
    return f'{n:10d}'


# ─── Per-variant profiling ──────────────────────────────────────────────────────

def profile_variant(name, cfg, device):
    print(f'\n  Building model: {name} | device={device} ...', flush=True)
    # Suppress construction prints
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    model = BinauralAttentionDCNN(**cfg).to(device).eval()
    sys.stdout = old_stdout

    # BinauralAttentionDCNN forward calls Encoder/STFT/iSTFT twice (L+R channel),
    # Attention once. Hooks fire naturally for each call, so FLOPs and timing
    # correctly reflect the full binaural forward pass.
    x = torch.randn(1, 2, int(SR * DUR), device=device)

    # ── FLOPs (single forward pass) ──
    counter = FlopCounter()
    counter.register(model)
    id_map = build_id_map(model)
    with torch.no_grad():
        model(x)
    flops = sum_flops_by_group(counter, id_map)
    counter.remove()

    # ── Timing ──
    timing_modules = {
        'Encoder':   model.encoder,
        'Decoder':   model.decoder,
        'Attention': model.mattn,
        'STFT':      model.stft,
        'iSTFT':     model.istft,
    }
    timer = ModuleTimer(device)
    timer.register(timing_modules)

    with torch.no_grad():
        for _ in range(N_WARM):
            model(x)
    timer.times.clear()
    with torch.no_grad():
        for _ in range(N_RUNS):
            model(x)
    timer.remove()

    # ── Params ──
    params = count_params(model)

    return params, flops, timer


# ─── Print helpers ─────────────────────────────────────────────────────────────

def print_single_result(name, device, params, flops, timer):
    total_macs   = sum(flops.values())
    total_params = params['Total']
    print(f'\n{"─"*72}')
    print(f'  {name} | device={device}')
    print(f'{"─"*72}')
    print(f'  {"Group":<18} {"Params":>10} {"%":>6}  {"MACs":>12} {"%":>6}')
    print(f'  {"-"*56}')
    for g in GROUPS:
        par  = params.get(g, 0)
        macs = flops.get(g, 0)
        print(f'  {g:<18} {fmt_params(par):>10} {par/total_params*100:5.1f}%'
              f'  {fmt_macs(macs):>12} {macs/total_macs*100:5.1f}%')
    print(f'  {"-"*56}')
    print(f'  {"Total":<18} {fmt_params(total_params):>10} {"100%":>6}'
          f'  {fmt_macs(total_macs):>12} {"100%":>6}')

    print(f'\n  Timing (avg ± std over {N_RUNS} runs):')
    print(f'  {"Module":<14} {"Avg (ms)":>10} {"±Std":>8} {"% total":>8}')
    print(f'  {"-"*44}')
    total_t = sum(timer.avg_ms(k) for k in timing_modules_order)
    for k in timing_modules_order:
        avg = timer.avg_ms(k)
        std = timer.std_ms(k)
        pct = avg / total_t * 100 if total_t > 0 else 0
        print(f'  {k:<14} {avg:>10.2f} {std:>8.2f} {pct:>7.1f}%')
    print(f'  {"-"*44}')
    print(f'  {"Sum":<14} {total_t:>10.2f}')


timing_modules_order = ['STFT', 'Encoder', 'Attention', 'Decoder', 'iSTFT']


def print_comparison(r256, r512, device):
    params_256, flops_256, timer_256 = r256
    params_512, flops_512, timer_512 = r512

    total_macs_256 = sum(flops_256.values())
    total_macs_512 = sum(flops_512.values())

    print(f'\n{"═"*72}')
    print(f'  COMPARISON: dim_256 vs dim_512  |  device={device}')
    print(f'{"═"*72}')

    # Parameters
    print(f'\n  PARAMETERS')
    print(f'  {"Group":<18} {"dim_256":>12} {"dim_512":>12} {"ratio":>8}')
    print(f'  {"-"*54}')
    for g in GROUPS + ['Total']:
        p256 = params_256.get(g, 0)
        p512 = params_512.get(g, 0)
        ratio = p512 / p256 if p256 > 0 else float('nan')
        print(f'  {g:<18} {fmt_params(p256):>12} {fmt_params(p512):>12} {ratio:>7.2f}x')

    # FLOPs
    print(f'\n  MACs (real-arithmetic)')
    print(f'  {"Group":<18} {"dim_256":>12} {"dim_512":>12} {"ratio":>8}')
    print(f'  {"-"*54}')
    all_g = GROUPS + ['Other'] if 'Other' in flops_256 or 'Other' in flops_512 else GROUPS
    for g in all_g:
        f256 = flops_256.get(g, 0)
        f512 = flops_512.get(g, 0)
        ratio = f512 / f256 if f256 > 0 else float('nan')
        print(f'  {g:<18} {fmt_macs(f256):>12} {fmt_macs(f512):>12} {ratio:>7.2f}x')
    ratio_total = total_macs_512 / total_macs_256 if total_macs_256 > 0 else float('nan')
    print(f'  {"-"*54}')
    print(f'  {"Total":<18} {fmt_macs(total_macs_256):>12} {fmt_macs(total_macs_512):>12} {ratio_total:>7.2f}x')

    # Timing
    print(f'\n  TIMING — avg over {N_RUNS} runs (ms)')
    print(f'  {"Module":<14} {"dim_256":>10} {"dim_512":>10} {"ratio":>8}')
    print(f'  {"-"*46}')
    for k in timing_modules_order:
        t256 = timer_256.avg_ms(k)
        t512 = timer_512.avg_ms(k)
        ratio = t512 / t256 if t256 > 0 else float('nan')
        print(f'  {k:<14} {t256:>10.2f} {t512:>10.2f} {ratio:>7.2f}x')
    sum_256 = sum(timer_256.avg_ms(k) for k in timing_modules_order)
    sum_512 = sum(timer_512.avg_ms(k) for k in timing_modules_order)
    ratio_t = sum_512 / sum_256 if sum_256 > 0 else float('nan')
    print(f'  {"-"*46}')
    print(f'  {"Sum":<14} {sum_256:>10.2f} {sum_512:>10.2f} {ratio_t:>7.2f}x')

    # Bottleneck analysis
    print(f'\n  BOTTLENECK BREAKDOWN (dim_256, % of total MACs)')
    for g in GROUPS:
        f256 = flops_256.get(g, 0)
        print(f'  {g:<20} {f256/total_macs_256*100:6.1f}%  {fmt_macs(f256)}')

    print(f'\n  BOTTLENECK BREAKDOWN (dim_512, % of total MACs)')
    for g in GROUPS:
        f512 = flops_512.get(g, 0)
        print(f'  {g:<20} {f512/total_macs_512*100:6.1f}%  {fmt_macs(f512)}')


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true',
                        help='Skip CUDA profiling even if GPU is available')
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu_only
    devices  = [torch.device('cpu')]
    if use_cuda:
        devices.append(torch.device('cuda'))

    print(f'BCCTN Profile Compare: dim_256 vs dim_512')
    print(f'  SR={SR} Hz, duration={DUR}s → {int(SR*DUR)} samples')
    print(f'  N_WARM={N_WARM}, N_RUNS={N_RUNS}')
    print(f'  Devices: {[str(d) for d in devices]}')

    all_results = {}  # (variant, device) → (params, flops, timer)

    for dev in devices:
        results_dev = {}
        for vname, cfg in VARIANTS.items():
            params, flops, timer = profile_variant(vname, cfg, dev)
            results_dev[vname] = (params, flops, timer)
            print_single_result(vname, dev, params, flops, timer)

        # Comparison on this device
        if len(VARIANTS) == 2:
            vnames = list(VARIANTS.keys())
            print_comparison(results_dev[vnames[0]], results_dev[vnames[1]], dev)

        all_results[str(dev)] = results_dev

    print(f'\n{"═"*72}')
    print('  Done.')


if __name__ == '__main__':
    main()
