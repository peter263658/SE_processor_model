"""
Weight Dynamic Range Analyzer for BCCTN
Per-layer, per-channel, and per-head granularity analysis.
"""
import argparse
import os
import json
import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def to_numpy(x):
    """Convert torch tensor or numpy array to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def compute_global_stats(arr, threshold=1e-6):
    """arr: numpy ndarray of weight magnitudes (non-negative)."""
    flat = arr.flatten()
    _min = float(np.min(flat))
    _max = float(np.max(flat))
    _mean = float(np.mean(flat))
    _std = float(np.std(flat))
    _kurtosis = float(scipy.stats.kurtosis(flat))
    _skew = float(scipy.stats.skew(flat))

    valid = flat[flat > threshold]
    _min_valid = float(np.min(valid)) if len(valid) > 0 else _min
    _dynamic_range_ratio = (_max / _min_valid) if _min_valid > 0 else 0.0

    return {
        'min': _min, 'max': _max, 'mean': _mean, 'std': _std,
        'kurtosis': _kurtosis, 'skewness': _skew,
        'dynamic_range_ratio': _dynamic_range_ratio,
    }


def compute_group_stats(groups):
    """groups: list of 1-D numpy arrays."""
    stats = []
    for g in groups:
        g = np.asarray(g).flatten()
        if len(g) == 0:
            continue
        stats.append({
            'min': float(np.min(g)),
            'max': float(np.max(g)),
            'std': float(np.std(g)),
            'kurtosis': float(scipy.stats.kurtosis(g)),
        })
    return stats


def cross_variance(stats_list):
    if not stats_list:
        return {'var_min': 0.0, 'var_max': 0.0, 'var_std': 0.0, 'var_kurtosis': 0.0}
    df = pd.DataFrame(stats_list)
    return {
        'var_min': float(df['min'].var()),
        'var_max': float(df['max'].var()),
        'var_std': float(df['std'].var()),
        'var_kurtosis': float(df['kurtosis'].var()),
    }


def quantize_w4_mse(arr, group_size):
    """
    Compute INT4 symmetric quantization MSE with the given group_size.
    Correctly handles tensors smaller than group_size without padding bias.
    arr: 1-D numpy array of raw (signed) weight values.
    """
    flat = arr.flatten()
    N = len(flat)
    if N == 0:
        return 0.0

    # If tensor is smaller than group_size, treat it as a single group
    eff_gsz = min(group_size, N)

    num_complete = N // eff_gsz
    remainder = N % eff_gsz

    total_mse = 0.0
    total_count = 0

    if num_complete > 0:
        groups = flat[:num_complete * eff_gsz].reshape(num_complete, eff_gsz)
        max_vals = np.max(np.abs(groups), axis=1, keepdims=True)
        scales = np.where(max_vals == 0, 1e-9, max_vals / 7.0)
        q = np.clip(np.round(groups / scales), -7, 7)
        dq = q * scales
        total_mse += float(np.sum((groups - dq) ** 2))
        total_count += num_complete * eff_gsz

    if remainder > 0:
        rem = flat[num_complete * eff_gsz:]
        scale_r = np.max(np.abs(rem)) / 7.0
        if scale_r == 0:
            scale_r = 1e-9
        q_r = np.clip(np.round(rem / scale_r), -7, 7)
        total_mse += float(np.sum((rem - q_r * scale_r) ** 2))
        total_count += remainder

    return total_mse / total_count if total_count > 0 else 0.0


def classify_layer(key):
    """
    Return ('conv_dw'|'conv_pw'|'mha_inproj'|'mha_outproj'|'mha_linear'|'bn'|'other')
    based on key name from the BCCTN checkpoint.
    """
    k = key.lower()
    if 'depthwise' in k:
        return 'conv_dw'
    if 'pointwise' in k:
        return 'conv_pw'
    if 'in_proj_weight' in k:
        return 'mha_inproj'
    if 'out_proj' in k and 'mattn' in k:
        return 'mha_outproj'
    if ('input_proj' in k or 'output_proj' in k) and 'mattn' in k:
        return 'mha_linear'
    if 'bn_r' in k or 'bn_i' in k or 'bn.weight' in k:
        return 'bn'
    if 'prelu' in k:
        return 'prelu'
    return 'other'


def plot_histogram(mag_arr, layer_name, channel_groups, out_dir):
    """mag_arr: 1-D numpy array of weight magnitudes."""
    plt.figure(figsize=(10, 6))
    flat = mag_arr.flatten()

    sns.histplot(flat, bins=100, color='blue', alpha=0.3, label='Overall', stat='density')

    # overlay first 10 per-channel KDEs
    if 0 < len(channel_groups) <= 256:
        for g in channel_groups[:10]:
            g_flat = np.asarray(g).flatten()
            if len(g_flat) >= 2 and np.std(g_flat) > 0:
                try:
                    sns.kdeplot(g_flat, lw=1, alpha=0.4)
                except Exception:
                    pass

    mean, std = float(np.mean(flat)), float(np.std(flat))
    plt.axvline(mean + 3 * std, color='red', linestyle='dashed', label='+3σ')

    plt.title(f'Weight Magnitude: {layer_name}', fontsize=9)
    plt.xlabel('Magnitude')
    plt.ylabel('Density')
    plt.legend(fontsize=8)

    safe = layer_name.replace('.', '_').replace('/', '_')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{safe}.png'), dpi=80)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--num_heads', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hist_dir = os.path.join(args.out_dir, 'histograms')
    os.makedirs(hist_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)

    group_sizes = [16, 32, 64, 128, 256]
    num_heads = args.num_heads

    global_stats_list = []
    per_channel_var_list = []
    per_head_var_list = []
    group_size_sensitivity = []
    format_recs = {}

    # For real/imag comparison
    complex_pairs = {}  # base_key → {real: stats, imag: stats}

    # -----------------------------------------------------------------------
    # Process each weight tensor
    # -----------------------------------------------------------------------
    for k, v in state_dict.items():
        if 'weight' not in k:
            continue

        layer_class = classify_layer(k)

        # Expand complex tensors into real/imag parts
        if v.dtype in (torch.complex64, torch.complex128):
            v_np = to_numpy(v)
            parts = [
                (f'{k}.real', np.abs(v_np.real), v_np.real),
                (f'{k}.imag', np.abs(v_np.imag), v_np.imag),
            ]
            is_complex_pair = True
        else:
            v_np = to_numpy(v)
            parts = [(k, np.abs(v_np), v_np)]
            is_complex_pair = False

        # For in_proj_weight (768×D): split Q/K/V before per-head analysis
        if layer_class == 'mha_inproj' and not is_complex_pair:
            # Expand in_proj into Q, K, V sub-tensors
            rows = v_np.shape[0]
            chunk = rows // 3
            qkv_labels = ['Q', 'K', 'V']
            new_parts = []
            for i, label in enumerate(qkv_labels):
                sub = v_np[i * chunk:(i + 1) * chunk]
                new_parts.append((f'{k}.{label}', np.abs(sub), sub))
            parts = new_parts
            layer_class_for_parts = 'mha_outproj'  # treat Q/K/V blocks as mha_outproj for head-split
        else:
            layer_class_for_parts = layer_class

        # ---------------------------------------------------------------
        # Analyse each part
        # ---------------------------------------------------------------
        for suffix_k, mag_v, raw_v in parts:
            if raw_v.size == 0:
                continue

            mag_flat = mag_v.flatten()
            raw_flat = raw_v.flatten()

            # 1. Global statistics
            g_stats = compute_global_stats(mag_flat)
            g_stats['layer'] = suffix_k
            g_stats['shape'] = str(raw_v.shape)
            g_stats['layer_class'] = layer_class_for_parts
            global_stats_list.append(g_stats)

            # 2. Per-channel / per-head grouping
            channel_groups = []

            lc = layer_class_for_parts
            if lc in ('conv_dw', 'conv_pw') and raw_v.ndim >= 1:
                # dim-0 is output channel
                channel_groups = [raw_v[c].flatten() for c in range(raw_v.shape[0])]
                if len(channel_groups) >= 2:
                    c_stats = compute_group_stats(channel_groups)
                    c_var = cross_variance(c_stats)
                    c_var['layer'] = suffix_k
                    c_var['num_channels'] = len(channel_groups)
                    per_channel_var_list.append(c_var)

            elif lc in ('mha_outproj', 'mha_linear', 'mha_inproj') and raw_v.ndim >= 2:
                # Split along dim-0 into num_heads chunks
                head_arrays = np.array_split(raw_v, num_heads, axis=0)
                channel_groups = [h.flatten() for h in head_arrays]
                if len(channel_groups) >= 2:
                    h_stats = compute_group_stats(channel_groups)
                    h_var = cross_variance(h_stats)
                    h_var['layer'] = suffix_k
                    h_var['num_heads'] = len(channel_groups)
                    per_head_var_list.append(h_var)

            # 3. Histogram
            plot_histogram(mag_flat, suffix_k, channel_groups, hist_dir)

            # 4. Group-size sensitivity (W4 quantization MSE)
            sens = {'layer': suffix_k}
            N = len(raw_flat)
            sens['per_tensor_mse'] = quantize_w4_mse(raw_flat, N)  # full tensor = 1 group
            for gsz in group_sizes:
                sens[f'gsz_{gsz}_mse'] = quantize_w4_mse(raw_flat, gsz)
            group_size_sensitivity.append(sens)

            # 5. Format recommendation
            kurt = g_stats['kurtosis']
            cv_thresh = 1e-4
            cv_val = 0.0
            if lc in ('conv_dw', 'conv_pw') and per_channel_var_list and per_channel_var_list[-1]['layer'] == suffix_k:
                cv_val = per_channel_var_list[-1]['var_max']
            elif lc in ('mha_outproj', 'mha_linear', 'mha_inproj') and per_head_var_list and per_head_var_list[-1]['layer'] == suffix_k:
                cv_val = per_head_var_list[-1]['var_max']

            if kurt > 5:
                rec_fmt = 'FP'
                rec_scale = 'N/A'
                rec_gsz = 'N/A'
            elif kurt < 3 and cv_val < cv_thresh:
                rec_fmt = 'INT4'
                rec_scale = 'per-tensor'
                rec_gsz = 'N/A'
            else:
                rec_fmt = 'INT4'
                rec_scale = 'per-channel/per-group'
                # Pick smallest group_size where MSE ≤ 2× gsz_16 (finest granularity)
                gsz16_mse = sens['gsz_16_mse']
                rec_gsz = 16  # default finest
                for gsz in reversed(group_sizes):
                    if sens[f'gsz_{gsz}_mse'] <= 2.0 * gsz16_mse:
                        rec_gsz = gsz
                        break

            format_recs[suffix_k] = {
                'format': rec_fmt,
                'scale_granularity': rec_scale,
                'group_size': rec_gsz,
                'kurtosis': round(kurt, 4),
                'cross_variance': round(cv_val, 8),
                'dynamic_range_ratio': round(g_stats['dynamic_range_ratio'], 2),
            }

            # Track complex pairs for real/imag comparison
            if is_complex_pair:
                base = k
                if base not in complex_pairs:
                    complex_pairs[base] = {}
                suffix = 'real' if suffix_k.endswith('.real') else 'imag'
                complex_pairs[base][suffix] = g_stats

    # -----------------------------------------------------------------------
    # Save CSVs
    # -----------------------------------------------------------------------
    df_global = pd.DataFrame(global_stats_list)
    df_global.to_csv(os.path.join(args.out_dir, 'dynamic_range_table.csv'), index=False)
    print(f'Wrote dynamic_range_table.csv ({len(df_global)} rows)')

    if per_channel_var_list:
        df_ch = pd.DataFrame(per_channel_var_list)
        df_ch.to_csv(os.path.join(args.out_dir, 'per_channel_variance.csv'), index=False)
        print(f'Wrote per_channel_variance.csv ({len(df_ch)} rows)')
    else:
        print('Warning: per_channel_var_list is empty — check conv layer detection')

    if per_head_var_list:
        df_head = pd.DataFrame(per_head_var_list)
        df_head.to_csv(os.path.join(args.out_dir, 'per_head_variance.csv'), index=False)
        print(f'Wrote per_head_variance.csv ({len(df_head)} rows)')

    df_sens = pd.DataFrame(group_size_sensitivity)
    df_sens.to_csv(os.path.join(args.out_dir, 'group_size_sensitivity.csv'), index=False)

    # -----------------------------------------------------------------------
    # Real / Imaginary comparison CSV
    # -----------------------------------------------------------------------
    real_imag_rows = []
    for base_key, parts in complex_pairs.items():
        if 'real' in parts and 'imag' in parts:
            r, im = parts['real'], parts['imag']
            std_ratio = r['std'] / im['std'] if im['std'] > 0 else float('inf')
            kurt_diff = abs(r['kurtosis'] - im['kurtosis'])
            similar = (0.8 <= std_ratio <= 1.2) and (kurt_diff < 2.0)
            real_imag_rows.append({
                'layer': base_key,
                'real_mean': round(r['mean'], 6),
                'imag_mean': round(im['mean'], 6),
                'real_std': round(r['std'], 6),
                'imag_std': round(im['std'], 6),
                'real_kurtosis': round(r['kurtosis'], 4),
                'imag_kurtosis': round(im['kurtosis'], 4),
                'real_dynamic_range': round(r['dynamic_range_ratio'], 2),
                'imag_dynamic_range': round(im['dynamic_range_ratio'], 2),
                'std_ratio': round(std_ratio, 4),
                'kurtosis_diff': round(kurt_diff, 4),
                'distributions_similar': similar,
            })
    if real_imag_rows:
        pd.DataFrame(real_imag_rows).to_csv(
            os.path.join(args.out_dir, 'real_imag_comparison.csv'), index=False)
        print(f'Wrote real_imag_comparison.csv ({len(real_imag_rows)} rows)')

    # -----------------------------------------------------------------------
    # Group-size sensitivity plot
    # -----------------------------------------------------------------------
    plt.figure(figsize=(10, 5))
    if not df_sens.empty:
        gsz_mses = [df_sens[f'gsz_{g}_mse'].mean() for g in group_sizes]
        per_tensor_avg = df_sens['per_tensor_mse'].mean()

        plt.plot(group_sizes, gsz_mses, marker='o', label='Per-group (avg over all layers)')
        plt.axhline(per_tensor_avg, color='red', linestyle='--', label=f'Per-tensor baseline ({per_tensor_avg:.2e})')
        plt.xlabel('Group Size')
        plt.ylabel('Average Quantization MSE (W4-INT4)')
        plt.title('Group-Size Sensitivity: Quantization Error vs Group Size\n(smaller group = finer scale = lower MSE)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'group_size_plot.png'), dpi=100)
    plt.close()

    # -----------------------------------------------------------------------
    # Format recommendation JSON
    # -----------------------------------------------------------------------
    with open(os.path.join(args.out_dir, 'format_recommendation.json'), 'w') as f:
        json.dump(format_recs, f, indent=2)

    # -----------------------------------------------------------------------
    # Summary markdown — answer all 5 key questions
    # -----------------------------------------------------------------------
    num_fp = sum(1 for v in format_recs.values() if v['format'] == 'FP')
    num_int = len(format_recs) - num_fp
    fp_layers = [k for k, v in format_recs.items() if v['format'] == 'FP']
    per_tensor_layers = [k for k, v in format_recs.items() if v['scale_granularity'] == 'per-tensor']
    per_group_layers = [k for k, v in format_recs.items() if v['scale_granularity'] == 'per-channel/per-group']

    # Q1: Channel/head uniformity
    high_var_heads = [(r['layer'], r['var_max']) for r in per_head_var_list if r['var_max'] > 1e-3]
    uniform_heads = [(r['layer'], r['var_max']) for r in per_head_var_list if r['var_max'] <= 1e-3]

    # Q2: Group-size recommendation
    per_tensor_avg = df_sens['per_tensor_mse'].mean() if not df_sens.empty else 0
    gsz_avgs = {g: df_sens[f'gsz_{g}_mse'].mean() for g in group_sizes} if not df_sens.empty else {}

    # Q4: Real/imag similarity
    n_similar = sum(1 for r in real_imag_rows if r['distributions_similar'])
    n_diff = len(real_imag_rows) - n_similar

    with open(os.path.join(args.out_dir, 'summary.md'), 'w') as f:
        f.write('# Weight Dynamic Range Analysis Summary\n\n')

        f.write('## Q1: Are weight distributions similar across channels/heads?\n\n')
        f.write(f'- **Conv layers**: {len(per_channel_var_list)} conv weight tensors analyzed per-channel.\n')
        high_var_ch = [(r['layer'], r['var_max']) for r in per_channel_var_list if r['var_max'] > 1e-4]
        if high_var_ch:
            f.write(f'  - {len(high_var_ch)} tensors show **high cross-channel variance** (var_max > 1e-4):\n')
            for lname, vm in sorted(high_var_ch, key=lambda x: -x[1])[:10]:
                f.write(f'    - `{lname}` (var_max = {vm:.2e})\n')
        else:
            f.write('  - All conv tensors show **uniform cross-channel distributions** (var_max ≤ 1e-4).\n')
        f.write(f'\n- **MHA layers**: {len(per_head_var_list)} head-group tensors analyzed.\n')
        f.write(f'  - {len(uniform_heads)} groups are **uniform** (var_max ≤ 1e-3), '
                f'{len(high_var_heads)} show **significant cross-head variance**.\n')
        if high_var_heads:
            f.write('  - High-variance head groups:\n')
            for lname, vm in sorted(high_var_heads, key=lambda x: -x[1])[:10]:
                f.write(f'    - `{lname}` (var_max = {vm:.2e})\n')
        f.write('\n')

        f.write('## Q2: Optimal group_size for W4 scalar factor\n\n')
        f.write(f'- Per-tensor baseline MSE: {per_tensor_avg:.6e}\n')
        for gsz in group_sizes:
            avg = gsz_avgs.get(gsz, 0)
            ratio = avg / per_tensor_avg if per_tensor_avg > 0 else 0
            f.write(f'- group_size={gsz}: avg MSE = {avg:.6e} ({ratio:.2f}x vs per-tensor)\n')
        f.write('\n')
        gsz16 = gsz_avgs.get(16, 0)
        best_gsz = 16
        for gsz in reversed(group_sizes):
            if gsz_avgs.get(gsz, 0) <= 2.0 * gsz16:
                best_gsz = gsz
                break
        f.write(f'**Recommendation**: group_size = **{best_gsz}** (largest size within 2× MSE of finest granularity gsz=16)\n\n')
        f.write('> Note: smaller group_size = more scale factors = lower MSE (better precision) but higher hardware overhead (~6% at gsz=32).\n\n')

        f.write('## Q3: Layers with outliers requiring FP format\n\n')
        f.write(f'- {num_fp} of {len(format_recs)} weight tensors are recommended for **FP** (kurtosis > 5).\n')
        if fp_layers:
            f.write('- FP-recommended layers:\n')
            for lname in fp_layers:
                k_val = format_recs[lname]['kurtosis']
                f.write(f'  - `{lname}` (kurtosis = {k_val:.2f})\n')
        f.write(f'- {num_int} weight tensors are recommended for **INT4**.\n')
        f.write(f'  - {len(per_tensor_layers)} can use per-tensor scale, '
                f'{len(per_group_layers)} need per-channel/per-group scaling.\n\n')

        f.write('## Q4: Real vs Imaginary weight distributions\n\n')
        f.write(f'- {n_similar} complex layers have **similar** real/imag distributions '
                f'(std ratio within 20%, kurtosis diff < 2).\n')
        f.write(f'- {n_diff} complex layers show **significant differences** between real and imaginary components.\n')
        if n_diff == 0:
            f.write('- **Conclusion**: Real and imaginary components can share the same scale factor.\n\n')
        else:
            f.write('- **Conclusion**: Consider separate scale factors for real and imaginary components.\n\n')

        f.write('## Q5: MHA per-head weight magnitude uniformity\n\n')
        if per_head_var_list:
            for r in per_head_var_list:
                tag = '**non-uniform**' if r['var_max'] > 1e-3 else 'uniform'
                f.write(f'- `{r["layer"]}`: var_max = {r["var_max"]:.6e} → {tag}\n')
        else:
            f.write('- No MHA layers analyzed (check key names).\n')
        if high_var_heads:
            f.write('\n**Conclusion**: Some heads have significantly different weight magnitudes — '
                    '**per-head scaling is recommended** for these MHA tensors.\n\n')
        else:
            f.write('\n**Conclusion**: Head magnitudes are uniform — per-tensor scale suffices for MHA.\n\n')

        f.write('## Group Size Sensitivity\n\n')
        f.write('Refer to `group_size_plot.png` for average MSE across different group sizes.\n')
        f.write('Smaller group sizes improve precision (lower MSE) at the cost of more scale factors.\n')

    print(f'\nAnalysis complete. Results in: {args.out_dir}')
    print(f'  Tensors analyzed: {len(format_recs)}')
    print(f'  FP recommended: {num_fp}')
    print(f'  INT4 (per-tensor): {len(per_tensor_layers)}')
    print(f'  INT4 (per-group):  {len(per_group_layers)}')
    if gsz_avgs:
        print(f'  Optimal group_size: {best_gsz}')


if __name__ == '__main__':
    main()
