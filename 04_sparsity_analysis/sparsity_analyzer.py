import os
import argparse
import glob
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# BCCTN imports
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from DCNN.trainer import DCNNLightningModule
import DCNN.utils.complexPyTorch.complexLayers as torch_complex

WEIGHT_THRESHOLDS = [1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
ACT_THRESHOLDS = [1e-3, 1e-2, 1e-1]
BLOCK_SIZES = [4, 8, 16, 32]
PRACTICAL_THRESHOLD = 5e-2  # Used for classification / pruning

class BCCTNSparsityAnalyzer:
    def __init__(self, model, test_dir, num_samples):
        self.model = model
        self.test_dir = test_dir
        self.num_samples = num_samples
        self.device = next(model.parameters()).device
        
        self.weight_stats = []
        self.block_stats = []
        
        self.act_hooks = []
        self.act_stats = {}
        
        self.attn_hooks = []
        self.attn_scores = []
        
        self.classes = {}
        self.pruning_candidates = []
        
    @staticmethod
    def _get_weight_tensor(module):
        """Safely extract weight tensor, handling FP8-QAT ParametrizationList."""
        import torch.nn.utils.parametrize as parametrize
        if parametrize.is_parametrized(module, 'weight'):
            # Run through the parametrization chain to get the final tensor
            return module.parametrizations.weight.original
        return module.weight

    def _is_complex(self, weight):
        return weight.dtype in [torch.complex64, torch.complex128]

    def _get_magnitude(self, weight):
        if self._is_complex(weight):
            return torch.abs(weight)
        return torch.abs(weight)

    def analyze_weights(self):
        print("Analyzing weight sparsity...")
        for name, module in self.model.named_modules():
            # Linear or Conv layers
            try:
                w = self._get_weight_tensor(module)
            except (AttributeError, TypeError):
                continue
            if not isinstance(w, torch.Tensor) or len(w.shape) < 2:
                continue
            w = w.detach()
            mag = self._get_magnitude(w)
            is_mha_proj = 'proj' in name or 'mattn' in name
            
            # We analyze it
            for t in WEIGHT_THRESHOLDS:
                mask = mag < t
                total_zero_ratio = mask.float().mean().item()
                
                # Channel level
                out_channels = mask.shape[0]
                mask_flat_channel = mask.view(out_channels, -1)
                channel_zeros = (mask_flat_channel.sum(dim=1) == mask_flat_channel.shape[1]).float().mean().item()
                
                # Row/Col for MHA
                row_zeros = 0.0
                col_zeros = 0.0
                if is_mha_proj and len(mask.shape) == 2:
                    row_zeros = (mask.sum(dim=1) == mask.shape[1]).float().mean().item()
                    col_zeros = (mask.sum(dim=0) == mask.shape[0]).float().mean().item()
                    
                self.weight_stats.append({
                    'layer': name,
                    'threshold': t,
                    'total_sparsity': total_zero_ratio,
                    'channel_sparsity': channel_zeros,
                    'row_col_sparsity': max(row_zeros, col_zeros)
                })
                
                # Block sparsity
                w_flat = mask.view(-1)
                for b in BLOCK_SIZES:
                    pad = (b - (len(w_flat) % b)) % b
                    if pad > 0:
                        w_pad = torch.nn.functional.pad(w_flat, (0, pad), value=False)
                    else:
                        w_pad = w_flat
                    w_blocks = w_pad.view(-1, b)
                    block_zero_ratio = (w_blocks.sum(dim=1) == b).float().mean().item()
                    self.block_stats.append({
                        'layer': name,
                        'threshold': t,
                        'block_size': b,
                        'block_zero_ratio': block_zero_ratio
                    })

    def _act_hook_fn(self, name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            mag = self._get_magnitude(out.detach())
            stats = self.act_stats.setdefault(name, {'sum_sparsity': {t: 0.0 for t in ACT_THRESHOLDS}, 'count': 0, 'prelu_zeros': 0.0})
            
            for t in ACT_THRESHOLDS:
                stats['sum_sparsity'][t] += (mag < t).float().mean().item()
            
            if 'prelu' in name.lower() or isinstance(module, (nn.PReLU, torch_complex.ComplexPReLU, nn.ReLU)):
                # exact zeros
                stats['prelu_zeros'] += (mag == 0).float().mean().item()

            stats['count'] += 1

            # channel wise consistently near zero across batch
            if len(mag.shape) >= 2:
                # Shape is likely [batch, channel, ...] or [batch, time, channel]
                # Use isinstance to handle FP8-wrapped ParametrizedConv2d etc.
                is_conv = isinstance(module, (nn.Conv2d, nn.Conv1d, torch_complex.ComplexConv2d,
                                              torch_complex.ComplexDepthwiseSeparableConv2d))
                if is_conv:
                    ch_mag = mag.mean(dim=[0, 2, 3]) if len(mag.shape) == 4 else mag.mean(dim=[0, 2])
                else: # linear / prelu
                    ch_mag = mag.mean(dim=0) if len(mag.shape) == 2 else mag.mean(dim=(0, 1))
                
                if 'ch_mag_sum' not in stats:
                    stats['ch_mag_sum'] = torch.zeros_like(ch_mag)
                stats['ch_mag_sum'] += ch_mag
                
        return hook

    def _attn_hook_fn(self, name):
        def hook(module, input, output):
            # nn.MultiheadAttention returns (attn_output, attn_weights)
            # attn_weights shape: (batch, tgt_len, src_len) — averaged over heads by default
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn_weights = output[1].detach()
                self.attn_scores.append(attn_weights.cpu())
        return hook

    def analyze_activations_and_attention(self):
        print("Analyzing activations and attention...")
        # Register hooks
        for name, module in self.model.named_modules():
            # Use isinstance to catch FP8-wrapped variants (ParametrizedConv2d, ParametrizedLinear, etc.)
            if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.Linear, torch_complex.ComplexConv2d,
                                   torch_complex.ComplexDepthwiseSeparableConv2d,
                                   torch_complex.ComplexPReLU, nn.PReLU)):
                h = module.register_forward_hook(self._act_hook_fn(name))
                self.act_hooks.append(h)

            # Hook inner nn.MultiheadAttention (FP8-wrapped as ParametrizedMultiheadAttention)
            # ComplexMultiheadAttention discards weights, so hook the inner real/imag MHA
            if isinstance(module, nn.MultiheadAttention):
                h = module.register_forward_hook(self._attn_hook_fn(name))
                self.attn_hooks.append(h)

        # Load samples
        wav_files = glob.glob(os.path.join(self.test_dir, '**', '*.wav'), recursive=True)
        if not wav_files:
            print(f"Warning: No valid WAV files found in {self.test_dir}. Skipping inference.")
        samples_to_run = wav_files[:self.num_samples] if getattr(self, 'num_samples', None) else wav_files
        
        self.model.eval()
        for idx, wf in enumerate(tqdm(samples_to_run, desc="Running Inference")):
            data, sr = sf.read(wf)
            if len(data.shape) > 1 and data.shape[1] >= 2:
                pass # it is binaural
            else:
                data = np.stack([data, data], axis=1) # fake binaural
            
            duration = len(data) / sr
            # Use ~2 sec chunk
            max_len = int(sr * 2.0)
            if len(data) > max_len:
                data = data[:max_len]
            input_tensor = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                self.model(input_tensor)

        # Remove hooks
        for h in self.act_hooks + self.attn_hooks:
            h.remove()

    def classify_and_prune(self):
        # Weight based classification
        # Aggregate by layer
        layer_summary = {}
        for w in self.weight_stats:
            name = w['layer']
            t = w['threshold']
            if name not in layer_summary:
                layer_summary[name] = {}
            if t == PRACTICAL_THRESHOLD:  # use 5e-2 as practical threshold
                layer_summary[name]['total'] = w['total_sparsity']
                layer_summary[name]['channel'] = w['channel_sparsity']

        for b in self.block_stats:
            name = b['layer']
            t = b['threshold']
            bs = b['block_size']
            if t == PRACTICAL_THRESHOLD:
                if 'blocks' not in layer_summary[name]:
                    layer_summary[name]['blocks'] = {}
                layer_summary[name]['blocks'][bs] = b['block_zero_ratio']

        for name, summ in layer_summary.items():
            tot = summ.get('total', 0)
            ch = summ.get('channel', 0)
            if tot < 0.1:
                cls_ = "No exploitable sparsity (dense)"
            elif ch > 0:
                cls_ = f"Channel-prunable: {ch*100:.1f}% channels removable"
                # If N channels removable, suggest for pruning
                mod = dict(self.model.named_modules())[name]
                w_tensor = self._get_weight_tensor(mod)
                out_c = w_tensor.shape[0]
                num_prune = int(ch * out_c)
                if num_prune > 0:
                    self.pruning_candidates.append(f"{name} (Channels): {num_prune} / {out_c} removable at {PRACTICAL_THRESHOLD}")
            else:
                # check blocks
                best_block = 0
                best_block_val = 0
                for bs, v in summ.get('blocks', {}).items():
                    if v > best_block_val:
                        best_block_val = v
                        best_block = bs
                
                if best_block_val > 0.3:
                    cls_ = f"Block-sparse at block={best_block}, threshold={PRACTICAL_THRESHOLD} (ratio={best_block_val:.2f})"
                else:
                    cls_ = f"Random sparse (ratio={tot:.2f}) - no structure"
            
            self.classes[name] = cls_
            
        # Attention pruning (Heads) — check Q/K/V output projections per head
        actual_model = getattr(self.model, 'model', self.model)
        mattn_block = getattr(actual_model, 'mattn', None)
        if mattn_block is not None and hasattr(mattn_block, 'output_proj'):
            out_proj_mod = mattn_block.output_proj
            try:
                out_proj = self._get_weight_tensor(out_proj_mod)
                if isinstance(out_proj, torch.Tensor):
                    out_proj = out_proj.detach()
                    # For complex linear: weight has shape [out_features, in_features] (real part)
                    # Check both real and imag if complex
                    if out_proj.is_complex():
                        mag = torch.abs(out_proj)
                    else:
                        mag = torch.abs(out_proj)
                    mask = mag < PRACTICAL_THRESHOLD
                    in_dim = mask.shape[1]
                    num_heads = getattr(getattr(mattn_block, 'mattn', mattn_block), 'num_heads', 16)
                    head_dim = in_dim // num_heads
                    head_prunable = 0
                    for h in range(num_heads):
                        head_cols = mask[:, h*head_dim:(h+1)*head_dim]
                        if head_cols.all():
                            head_prunable += 1
                    if head_prunable > 0:
                        self.pruning_candidates.append(
                            f"mattn.output_proj (Heads): {head_prunable} / {num_heads} removable at {PRACTICAL_THRESHOLD}")
            except Exception as e:
                print(f"Warning: Could not analyze attention heads: {e}")

    def generate_attention_stats(self, results_dir):
        if not self.attn_scores:
            return

        os.makedirs(results_dir, exist_ok=True)
        print("Saving attention analysis...")
        try:
            agg_scores = [score.mean(dim=(0, 1)) if score.ndim == 4 else score.mean(dim=0) for score in self.attn_scores]
            if len(agg_scores) > 1:
                avg_heatmap = torch.stack(agg_scores).mean(dim=0).numpy()
            else:
                avg_heatmap = agg_scores[0].numpy()
                
            plt.figure(figsize=(10, 8))
            plt.imshow(avg_heatmap, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title('Average Attention Pattern')
            plt.xlabel('Source Pos')
            plt.ylabel('Target Pos')
            plt.savefig(os.path.join(results_dir, 'attention_heatmap.png'))
            plt.close()
            
            N = avg_heatmap.shape[0]
            i_idx, j_idx = np.indices((N, N))
            in_window = (i_idx >= j_idx) & ((i_idx - j_idx) <= 31)
            
            window_scores = avg_heatmap[in_window]
            near_zero_ratio = (window_scores < 0.01).mean()
            
            pd.DataFrame([{
                'layer': 'attention',
                'window_size': 31,
                'in_window_elements': in_window.sum(),
                'fraction_lt_0.01': near_zero_ratio
            }]).to_csv(os.path.join(results_dir, 'attention_sparsity.csv'), index=False)
            
        except Exception as e:
            print(f"Error processing attention scores: {e}")

    def save_results(self, results_dir):
        os.makedirs(results_dir, exist_ok=True)
        # Weights
        pd.DataFrame(self.weight_stats).to_csv(os.path.join(results_dir, 'weight_sparsity.csv'), index=False)
        pd.DataFrame(self.block_stats).to_csv(os.path.join(results_dir, 'block_sparsity.csv'), index=False)
        
        # Plot weight sparsity (at PRACTICAL_THRESHOLD)
        w_df = pd.DataFrame([w for w in self.weight_stats if w['threshold'] == PRACTICAL_THRESHOLD])
        if not w_df.empty:
            plt.figure(figsize=(12, 6))
            plt.bar(w_df['layer'], w_df['total_sparsity'])
            plt.xticks(rotation=90)
            plt.ylabel(f'Total sparsity (T={PRACTICAL_THRESHOLD})')
            plt.title('Per-Layer Sparsity')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'weight_sparsity_plot.png'))
            plt.close()

        # Acts
        act_rows = []
        for name, stats in self.act_stats.items():
            cnt = max(stats['count'], 1)
            row = {'layer': name}
            for t in ACT_THRESHOLDS:
                row[f'sparsity_{t}'] = stats['sum_sparsity'][t] / cnt
            if 'prelu' in name.lower() or 'relu' in name.lower():
                row['prelu_exact_zeros'] = stats['prelu_zeros'] / cnt
                
            ch_mag_avg = stats.get('ch_mag_sum', 0.0) / cnt
            if isinstance(ch_mag_avg, torch.Tensor):
                row['consistent_dead_channels'] = (ch_mag_avg < 0.01).float().mean().item()
            act_rows.append(row)
            
        pd.DataFrame(act_rows).to_csv(os.path.join(results_dir, 'activation_sparsity.csv'), index=False)
        
        # JSON and TXT
        with open(os.path.join(results_dir, 'sparsity_summary.json'), 'w') as f:
            json.dump(self.classes, f, indent=2)
            
        with open(os.path.join(results_dir, 'pruning_candidates.txt'), 'w') as f:
            for c in self.pruning_candidates:
                f.write(c + "\n")
                
        # Markdown summary
        md_content = f"# Sparsity Analysis Summary\n\n## Findings\n"
        md_content += f"- **Attention**: Saved heatmap to `attention_heatmap.png`.\n"
        md_content += f"- **Dead Channels**: Found {len(self.pruning_candidates)} layers with prunable channels/heads.\n\n"
        md_content += f"## Pruning Candidates:\n"
        for c in self.pruning_candidates:
            md_content += f"- {c}\n"
        
        with open(os.path.join(results_dir, 'summary.md'), 'w') as f:
            f.write(md_content)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_dir', required=True)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init model using relative caller module path (works seamlessly from any cwd)
    GlobalHydra.instance().clear()
    initialize(config_path="config", version_base=None)
    config = compose(config_name="config")
    model = DCNNLightningModule(config)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    
    analyzer = BCCTNSparsityAnalyzer(model, args.test_dir, args.num_samples)
    analyzer.analyze_weights()
    analyzer.analyze_activations_and_attention()
    analyzer.classify_and_prune()
    
    analyzer.generate_attention_stats(args.results_dir)
    analyzer.save_results(args.results_dir)
    print(f"Done. Outputs saved to {args.results_dir}")

if __name__ == "__main__":
    main()
