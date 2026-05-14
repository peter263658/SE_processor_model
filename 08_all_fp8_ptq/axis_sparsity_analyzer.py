# axis_sparsity_analyzer.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class AxisSparsityAnalyzer:
    """分析模型中 Depthwise/Pointwise Conv 和 Attention 層的軸向稀疏性"""
    
    def __init__(self, model: nn.Module, threshold: float = 1e-6):
        """
        Args:
            model: PyTorch 模型
            threshold: 判定為 0 的閾值
        """
        self.model = model
        self.threshold = threshold
        self.results = {
            'depthwise': {},
            'pointwise': {},
            'attention': {},
            'summary': {}
        }
        
    def analyze(self) -> Dict[str, Any]:
        """執行完整分析"""
        print("開始分析模型稀疏性...")
        
        # 1. 定位並分析各類層
        self._analyze_separable_conv()
        self._analyze_attention()
        
        # 2. 計算總體統計
        self._compute_summary_stats()
        
        return self.results
    
    def _analyze_separable_conv(self):
        """分析 Separable Convolution 層"""
        print("\n分析 Separable Convolution 層...")
        
        for name, module in self.model.named_modules():
            # Depthwise Conv (groups = in_channels)
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(module, 'groups') and module.groups > 1:
                    if 'depthwise' in name:
                        self._analyze_conv_layer(name, module, 'depthwise')
                elif 'pointwise' in name:
                    self._analyze_conv_layer(name, module, 'pointwise')
                    
    def _analyze_conv_layer(self, name: str, layer: nn.Module, conv_type: str):
        """分析單個卷積層的軸向稀疏性"""
        weight = layer.weight.data
        
        # 對於 complex 層，分析實部和虛部
        if weight.dtype == torch.complex64:
            weight_real = weight.real
            weight_imag = weight.imag
            
            # 分別分析實部和虛部
            stats_real = self._compute_conv_sparsity(weight_real, f"{name}_real")
            stats_imag = self._compute_conv_sparsity(weight_imag, f"{name}_imag")
            
            # 合併統計
            stats = {
                'real': stats_real,
                'imag': stats_imag,
                'combined': self._compute_conv_sparsity(torch.abs(weight), f"{name}_abs")
            }
        else:
            stats = self._compute_conv_sparsity(weight, name)
            
        self.results[conv_type][name] = stats
        
    def _compute_conv_sparsity(self, weight: torch.Tensor, layer_name: str) -> Dict:
        """計算卷積權重的軸向稀疏性"""
        # weight shape: [out_channels, in_channels, height, width]
        out_c, in_c, h, w = weight.shape
        
        # 計算各軸稀疏性
        sparse_mask = torch.abs(weight) < self.threshold
        
        # Channel-wise sparsity (output channels)
        out_channel_sparsity = sparse_mask.reshape(out_c, -1).float().mean(dim=1)
        
        # Channel-wise sparsity (input channels)
        in_channel_sparsity = sparse_mask.permute(1, 0, 2, 3).reshape(in_c, -1).float().mean(dim=1)
        
        # Spatial sparsity (H and W)
        spatial_sparsity_h = sparse_mask.permute(2, 0, 1, 3).reshape(h, -1).float().mean(dim=1)
        spatial_sparsity_w = sparse_mask.permute(3, 0, 1, 2).reshape(w, -1).float().mean(dim=1)
        
        # 總體稀疏性
        total_sparsity = sparse_mask.float().mean().item()
        
        return {
            'shape': list(weight.shape),
            'total_sparsity': total_sparsity,
            'out_channel_sparsity': {
                'values': out_channel_sparsity.cpu().numpy(),
                'mean': out_channel_sparsity.mean().item(),
                'std': out_channel_sparsity.std().item(),
                'min': out_channel_sparsity.min().item(),
                'max': out_channel_sparsity.max().item()
            },
            'in_channel_sparsity': {
                'values': in_channel_sparsity.cpu().numpy(),
                'mean': in_channel_sparsity.mean().item(),
                'std': in_channel_sparsity.std().item(),
                'min': in_channel_sparsity.min().item(),
                'max': in_channel_sparsity.max().item()
            },
            'spatial_h_sparsity': {
                'values': spatial_sparsity_h.cpu().numpy(),
                'mean': spatial_sparsity_h.mean().item(),
                'std': spatial_sparsity_h.std().item()
            },
            'spatial_w_sparsity': {
                'values': spatial_sparsity_w.cpu().numpy(),
                'mean': spatial_sparsity_w.mean().item(),
                'std': spatial_sparsity_w.std().item()
            }
        }
    
    def _analyze_attention(self):
        """分析 Multi-head Attention 層"""
        print("\n分析 Attention 層...")
        
        for name, module in self.model.named_modules():
            # 尋找 MultiheadAttention 或自定義的 attention 模塊
            if 'MultiheadAttention' in type(module).__name__:
                self._analyze_attention_layer(name, module)
            elif hasattr(module, 'in_proj_weight') or hasattr(module, 'out_proj'):
                # 處理自定義 attention 實現
                self._analyze_attention_layer(name, module)
                
    def _analyze_attention_layer(self, name: str, layer: nn.Module):
        """分析單個 Attention 層的稀疏性"""
        stats = {}
        
        # 分析 Q/K/V projection
        if hasattr(layer, 'in_proj_weight') and layer.in_proj_weight is not None:
            weight = layer.in_proj_weight
            embed_dim = layer.embed_dim
            
            # 分割 Q, K, V
            q_weight = weight[:embed_dim]
            k_weight = weight[embed_dim:2*embed_dim]
            v_weight = weight[2*embed_dim:]
            
            stats['q_proj'] = self._compute_linear_sparsity(q_weight, f"{name}_q")
            stats['k_proj'] = self._compute_linear_sparsity(k_weight, f"{name}_k")
            stats['v_proj'] = self._compute_linear_sparsity(v_weight, f"{name}_v")
            
        # 分析 output projection
        if hasattr(layer, 'out_proj') and hasattr(layer.out_proj, 'weight'):
            stats['out_proj'] = self._compute_linear_sparsity(
                layer.out_proj.weight, f"{name}_out"
            )
            
        # 分析 head-wise 稀疏性
        if hasattr(layer, 'num_heads'):
            stats['num_heads'] = layer.num_heads
            stats['head_dim'] = layer.embed_dim // layer.num_heads
            
        self.results['attention'][name] = stats
        
    def _compute_linear_sparsity(self, weight: torch.Tensor, layer_name: str) -> Dict:
        """計算線性層權重的稀疏性"""
        # weight shape: [out_features, in_features]
        sparse_mask = torch.abs(weight) < self.threshold
        
        # Row-wise sparsity
        row_sparsity = sparse_mask.float().mean(dim=1)
        
        # Column-wise sparsity
        col_sparsity = sparse_mask.float().mean(dim=0)
        
        # Total sparsity
        total_sparsity = sparse_mask.float().mean().item()
        
        return {
            'shape': list(weight.shape),
            'total_sparsity': total_sparsity,
            'row_sparsity': {
                'values': row_sparsity.cpu().numpy(),
                'mean': row_sparsity.mean().item(),
                'std': row_sparsity.std().item(),
                'min': row_sparsity.min().item(),
                'max': row_sparsity.max().item()
            },
            'col_sparsity': {
                'values': col_sparsity.cpu().numpy(),
                'mean': col_sparsity.mean().item(),
                'std': col_sparsity.std().item(),
                'min': col_sparsity.min().item(),
                'max': col_sparsity.max().item()
            }
        }
    
    def _compute_summary_stats(self):
        """計算總體統計數據"""
        summary = {
            'depthwise_layers': len(self.results['depthwise']),
            'pointwise_layers': len(self.results['pointwise']),
            'attention_layers': len(self.results['attention'])
        }
        
        # 計算各類層的平均稀疏性
        for layer_type in ['depthwise', 'pointwise']:
            if self.results[layer_type]:
                sparsities = []
                for layer_stats in self.results[layer_type].values():
                    if isinstance(layer_stats, dict) and 'total_sparsity' in layer_stats:
                        sparsities.append(layer_stats['total_sparsity'])
                    elif 'real' in layer_stats:  # Complex layer
                        sparsities.append(layer_stats['combined']['total_sparsity'])
                        
                if sparsities:
                    summary[f'{layer_type}_avg_sparsity'] = np.mean(sparsities)
                    summary[f'{layer_type}_std_sparsity'] = np.std(sparsities)
                    
        self.results['summary'] = summary
        
    def plot_histograms(self, save_dir: str = './sparsity_plots'):
        """繪製稀疏性直方圖"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 設置繪圖風格
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. 繪製 Depthwise Conv 稀疏性
        self._plot_conv_histograms(self.results['depthwise'], 
                                   save_path / 'depthwise_sparsity.png',
                                   'Depthwise Convolution Sparsity')
        
        # 2. 繪製 Pointwise Conv 稀疏性
        self._plot_conv_histograms(self.results['pointwise'],
                                   save_path / 'pointwise_sparsity.png', 
                                   'Pointwise Convolution Sparsity')
        
        # 3. 繪製 Attention 稀疏性
        self._plot_attention_histograms(self.results['attention'],
                                        save_path / 'attention_sparsity.png')
        
        # 4. 繪製總體比較圖
        self._plot_summary_comparison(save_path / 'summary_comparison.png')
        
    def _plot_conv_histograms(self, layer_stats: Dict, save_path: Path, title: str):
        """繪製卷積層稀疏性直方圖"""
        if not layer_stats:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(title, fontsize=16)
        
        # 收集所有層的稀疏性數據
        out_channel_sparsities = []
        in_channel_sparsities = []
        spatial_h_sparsities = []
        spatial_w_sparsities = []
        
        for name, stats in layer_stats.items():
            if isinstance(stats, dict) and 'out_channel_sparsity' in stats:
                out_channel_sparsities.extend(stats['out_channel_sparsity']['values'])
                in_channel_sparsities.extend(stats['in_channel_sparsity']['values'])
                spatial_h_sparsities.extend(stats['spatial_h_sparsity']['values'])
                spatial_w_sparsities.extend(stats['spatial_w_sparsity']['values'])
            elif 'combined' in stats:  # Complex layer
                out_channel_sparsities.extend(stats['combined']['out_channel_sparsity']['values'])
                in_channel_sparsities.extend(stats['combined']['in_channel_sparsity']['values'])
                
        # 繪製直方圖
        axes[0, 0].hist(out_channel_sparsities, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Output Channel Sparsity')
        axes[0, 0].set_xlabel('Sparsity')
        axes[0, 0].set_ylabel('Count')
        
        axes[0, 1].hist(in_channel_sparsities, bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Input Channel Sparsity')
        axes[0, 1].set_xlabel('Sparsity')
        axes[0, 1].set_ylabel('Count')
        
        axes[1, 0].hist(spatial_h_sparsities, bins=30, alpha=0.7, color='red')
        axes[1, 0].set_title('Spatial H Sparsity')
        axes[1, 0].set_xlabel('Sparsity')
        axes[1, 0].set_ylabel('Count')
        
        axes[1, 1].hist(spatial_w_sparsities, bins=30, alpha=0.7, color='purple')
        axes[1, 1].set_title('Spatial W Sparsity')
        axes[1, 1].set_xlabel('Sparsity')
        axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_attention_histograms(self, layer_stats: Dict, save_path: Path):
        """繪製 Attention 層稀疏性直方圖"""
        if not layer_stats:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Attention Layer Sparsity', fontsize=16)
        
        # 收集數據
        q_row_sparsities = []
        k_row_sparsities = []
        v_row_sparsities = []
        out_row_sparsities = []
        
        for name, stats in layer_stats.items():
            if 'q_proj' in stats:
                q_row_sparsities.extend(stats['q_proj']['row_sparsity']['values'])
            if 'k_proj' in stats:
                k_row_sparsities.extend(stats['k_proj']['row_sparsity']['values'])
            if 'v_proj' in stats:
                v_row_sparsities.extend(stats['v_proj']['row_sparsity']['values'])
            if 'out_proj' in stats:
                out_row_sparsities.extend(stats['out_proj']['row_sparsity']['values'])
                
        # 繪製直方圖
        if q_row_sparsities:
            axes[0, 0].hist(q_row_sparsities, bins=50, alpha=0.7, color='blue')
            axes[0, 0].set_title('Q Projection Row Sparsity')
            
        if k_row_sparsities:
            axes[0, 1].hist(k_row_sparsities, bins=50, alpha=0.7, color='green')
            axes[0, 1].set_title('K Projection Row Sparsity')
            
        if v_row_sparsities:
            axes[1, 0].hist(v_row_sparsities, bins=50, alpha=0.7, color='red')
            axes[1, 0].set_title('V Projection Row Sparsity')
            
        if out_row_sparsities:
            axes[1, 1].hist(out_row_sparsities, bins=50, alpha=0.7, color='purple')
            axes[1, 1].set_title('Output Projection Row Sparsity')
            
        for ax in axes.flat:
            ax.set_xlabel('Sparsity')
            ax.set_ylabel('Count')
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_summary_comparison(self, save_path: Path):
        """繪製總體比較圖"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 層類型分佈
        layer_counts = [
            self.results['summary'].get('depthwise_layers', 0),
            self.results['summary'].get('pointwise_layers', 0),
            self.results['summary'].get('attention_layers', 0)
        ]
        layer_types = ['Depthwise', 'Pointwise', 'Attention']
        
        ax1.bar(layer_types, layer_counts, color=['blue', 'green', 'red'])
        ax1.set_title('Layer Type Distribution')
        ax1.set_ylabel('Number of Layers')
        
        # 平均稀疏性比較
        avg_sparsities = []
        labels = []
        
        for layer_type in ['depthwise', 'pointwise']:
            key = f'{layer_type}_avg_sparsity'
            if key in self.results['summary']:
                avg_sparsities.append(self.results['summary'][key])
                labels.append(layer_type.capitalize())
                
        if avg_sparsities:
            ax2.bar(labels, avg_sparsities, color=['blue', 'green'])
            ax2.set_title('Average Sparsity by Layer Type')
            ax2.set_ylabel('Average Sparsity')
            ax2.set_ylim(0, 1)
            
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def export_results(self, save_dir: str = './sparsity_results'):
        """導出分析結果"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 1. 導出詳細 JSON 報告
        with open(save_path / 'detailed_results.json', 'w') as f:
            # 轉換 numpy arrays 為 lists
            json_results = self._convert_numpy_to_list(self.results)
            json.dump(json_results, f, indent=2)
            
        # 2. 導出 CSV 摘要
        self._export_csv_summary(save_path / 'summary_stats.csv')
        
        # 3. 導出剪枝建議
        self._export_pruning_suggestions(save_path / 'pruning_suggestions.csv')
        
    def _convert_numpy_to_list(self, obj):
        """遞歸轉換 numpy arrays 為 lists"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj
            
    def _export_csv_summary(self, save_path: Path):
        """導出 CSV 格式的摘要統計"""
        rows = []
        
        # Depthwise layers
        for name, stats in self.results['depthwise'].items():
            if isinstance(stats, dict) and 'total_sparsity' in stats:
                rows.append({
                    'Layer': name,
                    'Type': 'Depthwise',
                    'Shape': str(stats['shape']),
                    'Total Sparsity': stats['total_sparsity'],
                    'Out Channel Sparsity (mean)': stats['out_channel_sparsity']['mean'],
                    'In Channel Sparsity (mean)': stats['in_channel_sparsity']['mean']
                })
                
        # Pointwise layers
        for name, stats in self.results['pointwise'].items():
            if isinstance(stats, dict) and 'total_sparsity' in stats:
                rows.append({
                    'Layer': name,
                    'Type': 'Pointwise',
                    'Shape': str(stats['shape']),
                    'Total Sparsity': stats['total_sparsity'],
                    'Out Channel Sparsity (mean)': stats['out_channel_sparsity']['mean'],
                    'In Channel Sparsity (mean)': stats['in_channel_sparsity']['mean']
                })
                
        # Attention layers
        for name, layer_stats in self.results['attention'].items():
            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                if proj_type in layer_stats:
                    stats = layer_stats[proj_type]
                    rows.append({
                        'Layer': f"{name}_{proj_type}",
                        'Type': 'Attention',
                        'Shape': str(stats['shape']),
                        'Total Sparsity': stats['total_sparsity'],
                        'Row Sparsity (mean)': stats['row_sparsity']['mean'],
                        'Col Sparsity (mean)': stats['col_sparsity']['mean']
                    })
                    
        df = pd.DataFrame(rows)
        df.to_csv(save_path, index=False)
        
    def _export_pruning_suggestions(self, save_path: Path):
        """導出結構化剪枝建議"""
        suggestions = []
        
        # 設定剪枝閾值
        channel_prune_threshold = 0.8  # 80% 稀疏的 channel 可以剪枝
        head_prune_threshold = 0.7     # 70% 稀疏的 attention head 可以剪枝
        
        # 分析 Conv layers
        for layer_type in ['depthwise', 'pointwise']:
            for name, stats in self.results[layer_type].items():
                if isinstance(stats, dict) and 'out_channel_sparsity' in stats:
                    # 找出可剪枝的 output channels
                    out_sparsity = stats['out_channel_sparsity']['values']
                    prunable_channels = np.where(out_sparsity > channel_prune_threshold)[0]
                    
                    if len(prunable_channels) > 0:
                        suggestions.append({
                            'Layer': name,
                            'Type': f'{layer_type} (out_channels)',
                            'Prunable Units': len(prunable_channels),
                            'Total Units': len(out_sparsity),
                            'Pruning Ratio': len(prunable_channels) / len(out_sparsity),
                            'Indices': prunable_channels.tolist()[:10]  # 只顯示前10個
                        })
                        
        # 分析 Attention heads
        for name, layer_stats in self.results['attention'].items():
            if 'num_heads' in layer_stats and 'out_proj' in layer_stats:
                num_heads = layer_stats['num_heads']
                head_dim = layer_stats['head_dim']
                
                # 分析每個 head 的稀疏性
                out_proj_stats = layer_stats['out_proj']
                row_sparsity = out_proj_stats['row_sparsity']['values']
                
                # 將 row sparsity 按 head 分組
                head_sparsities = []
                for h in range(num_heads):
                    start_idx = h * head_dim
                    end_idx = (h + 1) * head_dim
                    head_sparsity = np.mean(row_sparsity[start_idx:end_idx])
                    head_sparsities.append(head_sparsity)
                    
                # 找出可剪枝的 heads
                head_sparsities = np.array(head_sparsities)
                prunable_heads = np.where(head_sparsities > head_prune_threshold)[0]
                
                if len(prunable_heads) > 0:
                    suggestions.append({
                        'Layer': name,
                        'Type': 'Attention Heads',
                        'Prunable Units': len(prunable_heads),
                        'Total Units': num_heads,
                        'Pruning Ratio': len(prunable_heads) / num_heads,
                        'Indices': prunable_heads.tolist()
                    })
                    
        df = pd.DataFrame(suggestions)
        df.to_csv(save_path, index=False)
        
        return suggestions


def analyze_model_from_checkpoint(checkpoint_path: str, config_path: str = None):
    """從 checkpoint 加載模型並分析"""
    # 導入必要的模塊
    import sys
    sys.path.append('.')
    
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from DCNN.trainer import DCNNLightningModule
    
    # 初始化 Hydra
    GlobalHydra.instance().clear()
    initialize(config_path="config")
    config = compose(config_name="config")
    
    # 加載模型
    print(f"加載模型: {checkpoint_path}")
    model = DCNNLightningModule(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    # 執行分析
    analyzer = AxisSparsityAnalyzer(model)
    results = analyzer.analyze()
    
    # 繪製圖表
    analyzer.plot_histograms()
    
    # 導出結果
    analyzer.export_results()
    
    print("\n分析完成！結果已保存到:")
    print("- 圖表: ./sparsity_plots/")
    print("- 數據: ./sparsity_results/")
    
    return analyzer


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='分析模型稀疏性')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路徑')
    parser.add_argument('--threshold', type=float, default=1e-6,
                        help='稀疏性閾值')
    parser.add_argument('--output_dir', type=str, default='./sparsity_analysis',
                        help='輸出目錄')
    
    args = parser.parse_args()
    
    # 執行分析
    analyzer = analyze_model_from_checkpoint(args.checkpoint)
    
    # 打印摘要
    print("\n=== 稀疏性分析摘要 ===")
    summary = analyzer.results['summary']
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()