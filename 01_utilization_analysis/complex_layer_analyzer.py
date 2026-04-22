# complex_layer_analyzer_fixed.py
"""A cleaned-up, syntax-error-free re-implementation of the original
`complex_layer_analyzer.py`.  The public API stays identical, so it can be
imported by existing scripts that expect the old module.

Key improvements
----------------
* **Indentation / syntax** - removed duplicate blocks, fixed mis-indents and
  incomplete statements.
* **Missing imports** - added `pandas` and `plotly.graph_objects as go`.
* **Robust helpers** - factored repetitive logic into small private methods.
* **Type hints & doc-strings** - improve readability.
* **No side-effects on import** - all heavy work is shielded behind
  functions / `if __name__ == "__main__"`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn

__all__ = [
    "ComplexLayerAnalyzer",
    "generate_pruning_report",
    "create_visualization_report",
    "run_complete_analysis",
]


# -----------------------------------------------------------------------------
#  Core analyser class
# -----------------------------------------------------------------------------
class ComplexLayerAnalyzer:
    """Compute channel- / block- / phase-sparsity statistics for complex convs."""

    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------
    def analyze_complex_conv(
        self,
        layer_name: str,
        conv_r: nn.Module,
        conv_i: nn.Module,
    ) -> Dict[str, Any]:
        """Analyse a Complex*Conv2d layer implemented as real/imag pairs."""
        wr = conv_r.weight.data
        wi = conv_i.weight.data
        mag = torch.sqrt(wr ** 2 + wi ** 2)
        phase = torch.atan2(wi, wr)

        return {
            "layer_name": layer_name,
            "shape": list(wr.shape),
            "real_part": self._tensor_sparsity(wr),
            "imag_part": self._tensor_sparsity(wi),
            "magnitude": self._tensor_sparsity(mag),
            "phase_distribution": self._phase_distribution(phase, mag),
            "structured_sparsity": self._structured_patterns(wr, wi, mag),
        }

    # ------------------------------------------------------------------
    #  Low-level utilities
    # ------------------------------------------------------------------
    def _tensor_sparsity(self, tensor: torch.Tensor) -> Dict[str, Any]:
        mask = tensor.abs() < self.threshold
        sparsity = mask.float().mean().item()

        if tensor.ndim == 4:  # Conv filter [out, in, H, W]
            o, i, h, w = tensor.shape
            dim_stats = {
                "out_channel": mask.reshape(o, -1).float().mean(1).cpu().numpy(),
                "in_channel": mask.permute(1, 0, 2, 3).reshape(i, -1).float().mean(1).cpu().numpy(),
                "spatial": mask.reshape(-1, h * w).float().mean(0).cpu().numpy(),
            }
        else:  # Linear weight [out, in]
            dim_stats = {
                "row": mask.float().mean(1).cpu().numpy(),
                "col": mask.float().mean(0).cpu().numpy(),
            }

        nz = (~mask).nonzero(as_tuple=False)
        mean, std, vmax, vmin = 0.0, 0.0, 0.0, 0.0
        if nz.numel():
            vals = tensor[~mask]
            mean, std = vals.mean().item(), vals.std().item()
            vmax, vmin = vals.abs().max().item(), vals.abs().min().item()

        return {
            "total_sparsity": sparsity,
            "num_zeros": mask.sum().item(),
            "num_elements": tensor.numel(),
            "dim_sparsities": dim_stats,
            "value_stats": {
                "mean": mean,
                "std": std,
                "max": vmax,
                "min": vmin,
            },
        }

    def _phase_distribution(
        self, phase: torch.Tensor, magnitude: torch.Tensor
    ) -> Dict[str, Any]:
        valid = magnitude > self.threshold
        if not valid.any():
            return {"valid_count": 0}

        phases = phase[valid].cpu().numpy()
        hist, bins = np.histogram(phases, bins=36, range=(-np.pi, np.pi))
        return {
            "valid_count": int(valid.sum()),
            "phase_histogram": hist.tolist(),
            "phase_bins": bins.tolist(),
            "dominant_phase": float(bins[hist.argmax()]),
            "phase_std": float(phases.std()),
        }

    def _structured_patterns(
        self, wr: torch.Tensor, wi: torch.Tensor, mag: torch.Tensor
    ) -> Dict[str, Any]:
        if mag.ndim != 4:
            return {}

        out_c, *_ = mag.shape
        ch_sparsity = (mag < self.threshold).reshape(out_c, -1).float().mean(1)
        fully_sparse = (ch_sparsity > 0.99).sum().item()

        return {
            "fully_sparse_output_channels": fully_sparse,
            "fully_sparse_ratio": fully_sparse / out_c,
            "block_2x2_sparsity": self._block_sparsity(mag, 2),
            "real_imag_correlation": self._real_imag_corr(wr, wi),
        }

    # ------------------------------------------------------------------
    #  Helper maths
    # ------------------------------------------------------------------
    @staticmethod
    def _block_sparsity(tensor: torch.Tensor, block: int = 2) -> float:
        if tensor.ndim != 4:
            return 0.0
        _, _, h, w = tensor.shape
        hb, wb = h // block, w // block
        if hb == 0 or wb == 0:
            return 0.0
        view = tensor[:, :, : hb * block, : wb * block]
        view = view.reshape(*tensor.shape[:2], hb, block, wb, block)
        view = view.permute(0, 1, 2, 4, 3, 5).reshape(-1, block * block)
        blk_zero = (view.abs() < 1e-6).all(dim=1).float().mean().item()
        return blk_zero

    @staticmethod
    def _real_imag_corr(wr: torch.Tensor, wi: torch.Tensor) -> float:
        wr_f = wr.flatten().float()
        wi_f = wi.flatten().float()
        if wr_f.std() == 0 or wi_f.std() == 0:
            return 0.0
        return float(np.corrcoef(wr_f.numpy(), wi_f.numpy())[0, 1])


# -----------------------------------------------------------------------------
#  Stand-alone helpers (kept for backward-compatibility with original script)
# -----------------------------------------------------------------------------

def generate_pruning_report(
    standard_results: Dict[str, Any],
    complex_results: Dict[str, Any],
    save_dir: str | Path = "./sparsity_results",
) -> pd.DataFrame:
    """Create a CSV with channel / head pruning suggestions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    rows: List[Dict[str, Any]] = []

    # 1) Depthwise / Pointwise conv suggestions
    for layer_type in ("depthwise", "pointwise"):
        for lname, stats in standard_results.get(layer_type, {}).items():
            if "out_channel_sparsity" not in stats:
                continue
            spars = np.asarray(stats["out_channel_sparsity"]["values"])
            hi = np.where(spars > 0.8)[0]
            mid = np.where((spars > 0.5) & (spars <= 0.8))[0]
            rows.append(
                {
                    "Layer": lname,
                    "Type": layer_type,
                    "Total Channels": len(spars),
                    "High Sparsity (>80%)": len(hi),
                    "Medium Sparsity (50-80%)": len(mid),
                    "Suggested Pruning": len(hi),
                    "Pruning Ratio": f"{len(hi) / len(spars) * 100:.1f}%",
                }
            )

    # 2) Complex conv suggestions
    for lname, cstats in complex_results.items():
        s = cstats.get("structured_sparsity", {})
        if not s:
            continue
        rows.append(
            {
                "Layer": lname,
                "Type": "Complex",
                "Total Channels": cstats["shape"][0],
                "High Sparsity (>80%)": s.get("fully_sparse_output_channels", 0),
                "Medium Sparsity (50-80%)": 0,
                "Suggested Pruning": s.get("fully_sparse_output_channels", 0),
                "Pruning Ratio": f"{s.get('fully_sparse_ratio', 0) * 100:.1f}%",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "pruning_recommendations.csv", index=False)
    return df


def create_visualization_report(
    analyzer_results: Dict[str, Any], save_path: str | Path = "./sparsity_results/report.html"
) -> None:
    """Quick HTML report with phase histograms (Plotly)."""
    html_parts: List[str] = [
        """<html><head><meta charset='utf-8'><title>Complex Layer Sparsity Report</title></head><body>""",
        "<h1>Complex Layer Sparsity Analysis</h1>",
    ]

    for lname, ldata in analyzer_results.items():
        if not isinstance(ldata, dict):
            continue
        html_parts.append(f"<h2>{lname}</h2>")
        html_parts.append("<ul>")
        for k in ("real_part", "imag_part", "magnitude"):
            html_parts.append(
                f"<li>{k} sparsity: {ldata.get(k, {}).get('total_sparsity', 0):.4f}</li>"
            )
        html_parts.append("</ul>")

        phase_info = ldata.get("phase_distribution", {})
        if phase_info.get("valid_count", 0):
            fig = go.Figure(
                [
                    go.Bar(
                        x=phase_info["phase_bins"][:-1],
                        y=phase_info["phase_histogram"],
                    )
                ]
            )
            fig.update_layout(title=f"Phase Distribution – {lname}")
            html_parts.append(fig.to_html(include_plotlyjs="cdn", full_html=False))

    html_parts.append("</body></html>")
    Path(save_path).write_text("\n".join(html_parts), encoding="utf-8")


# -----------------------------------------------------------------------------
#  End-to-end runner (kept lightweight – heavy deps like Hydra removed)
# -----------------------------------------------------------------------------

def run_complete_analysis(checkpoint_path: str) -> None:
    """Entry point that reproduces the behaviour of the original script."""
    import importlib
    import sys

    sys.path.append(".")
    
    try:
        AxisSparsityAnalyzer = importlib.import_module("axis_sparsity_analyzer").AxisSparsityAnalyzer
    except ImportError:
        print("Warning: axis_sparsity_analyzer not found, skipping axis analysis")
        AxisSparsityAnalyzer = None
    
    try:
        from DCNN.trainer import DCNNLightningModule
    except ImportError as e:
        print(f"Error importing DCNNLightningModule: {e}")
        return

    # --- load model -----------------------------------------------------
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    config = ckpt.get("hyper_parameters", {})
    
    if not config:
        print("Warning: No hyper_parameters found in checkpoint, using default config")
        # You might need to provide a default config here
        config = {}
    
    try:
        model = DCNNLightningModule(config)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- run axis-wise analysis ----------------------------------------
    if AxisSparsityAnalyzer:
        try:
            axis_analyzer = AxisSparsityAnalyzer(model)
            axis_results = axis_analyzer.analyze()
            axis_analyzer.plot_histograms()
            axis_analyzer.export_results()
        except Exception as e:
            print(f"Error in axis analysis: {e}")
            axis_results = {}
    else:
        axis_results = {}

    # --- run complex-layer analysis ------------------------------------
    c_analyzer = ComplexLayerAnalyzer()
    comp_results: Dict[str, Any] = {}
    
    print("Analyzing complex layers...")
    for name, mod in model.named_modules():
        # Look for complex conv layers in your model architecture
        if hasattr(mod, "conv_r") and hasattr(mod, "conv_i"):
            comp_results[f"{name}_conv"] = c_analyzer.analyze_complex_conv(
                f"{name}_conv", mod.conv_r, mod.conv_i
            )
        # Also check for depthwise/pointwise if they exist
        if hasattr(mod, "depthwise_r") and hasattr(mod, "depthwise_i"):
            comp_results[f"{name}_depthwise"] = c_analyzer.analyze_complex_conv(
                f"{name}_depthwise", mod.depthwise_r, mod.depthwise_i
            )
        if hasattr(mod, "pointwise_r") and hasattr(mod, "pointwise_i"):
            comp_results[f"{name}_pointwise"] = c_analyzer.analyze_complex_conv(
                f"{name}_pointwise", mod.pointwise_r, mod.pointwise_i
            )

    if not comp_results:
        print("Warning: No complex layers found in the model")
        # Let's also check for the actual complex layers in your model
        print("Available modules in model:")
        for name, mod in model.named_modules():
            if "complex" in name.lower() or hasattr(mod, "weight"):
                print(f"  {name}: {type(mod)}")

    # --- reports --------------------------------------------------------
    save_dir = Path("./sparsity_results")
    save_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        create_visualization_report(comp_results, save_dir / "complex_report.html")
        generate_pruning_report(axis_results, comp_results, save_dir)
        print(f"\n✔ Analysis finished – reports saved to {save_dir}\n")
    except Exception as e:
        print(f"Error generating reports: {e}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Complex Layer Sparsity Analysis")
    p.add_argument("--checkpoint", required=True, help="Path to lightning ckpt")
    args = p.parse_args()
    run_complete_analysis(args.checkpoint)