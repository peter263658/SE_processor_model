#!/usr/bin/env python3
"""
Post-Training Quantization (PTQ) Export Tool for BCCTN
Supports INT8 quantization for Conv/Linear layers and BF16 for Attention layers.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import importlib.util
import importlib

import torch
import torch.nn as nn
import numpy as np

from quant_utils import QuantizationManager, INT8Quantizer, BF16Quantizer
from dump_utils import DumpManager, write_tensor_hex, write_shape_file, write_json_file
from frame_runner import FrameRunner

from quant_utils import fake_quant_activation


# --- [NEW] complex layers support (best-effort import) ---
try:
    import complexLayers as cl
    COMPLEX_CONVS = (
        getattr(cl, "ComplexConv2d", None),
        getattr(cl, "ComplexDepthwiseSeparableConv2d", None),
        getattr(cl, "ComplexConvTranspose2d", None),
        getattr(cl, "ComplexDepthwiseSeparableConvTranspose2d", None),
    )
    COMPLEX_LINEAR = (getattr(cl, "ComplexLinear", None),)
    COMPLEX_BN = (
        getattr(cl, "ComplexBatchNorm1d", None),
        getattr(cl, "ComplexBatchNorm2d", None),
        getattr(cl, "NaiveComplexBatchNorm1d", None),
        getattr(cl, "NaiveComplexBatchNorm2d", None),
    )
    COMPLEX_ATTN = (getattr(cl, "ComplexMultiheadAttention", None),)
    # 濾掉 None
    COMPLEX_CONVS = tuple([c for c in COMPLEX_CONVS if c])
    COMPLEX_LINEAR = tuple([c for c in COMPLEX_LINEAR if c])
    COMPLEX_BN = tuple([c for c in COMPLEX_BN if c])
    COMPLEX_ATTN = tuple([c for c in COMPLEX_ATTN if c])
except Exception:
    COMPLEX_CONVS = COMPLEX_LINEAR = COMPLEX_BN = COMPLEX_ATTN = tuple()


# ====== [BEGIN] hardcoded absolute import for DCNN model ======
def _import_binaural_model_class():
    """
    寫死載入 DCNN.models.binaural_attention_model.BinauralAttentionDCNN
    讓 .model 的相對匯入在包內自然生效。
    """
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    PKG_ROOT     = os.path.join(PROJECT_ROOT, "DCNN")
    MODELS_DIR   = os.path.join(PKG_ROOT, "models")

    # 確保可以當成 package 被 import
    for p in (PKG_ROOT, MODELS_DIR):
        init_py = os.path.join(p, "__init__.py")
        if not os.path.exists(init_py):
            try:
                open(init_py, "a").close()
            except Exception:
                pass

    # 把專案根加進 sys.path（放在最前面避免被其他版本覆蓋）
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    module_name = "DCNN.models.binaural_attention_model"
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        # 提供更可讀的錯誤訊息與診斷建議
        raise RuntimeError(
            f"[ImportError] 無法 import {module_name}. "
            f"請確認目錄結構與 __init__.py 是否存在：\n"
            f"  - {PKG_ROOT}\n  - {MODELS_DIR}\n"
            f"原始錯誤：{type(e).__name__}: {e}"
        )
    if not hasattr(mod, "BinauralAttentionDCNN"):
        raise RuntimeError(
            f"[ImportError] 模組 {module_name} 找不到 'BinauralAttentionDCNN' 類別。"
        )
    return getattr(mod, "BinauralAttentionDCNN")
# ====== [END] hardcoded absolute import for DCNN model ======


class ActivationFakeQuantManager:
    """
    管理活化 fake-quant 的 forward pre-hook。
    """
    def __init__(self):
        self._handles = []  # type: list[torch.utils.hooks.RemovableHandle]

    def clear(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

    def attach(self, model: torch.nn.Module, act_scale_map: Dict[str, float], name_resolver) -> None:
        """
        act_scale_map: {layer_name: scale_float}
        name_resolver: fn(module) -> layer_name  用來把 module 轉回我們的命名
        """
        self.clear()

        def make_hook(layer_name: str, scale_val: float):
            scale_t = torch.tensor(scale_val, dtype=torch.float32)
            def _pre_hook(mod: torch.nn.Module, inputs: Tuple[torch.Tensor, ...]):
                if not inputs:
                    return
                x = inputs[0]
                # 僅對張量輸入做處理（避免 tuple 或其他型別）
                if torch.is_tensor(x):
                    # 將 scale 搬到正確的 device/dtype
                    s = scale_t.to(device=x.device, dtype=x.dtype)
                    x_dq = fake_quant_activation(x, s)
                    return (x_dq, )
                return inputs
            return _pre_hook

        # 掃描所有子模組，對「要做 INT8 的層」掛 pre-hook
        for module in model.modules():
            if not hasattr(module, "_ptq_int8_target"):
                continue
            if not getattr(module, "_ptq_int8_target"):
                continue
            layer_name = name_resolver(module)
            if layer_name in act_scale_map and act_scale_map[layer_name] is not None:
                h = module.register_forward_pre_hook(make_hook(layer_name, act_scale_map[layer_name]))
                self._handles.append(h)


class PTQExporter:
    """Main PTQ exporter class."""
    
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.quant_manager = QuantizationManager()
        self.dump_manager = DumpManager(args.output_root, args.run_id, args.dump_complex_split)
        self.layer_order = []
        self.layer_index = 0
        # 活化 fake-quant 管理器
        self.act_fakequant_mgr = ActivationFakeQuantManager()
        # 保存已載入的模型（_load_model 後會設）
        self.model: Optional[nn.Module] = None

    def _load_act_scales_from_disk(self) -> Dict[str, float]:
        """
        嘗試從 artifacts/<RUN_ID>/quant_params 讀回各層活化 scale。
        若你已在內存持有，可直接回傳內存版本。
        """
        import json, os
        qdir = self.dump_manager.output_dir / "quant_params"
        mp: Dict[str, float] = {}
        if not qdir.exists():
            return mp
        for fn in os.listdir(qdir):
            if not fn.endswith("_act.json"):
                continue
            with open(qdir / fn, "r") as f:
                obj = json.load(f)
            name = obj.get("name")
            scale = obj.get("scale")
            if isinstance(scale, (int, float)):
                mp[name] = float(scale)
        return mp

    def _module_name_resolver(self, module: torch.nn.Module) -> str:
        # 假設每個 module 先前在註冊時加過 attr: _ptq_layer_name
        return getattr(module, "_ptq_layer_name", None) or "unknown"

    def _mark_int8_targets(self, model: Optional[nn.Module]=None):
        mroot = model if model is not None else self.model
        if mroot is None:
            return
        for m in mroot.modules():
            # 依你的程式邏輯：Conv/Depthwise/Linear 等
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear) + COMPLEX_CONVS + COMPLEX_LINEAR + COMPLEX_BN):
                setattr(m, "_ptq_int8_target", True)
            else:
                setattr(m, "_ptq_int8_target", False)
        
    def _get_device(self) -> torch.device:
        """Get compute device."""
        if self.args.device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        elif self.args.device == 'cpu':
            return torch.device('cpu')
        else:
            # Auto detect
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_model(self):
        """載入模型（寫死版本，不透過 argparse）"""
        import torch

        # 1) 硬編模型 class（不透過 CLI 參數）
        model_class = _import_binaural_model_class()

        # 2) 讀權重檔
        weights_path = "/raid/R12K41024/BCCTN_hardware/model.ckpt"  # 寫死
        state = torch.load(weights_path, map_location="cpu")
        state_dict = state.get("state_dict", state)

        # 3) 清理前綴：支援 'module.' / 'model.' / 無前綴
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                nk = k[7:]
            elif k.startswith("model."):
                nk = k[6:]
            else:
                nk = k
            cleaned[nk] = v
        state_dict = cleaned

        # 4) 寫死模型建構參數（避免 argparse）
        #    這些值請依你的訓練版本做一致化（根據 config.yaml）
        model_kwargs = dict(
            # 從 config.yaml 看到的參數
            attention=True,
            binaural=True,
            use_depthwise=True,
            use_clstm=True,
            use_cbn=True,
            masking_mode='E',
            rnn_layers=2,
            rnn_units=128,
            win_len=128,
            win_inc=80,
            fft_len=128,
            win_type='hann',
            kernel_size=5,
            kernel_num=[16, 32, 64, 128, 256, 256],
            bidirectional=False,
            embed_dim=512,
            num_heads=32,
            attn_win=32,
            # 若有其它必要超參數，一併放進來
        )

        # 5) 建立模型 + 載權重（印出 missing/unexpected 供診斷）
        model = model_class(**model_kwargs)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        print(f"✓ Model loaded (hardcoded) → BinauralAttentionDCNN on {device}")
        print(f"  ↳ weights: loaded_keys={len(state_dict)} "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("    missing (top-10):", list(missing)[:10])
        if unexpected:
            print("    unexpected (top-10):", list(unexpected)[:10])

        return model
    
    def _register_layer_name(self, layer: nn.Module, base_name: str) -> str:
        """Register and return standardized layer name."""
        layer_type = layer.__class__.__name__.lower()
        layer_name = f"{self.layer_index:02d}_{layer_type}_{base_name}"
        self.layer_order.append(layer_name)
        self.layer_index += 1
        # 讓後續 fake-quant / dump 都能用到一致命名
        try:
            setattr(layer, "_ptq_layer_name", layer_name)
        except Exception:
            pass
        return layer_name
    
    def _should_quantize_layer(self, layer: nn.Module) -> Tuple[bool, str]:
        """Determine quantization strategy for layer."""
        layer_name = layer.__class__.__name__

        # --- Attention: 原生 + complex 版本 ---
        if isinstance(layer, (nn.MultiheadAttention,) + COMPLEX_ATTN) or \
        'attention' in layer_name.lower() or 'multihead' in layer_name.lower():
            return True, 'bf16'
        
        # --- Conv / Linear：原生 + complex 版本（INT8）---
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear) + COMPLEX_CONVS + COMPLEX_LINEAR) or \
        any(x in layer_name.lower() for x in ['conv', 'linear']):
            return True, 'int8'
        
        # --- BN / Activation：原生 + complex 版本（但只做 activation fake-quant，不量化權重）---
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d) + COMPLEX_BN) or \
        any(x in layer_name.lower() for x in ['batchnorm', 'prelu', 'relu', 'tanh']):
            return True, 'int8_act_only'  # 新增：只做 activation 量化
        
        return False, 'none'
    
    def _calibrate_model(self, model: nn.Module) -> Dict[str, Any]:
        """Calibrate model using calibration dataset."""
        print(f"🔧 Calibrating model with data from {self.args.calib_dir}")
        
        calib_files = list(Path(self.args.calib_dir).glob("*.wav"))
        if not calib_files:
            print(f"⚠️  No calibration files found in {self.args.calib_dir}")
            return {}
        
        calibration_stats = {}
        model.eval()
        
        # Create hook factory function
        def make_calibration_hook(module_name):
            @torch.no_grad()
            def hook_fn(module, input, output):
                if module_name not in calibration_stats:
                    calibration_stats[module_name] = {
                        'input_max': [],
                        'output_max': [],
                        'weight_max': []
                    }
                
                # Input activation stats
                if input and len(input) > 0:
                    inp = input[0]
                    if isinstance(inp, torch.Tensor) and inp.numel() > 0:
                        # Handle complex tensors
                        if torch.is_complex(inp):
                            inp_max = torch.maximum(inp.real.abs().max(), inp.imag.abs().max()).item()
                        else:
                            inp_max = inp.abs().max().item()
                        calibration_stats[module_name]['input_max'].append(inp_max)
                
                # Output activation stats
                if isinstance(output, torch.Tensor) and output.numel() > 0:
                    # Handle complex tensors
                    if torch.is_complex(output):
                        out_max = torch.maximum(output.real.abs().max(), output.imag.abs().max()).item()
                    else:
                        out_max = output.abs().max().item()
                    calibration_stats[module_name]['output_max'].append(out_max)
                
                # Weight stats（修正：1D 權重改為每通道統計）
                if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2:
                    weight = module.weight
                    if torch.is_complex(weight):
                        # Complex weights: handle real and imag separately
                        weight_real_max = weight.real.abs().amax(dim=tuple(range(1, weight.dim())))
                        weight_imag_max = weight.imag.abs().amax(dim=tuple(range(1, weight.dim())))
                        weight_max = torch.maximum(weight_real_max, weight_imag_max)
                    else:
                        # Real weights: per out_channel max
                        weight_max = weight.abs().amax(dim=tuple(range(1, weight.dim())))
                    calibration_stats[module_name]['weight_max'] = weight_max.cpu().tolist()
            
            return hook_fn
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            should_quant, _ = self._should_quantize_layer(module)
            if should_quant:
                hook_fn = make_calibration_hook(name)
                handle = module.register_forward_hook(hook_fn)
                hooks.append(handle)
        
        # Run calibration
        with torch.no_grad():
            for i, calib_file in enumerate(calib_files[:min(50, len(calib_files))]):  # Limit to 50 files
                try:
                    # Load audio file (assuming torchaudio format)
                    import torchaudio
                    audio, sr = torchaudio.load(calib_file)
                    
                    # Ensure binaural (2 channels)
                    if audio.shape[0] == 1:
                        audio = audio.repeat(2, 1)
                    elif audio.shape[0] > 2:
                        audio = audio[:2]
                    
                    # Ensure proper length (2 seconds at 16kHz = 32000 samples)
                    target_length = 32000
                    if audio.shape[1] < target_length:
                        audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[1]))
                    else:
                        audio = audio[:, :target_length]
                    
                    audio = audio.unsqueeze(0).to(self.device)  # Add batch dimension
                    
                    # Forward pass
                    _ = model(audio)
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Processed {i + 1}/{len(calib_files)} calibration files")
                        
                except Exception as e:
                    print(f"  ⚠️  Error processing {calib_file}: {e}")
                    continue
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        print(f"✓ Calibration completed with {len(calib_files)} files")
        return calibration_stats
    
    def _quantize_model(self, model: nn.Module, calibration_stats: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply quantization to model layers."""
        print("🔄 Applying quantization...")
        
        quant_summary = {}
        
        for name, module in model.named_modules():
            should_quant, quant_type = self._should_quantize_layer(module)
            if not should_quant:
                continue
            
            layer_name = self._register_layer_name(module, name.replace('.', '_'))
            stats = calibration_stats.get(name, {})
            
            if quant_type == 'int8':
                quantizer = INT8Quantizer()
                
                # Weight quantization（修正：跳過 1D 權重）
                if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2:
                    weight_scales = quantizer.calibrate_weights(module.weight, stats.get('weight_max', []))
                    quant_weight = quantizer.quantize_weights(module.weight, weight_scales)

                    # -- NEW: complex dict path --
                    if isinstance(quant_weight, dict):
                        # dequant 回寫 module 以便後續 FP32/BF16 前向
                        dequant_w = quantizer.dequantize_weights(quant_weight, weight_scales)
                        module.weight.data.copy_(dequant_w)
                        # 存 scale
                        self.quant_manager.save_weight_params(layer_name, weight_scales)
                        # 分別 dump real / imag 為 INT8
                        self.dump_manager.dump_weights(layer_name + "_real", quant_weight['real'], weight_scales)
                        self.dump_manager.dump_weights(layer_name + "_imag", quant_weight['imag'], weight_scales)
                    else:
                        # 實數權重原路徑
                        dequant_w = quantizer.dequantize_weights(quant_weight, weight_scales)
                        module.weight.data.copy_(dequant_w)
                        self.quant_manager.save_weight_params(layer_name, weight_scales)
                        self.dump_manager.dump_weights(layer_name, quant_weight, weight_scales)
                
                # Activation quantization
                if stats.get('output_max'):
                    act_scale = quantizer.calibrate_activations(stats['output_max'])
                    self.quant_manager.save_activation_params(layer_name, act_scale)
                
                quant_summary[layer_name] = {
                    'w_bit': 8 if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2 else 0,
                    'w_axis': 'out_ch' if hasattr(module, 'weight') and module.weight is not None and module.weight.dim() >= 2 else 'none',
                    'w_scale_len': len(weight_scales) if 'weight_scales' in locals() else 0,
                    'act_bit': 8,
                    'act_scale': stats.get('output_max', [0])[-1] if stats.get('output_max') else 0
                }
            
            elif quant_type == 'int8_act_only':
                # 只做 activation 量化，不動權重
                quantizer = INT8Quantizer()
                
                # Activation quantization only
                if stats.get('output_max'):
                    act_scale = quantizer.calibrate_activations(stats['output_max'])
                    self.quant_manager.save_activation_params(layer_name, act_scale)
                
                quant_summary[layer_name] = {
                    'w_bit': 0,  # 不量化權重
                    'w_axis': 'none',
                    'w_scale_len': 0,
                    'act_bit': 8,
                    'act_scale': stats.get('output_max', [0])[-1] if stats.get('output_max') else 0
                }
            
            # ... 其他 quant_type 處理 ...
        
        return model, quant_summary
    
    def _evaluate_model(self, model: nn.Module) -> Dict[str, float]:
        """
        Evaluate using user's eval.py:
        - pairs clean/noisy by tag (base + azimuth)
        - calls safe_evaluate_file(clean, noisy, model, device)
        Returns averaged metrics with focus on 'mbstoi'.
        """
        print("📊 Evaluating model using eval.safe_evaluate_file")
        try:
            from eval import safe_evaluate_file
            import soundfile as sf  # used inside eval as well
            from pathlib import Path
            import re
        except Exception as e:
            print(f"⚠️  Cannot import eval helpers: {e}")
            return {'mbstoi': 0.0, 'error': str(e)}

        # Resolve dirs (優先使用 --eval-clean-dir/--eval-noisy-dir；否則用 --eval-dir/{clean,noisy})
        clean_dir = Path(self.args.eval_clean_dir) if self.args.eval_clean_dir else None
        noisy_dir = Path(self.args.eval_noisy_dir) if self.args.eval_noisy_dir else None
        if clean_dir is None or noisy_dir is None:
            if self.args.eval_dir is None:
                return {'mbstoi': 0.0, 'error': 'Either provide --eval-clean-dir & --eval-noisy-dir, or set --eval-dir.'}
            root = Path(self.args.eval_dir)
            clean_dir = clean_dir or (root / "clean")
            noisy_dir = noisy_dir or (root / "noisy")
        if not clean_dir.exists() or not noisy_dir.exists():
            return {'mbstoi': 0.0, 'error': f"Eval dirs not found: {clean_dir} / {noisy_dir}"}

        clean_files = list(clean_dir.rglob("*.wav"))
        noisy_files = list(noisy_dir.rglob("*.wav"))
        if not clean_files or not noisy_files:
            return {'mbstoi': 0.0, 'error': "No wav files under eval clean/noisy dirs"}

        # Build dicts keyed by (base, az)
        def _parse(stem: str):
            m = re.search(r'_az(-?\d+)(?:_snr([-+]?\d+(?:\.\d+)?))?$', stem)
            az = int(m.group(1)) if m else None
            base = stem.split('_az')[0]
            return base, az

        clean_map = {}
        for f in clean_files:
            base, az = _parse(f.stem)
            if az is not None:
                clean_map[(base, az)] = f

        noisy_map = {}
        for f in noisy_files:
            base, az = _parse(f.stem)
            if az is not None:
                # 若同 key 有多個 SNR 版本，任取其一（或可加策略挑最接近 paper SNR）
                noisy_map.setdefault((base, az), f)

        # Iterate pairs
        pairs = []
        for key, cpath in clean_map.items():
            npath = noisy_map.get(key)
            if npath:
                pairs.append((cpath, npath))
        if not pairs:
            return {'mbstoi': 0.0, 'error': "No (clean,noisy) pairs matched by base+az"}

        # Limit evaluation size for speed (可改成全部)
        max_pairs = min(len(pairs), 100)
        pairs = pairs[:max_pairs]

        mbstoi_vals = []
        for idx, (cpath, npath) in enumerate(pairs, 1):
            res = safe_evaluate_file(cpath, npath, model, self.device)
            if isinstance(res, dict) and 'MBSTOI' in res:
                mbstoi_vals.append(float(res['MBSTOI']))
            if idx % 20 == 0:
                print(f"  Evaluated {idx}/{len(pairs)} pairs")

        mb = float(np.mean(mbstoi_vals)) if mbstoi_vals else 0.0
        print(f"✓ Evaluation completed: MBSTOI = {mb:.3f} (avg over {len(mbstoi_vals)} pairs)")
        return {'mbstoi': mb, 'num_pairs': len(mbstoi_vals)}
    
    def _generate_streaming_output(self, model: nn.Module):
        """Generate per-frame streaming output."""
        print("🎬 Generating streaming output...")
        
        frame_runner = FrameRunner(
            model=model,
            device=self.device,
            attn_window=self.args.attn_window,
            dump_manager=self.dump_manager,
            layer_order=self.layer_order
        )
        
        # Process evaluation files
        # Fix: Handle the case where eval_dir is None but eval_clean_dir or eval_noisy_dir are provided
        eval_dir_path = None
        
        if self.args.eval_dir is not None:
            eval_dir_path = Path(self.args.eval_dir)
        elif self.args.eval_clean_dir is not None:
            # Use the clean directory as evaluation source
            eval_dir_path = Path(self.args.eval_clean_dir)
        elif self.args.eval_noisy_dir is not None:
            # Use the noisy directory as evaluation source
            eval_dir_path = Path(self.args.eval_noisy_dir)
        else:
            print("  ⚠️  No evaluation directory specified, skipping streaming output generation")
            return
        
        if not eval_dir_path.exists():
            print(f"  ⚠️  Evaluation directory {eval_dir_path} does not exist, skipping streaming output generation")
            return
        
        eval_files = list(eval_dir_path.glob("*.wav"))[:5]  # Limit to 5 files
        
        if not eval_files:
            # Try recursive search
            eval_files = list(eval_dir_path.rglob("*.wav"))[:5]
        
        if not eval_files:
            print(f"  ⚠️  No WAV files found in {eval_dir_path}, skipping streaming output generation")
            return
        
        for file_idx, eval_file in enumerate(eval_files):
            print(f"  Processing {eval_file.name} ({file_idx + 1}/{len(eval_files)})")
            
            try:
                import torchaudio
                audio, sr = torchaudio.load(eval_file)
                
                # Ensure binaural
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)
                elif audio.shape[0] > 2:
                    audio = audio[:2]
                
                frame_runner.process_audio(audio, file_idx)
                
            except Exception as e:
                print(f"    ⚠️  Error processing {eval_file}: {e}")
                continue
        
        print("✓ Streaming output generated")
    
    def run(self):
        """Main execution pipeline."""
        print(f"🚀 Starting PTQ Export (Run ID: {self.args.run_id})")
        print(f"   Model: BinauralAttentionDCNN (hardcoded)")
        print(f"   Weights: /raid/R12K41024/BCCTN_hardware/model.ckpt")
        print(f"   Output: {self.args.output_root}/{self.args.run_id}")
        
        start_time = time.time()
        
        # Create output directory
        self.dump_manager.setup_directories()
        
        # Load model
        model = self._load_model()
        self.model = model
        
        # Baseline evaluation (FP32)
        print("\n📊 Baseline FP32 Evaluation")
        fp32_metrics = self._evaluate_model(model)
        
        # Calibration
        calibration_stats = self._calibrate_model(model)
        
        # Quantization
        model, quant_summary = self._quantize_model(model, calibration_stats)

        # [新增] 掛活化 fake-quant（INT8）
        # 先標記哪些層是 INT8 目標層（Conv/Linear…）
        self._mark_int8_targets(model)
        # 從 quant_params 讀回各層活化 scale（或改成你內存的 map）
        act_scales = self._load_act_scales_from_disk()
        # 要提供一個能把 module -> 層名的 resolver（這裡用你已有的 _module_name_resolver）
        self.act_fakequant_mgr.attach(self.model, act_scales, self._module_name_resolver)

        # Post-quantization evaluation
        print("\n📊 Post-Quantization Evaluation")
        ptq_metrics = self._evaluate_model(model)
        
        # Generate streaming output
        if self.args.stream_dump == 'per-frame':
            self._generate_streaming_output(model)
        
        # Generate report
        report = {
            'run_id': self.args.run_id,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'env': {
                'torch': torch.__version__,
                'cuda': torch.version.cuda if torch.cuda.is_available() else 'N/A',
                'device': str(self.device)
            },
            'config': {
                'model_class': 'BinauralAttentionDCNN',
                'model_import': 'DCNN.models.binaural_attention_model (hardcoded)',
                'weights': '/raid/R12K41024/BCCTN_hardware/model.ckpt',
                'attn_window': self.args.attn_window,
                'stream_dump': self.args.stream_dump
            },
            'layer_order': self.layer_order,
            'quant_summary': quant_summary,
            'metrics': {
                'mbstoi_fp32': fp32_metrics.get('mbstoi', 0.0),
                'mbstoi_ptq': ptq_metrics.get('mbstoi', 0.0),
                'delta': ptq_metrics.get('mbstoi', 0.0) - fp32_metrics.get('mbstoi', 0.0)
            },
            'execution_time': time.time() - start_time
        }
        
        # Save report
        report_path = self.dump_manager.output_dir / 'report.json'
        write_json_file(report_path, report)
        metrics_dir = self.dump_manager.output_dir / 'metrics'
        metrics_dir.mkdir(exist_ok=True, parents=True)
        with open(metrics_dir / 'mbstoi.txt', 'w') as f:
            f.write(f"{report['metrics']['mbstoi_ptq']:.6f}\n")
        
        print(f"\n✅ PTQ Export completed in {report['execution_time']:.1f}s")
        print(f"📄 Report saved to: {report_path}")
        print(f"📊 MBSTOI: {fp32_metrics.get('mbstoi', 0.0):.3f} → {ptq_metrics.get('mbstoi', 0.0):.3f} (Δ{report['metrics']['delta']:+.3f})")


def main():
    parser = argparse.ArgumentParser(description='PTQ Export Tool for BCCTN')
    
    # Required arguments (一些改為可選，因為已寫死)
    parser.add_argument('--model-import', default='DCNN/models/binaural_attention_model.py',
                       help='Path to model Python file (hardcoded, for compatibility)')
    parser.add_argument('--model-class', default='BinauralAttentionDCNN',
                       help='Model class name (hardcoded, for compatibility)')
    parser.add_argument('--weights', default='/raid/R12K41024/BCCTN_hardware/model.ckpt',
                       help='Path to FP32 checkpoint (hardcoded, for compatibility)')
    parser.add_argument('--calib-dir', required=True,
                       help='Calibration dataset directory')
    parser.add_argument('--eval-dir', default=None,
                       help='Evaluation dataset root (optional unless --eval-clean-dir/--eval-noisy-dir are missing)')
    # [NEW] 可明確指定 clean/noisy 位置；若不給，預設用 eval-dir/clean & eval-dir/noisy
    parser.add_argument('--eval-clean-dir', default=None,
                       help='Clean dataset directory (optional)')
    parser.add_argument('--eval-noisy-dir', default=None,
                       help='Noisy dataset directory (optional)')
    
    # Optional arguments
    parser.add_argument('--output-root', default='artifacts',
                       help='Output root directory')
    parser.add_argument('--run-id', default=None,
                       help='Run identifier (default: timestamp)')
    parser.add_argument('--stream-dump', choices=['per-frame', 'full'], default='per-frame',
                       help='Streaming dump mode')
    parser.add_argument('--attn-window', default='32',
                       help='Attention window size (int or "none")')
    parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto',
                       help='Compute device')
    parser.add_argument('--eval-entry', default='eval:compute_mbstoi',
                       help='Evaluation function (module:function)')
    parser.add_argument('--dump-complex-split', type=str, default='true',
                       help='Split complex tensors into real/imag files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Generate run ID if not provided
    if args.run_id is None:
        args.run_id = time.strftime('ptq_%Y%m%d_%H%M%S')
    
    # Parse attention window
    if args.attn_window.lower() == 'none':
        args.attn_window = None
    else:
        try:
            args.attn_window = int(args.attn_window)
        except ValueError:
            print(f"❌ Invalid attention window: {args.attn_window}")
            sys.exit(1)
    
    # Normalize boolean flags
    args.dump_complex_split = str(args.dump_complex_split).lower() == 'true'
    
    # Run PTQ export
    exporter = PTQExporter(args)
    exporter.run()


if __name__ == '__main__':
    main()