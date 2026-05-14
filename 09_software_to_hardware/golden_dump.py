"""
Generate golden input/output patterns for the strict 09 weight-tied-MHA model.

Default output:
    golden/input_waveform.fp32.{bin,hex,dec}
    golden/output_waveform.fp32.{bin,hex,dec}
    golden/meta.json

Use --capture-layer-io to additionally dump first tensor input/output for each
leaf compute module. Complex tensors are split into real/imag fp32 streams.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import ckpt2hwdat as hw
import export_helpers as eh


DEFAULT_WAV_GLOB = "/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/*.wav"


def _ensure(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            out = _first_tensor(item)
            if out is not None:
                return out
    if isinstance(obj, dict):
        for item in obj.values():
            out = _first_tensor(item)
            if out is not None:
                return out
    return None


def _write_tensor(prefix: str, tensor: torch.Tensor) -> Dict[str, Any]:
    t = tensor.detach().cpu()
    meta: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "is_complex": bool(t.is_complex()),
    }

    if t.is_complex():
        real = t.real.contiguous().to(torch.float32).numpy()
        imag = t.imag.contiguous().to(torch.float32).numpy()
        hw._w_bin(prefix + ".real.fp32.bin", real)
        hw._w_bin(prefix + ".imag.fp32.bin", imag)
        meta["files"] = {
            "real": os.path.basename(prefix + ".real.fp32.bin"),
            "imag": os.path.basename(prefix + ".imag.fp32.bin"),
        }
    else:
        arr = t.contiguous().to(torch.float32).numpy()
        hw._w_bin(prefix + ".fp32.bin", arr)
        meta["files"] = {"value": os.path.basename(prefix + ".fp32.bin")}

    return meta


def _load_waveform(path: str, input_len: Optional[int]) -> tuple[torch.Tensor, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    audio = audio[:, :2]
    if input_len is not None:
        if audio.shape[0] < input_len:
            pad = np.zeros((input_len - audio.shape[0], 2), dtype=np.float32)
            audio = np.concatenate([audio, pad], axis=0)
        else:
            audio = audio[:input_len]
    # Model expects (batch, binaural_channels, time).
    x = torch.from_numpy(audio.T).unsqueeze(0).contiguous()
    return x, sr


def _build_model(config_path: str, ckpt_path: str, device: torch.device) -> tuple[nn.Module, Dict[str, Any], Dict[str, Any]]:
    cfg = OmegaConf.load(config_path)
    model_cfg = OmegaConf.to_container(cfg.model)
    training_cfg = OmegaConf.to_container(cfg.training)

    num_heads = model_cfg.get("num_heads", 16)
    int4_group_size = training_cfg.get("int4_group_size", 32)
    mp_cfg = {
        "encoder_dw": {"weight_bits": 4, "weight_scale": "per_channel"},
        "encoder_pw": {"weight_bits": 4, "weight_scale": "per_group", "group_size": int4_group_size},
        "decoder_dw": {"weight_bits": 4, "weight_scale": "per_channel"},
        "decoder_pw_outlier": {"weight_bits": 8, "weight_format": "fp8_e4m3"},
        "decoder_pw_normal": {"weight_bits": 4, "weight_scale": "per_group", "group_size": int4_group_size},
        "mha": {"weight_bits": 4, "weight_scale": "per_head", "num_heads": num_heads},
    }

    model = hw.BinauralAttentionDCNN(**model_cfg).eval()
    hw.enable_mixed_precision_qat(
        model,
        config=mp_cfg,
        activation_format=training_cfg.get("fp8_dtype", "e4m3"),
        top_k=training_cfg.get("top_k_attention", 4),
        fp16_accumulator=training_cfg.get("fp16_accumulator", False),
    )

    sd = eh.load_state_dict_from_ckpt(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_state_dict] missing={len(missing)} first={missing[:5]}")
    if unexpected:
        print(f"[load_state_dict] unexpected={len(unexpected)} first={unexpected[:5]}")

    return model.to(device), model_cfg, training_cfg


def _install_layer_hooks(model: nn.Module, outdir: str):
    counters: Dict[str, int] = {}
    layer_meta: Dict[str, Any] = {}
    hooks = []
    leaf_types = (
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.BatchNorm2d,
        nn.PReLU,
        nn.Linear,
        nn.MultiheadAttention,
    )

    def hook_for(name: str):
        safe = _sanitize(name)

        def _hook(module, inputs, output):
            idx = counters.get(safe, 0)
            counters[safe] = idx + 1
            call_dir = os.path.join(outdir, "layers", f"{safe}__{idx}")
            _ensure(call_dir)

            in_t = _first_tensor(inputs)
            out_t = _first_tensor(output)
            item: Dict[str, Any] = {
                "module": type(module).__name__,
                "name": name,
                "call_index": idx,
            }
            if in_t is not None:
                item["input"] = _write_tensor(os.path.join(call_dir, "input"), in_t)
            if out_t is not None:
                item["output"] = _write_tensor(os.path.join(call_dir, "output"), out_t)
            layer_meta[f"{name}__{idx}"] = item

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, leaf_types) or type(module).__name__ in {"Stft", "IStft"}:
            hooks.append(module.register_forward_hook(hook_for(name)))

    return hooks, layer_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=hw.DEFAULT_CKPT)
    ap.add_argument("--config", default=hw.DEFAULT_CONFIG)
    ap.add_argument("--input-wav", default=None)
    ap.add_argument("--wav-glob", default=DEFAULT_WAV_GLOB)
    ap.add_argument("--input-len", type=int, default=32000)
    ap.add_argument("--outdir", default="./export_out/golden")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--capture-layer-io", action="store_true")
    args = ap.parse_args()

    wav_path = args.input_wav
    if wav_path is None:
        matches = sorted(glob.glob(args.wav_glob))
        if not matches:
            raise FileNotFoundError(f"no wav files matched {args.wav_glob}")
        wav_path = matches[0]

    _ensure(args.outdir)
    device = torch.device(args.device)
    model, model_cfg, training_cfg = _build_model(args.config, args.ckpt, device)
    x, sr = _load_waveform(wav_path, args.input_len)
    x = x.to(device)

    hooks = []
    layer_meta: Dict[str, Any] = {}
    if args.capture_layer_io:
        hooks, layer_meta = _install_layer_hooks(model, args.outdir)

    with torch.no_grad():
        y = model(x)

    for h in hooks:
        h.remove()

    meta = {
        "source_ckpt": os.path.abspath(args.ckpt),
        "config": os.path.abspath(args.config),
        "input_wav": os.path.abspath(wav_path),
        "sample_rate": sr,
        "input": _write_tensor(os.path.join(args.outdir, "input_waveform"), x),
        "output": _write_tensor(os.path.join(args.outdir, "output_waveform"), y),
        "model": {
            "win_len": model_cfg.get("win_len"),
            "win_inc": model_cfg.get("win_inc"),
            "fft_len": model_cfg.get("fft_len"),
            "win_type": model_cfg.get("win_type"),
            "masking_mode": model_cfg.get("masking_mode"),
            "tied_mha": model_cfg.get("tied_mha"),
        },
        "training": {
            "qat_scope": training_cfg.get("qat_scope"),
            "top_k_attention": training_cfg.get("top_k_attention"),
            "int4_group_size": training_cfg.get("int4_group_size"),
        },
        "layer_io": layer_meta,
    }
    hw._w_json(os.path.join(args.outdir, "meta.json"), meta)
    print(f"[OK] golden dumped to {args.outdir}")
    print(f"  input:  {tuple(x.shape)}")
    print(f"  output: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
