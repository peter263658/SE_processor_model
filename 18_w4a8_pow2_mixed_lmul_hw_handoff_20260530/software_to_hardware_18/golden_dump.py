from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch

import export_helpers as eh


DEFAULT_WAV_GLOB = "/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/*.wav"


def _first_tensor(obj: Any) -> Optional[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(obj, dict):
        for item in obj.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _collect_tensors(obj: Any) -> list[torch.Tensor]:
    found: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        for item in obj:
            found.extend(_collect_tensors(item))
    elif isinstance(obj, dict):
        for item in obj.values():
            found.extend(_collect_tensors(item))
    return found


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _write_tensor(prefix: str, tensor: torch.Tensor) -> Dict[str, Any]:
    t = tensor.detach().cpu().contiguous()
    meta: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "is_complex": bool(t.is_complex()),
    }
    if t.is_complex():
        eh.write_tensor(prefix + ".real.fp32.bin", t.real, torch.float32)
        eh.write_tensor(prefix + ".imag.fp32.bin", t.imag, torch.float32)
        meta["files"] = {
            "real": os.path.basename(prefix + ".real.fp32.bin"),
            "imag": os.path.basename(prefix + ".imag.fp32.bin"),
        }
    else:
        eh.write_tensor(prefix + ".fp32.bin", t, torch.float32)
        meta["files"] = {"value": os.path.basename(prefix + ".fp32.bin")}
    return meta


def _write_tensor_list(call_dir: str, stem: str, values: list[torch.Tensor]) -> list[Dict[str, Any]]:
    entries = []
    for idx, value in enumerate(values):
        entries.append(_write_tensor(os.path.join(call_dir, f"{stem}_{idx}"), value))
    return entries


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
    return torch.from_numpy(audio.T).unsqueeze(0).contiguous(), sr


def _install_hooks(model, outdir: str):
    counters: Dict[str, int] = {}
    layer_meta: Dict[str, Any] = {}
    hooks = []
    leaf_type_names = {
        "Stft", "IStft",
        # Parametrized* names are produced by torch.nn.utils.parametrize and are
        # the actual QAT modules used in the final deploy graph.
        "ParametrizedConv2d", "ParametrizedConvTranspose2d",
        "ParametrizedGroupedLinear", "ParametrizedMultiheadAttention",
        "ParametrizedNonDynamicallyQuantizableLinear",
        "Conv2d", "ConvTranspose2d", "BatchNorm2d", "PReLU", "Linear",
        "MultiheadAttention",
    }

    def hook_for(name: str):
        safe = _safe_name(name)

        def _hook(module, inputs, output):
            idx = counters.get(safe, 0)
            counters[safe] = idx + 1
            call_dir = os.path.join(outdir, "layers", f"{safe}__{idx}")
            eh.ensure(call_dir)
            item: Dict[str, Any] = {
                "name": name,
                "module": type(module).__name__,
                "call_index": idx,
            }
            xs = _collect_tensors(inputs)
            ys = _collect_tensors(output)
            if xs:
                item["inputs"] = _write_tensor_list(call_dir, "input", xs)
                item["input"] = item["inputs"][0]
            if ys:
                item["outputs"] = _write_tensor_list(call_dir, "output", ys)
                item["output"] = item["outputs"][0]
            layer_meta[f"{name}__{idx}"] = item

        return _hook

    for name, module in model.named_modules():
        if type(module).__name__ in leaf_type_names:
            hooks.append(module.register_forward_hook(hook_for(name)))
    return hooks, layer_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=eh.DEFAULT_CKPT)
    ap.add_argument("--config", default=eh.DEFAULT_CONFIG)
    ap.add_argument("--code-dir", default=eh.DEFAULT_CODE_DIR)
    ap.add_argument("--skip-indices", default=eh.DEFAULT_SKIP_INDICES)
    ap.add_argument("--input-wav", default=None)
    ap.add_argument("--wav-glob", default=DEFAULT_WAV_GLOB)
    ap.add_argument("--input-len", type=int, default=32000)
    ap.add_argument("--outdir", default=os.path.join(eh.PACKAGE_ROOT, "export_out_strict18", "golden"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--capture-layer-io", action="store_true")
    args = ap.parse_args()

    wav_path = args.input_wav
    if wav_path is None:
        matches = sorted(glob.glob(args.wav_glob))
        if not matches:
            raise FileNotFoundError(f"no wav matched {args.wav_glob}")
        wav_path = matches[0]

    eh.ensure(args.outdir)
    model, model_cfg, training_cfg, load_report = eh.build_model(
        config_path=args.config,
        ckpt_path=args.ckpt,
        code_dir=args.code_dir,
        skip_indices_path=args.skip_indices,
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    x, sr = _load_waveform(wav_path, args.input_len)
    x = x.to(device)

    hooks = []
    layer_meta: Dict[str, Any] = {}
    if args.capture_layer_io:
        hooks, layer_meta = _install_hooks(model, args.outdir)

    with torch.no_grad():
        y = model(x)

    for h in hooks:
        h.remove()

    meta = {
        "source_ckpt": os.path.abspath(args.ckpt),
        "config": os.path.abspath(args.config),
        "input_wav": os.path.abspath(wav_path),
        "sample_rate": int(sr),
        "input": _write_tensor(os.path.join(args.outdir, "input_waveform"), x),
        "output": _write_tensor(os.path.join(args.outdir, "output_waveform"), y),
        "load_report": {
            "missing_count": len(load_report["missing"]),
            "unexpected_count": len(load_report["unexpected"]),
        },
        "model": {
            "win_len": model_cfg.get("win_len"),
            "win_inc": model_cfg.get("win_inc"),
            "fft_len": model_cfg.get("fft_len"),
            "win_type": model_cfg.get("win_type"),
            "masking_mode": model_cfg.get("masking_mode"),
        },
        "training": {
            "qat_scope": training_cfg.get("qat_scope"),
            "top_k_attention": training_cfg.get("top_k_attention"),
            "int4_group_size": training_cfg.get("int4_group_size"),
            "int4_scale_format": training_cfg.get("int4_scale_format"),
        },
        "activation": {
            "fp8_mode": os.environ.get("BCCTN_ACTIVATION_FP8_MODE", "scaled"),
            "calib_path": os.environ.get("BCCTN_ACTIVATION_CALIB_PATH", ""),
            "pow2_round": os.environ.get("BCCTN_ACTIVATION_POW2_ROUND", "floor"),
            "note": "calibrated_pow2_e4m3 => block-float: raw e4m3 code + per-tensor pow2 exponent; "
                    "eager==fused exactly, so golden == deployed HW. out_proj input is the fp16 "
                    "Score x V accumulator (W4A16, not block-float A8).",
        },
        "outlier_descale": {
            "delayed_apot_descale": os.environ.get("BCCTN_OUTLIER_DELAYED_DESCALE", "0") == "1",
            "apot_exp_min": int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MIN", "-16")),
            "apot_exp_max": int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MAX", "-6")),
            "note": "delayed=true => golden = true §3.4 HW path: APoT(amax/448)*sum(LMul(code,act)). "
                    "delayed=false => effective-weight path (NOT the deployed hardware).",
        },
        "layer_io": layer_meta,
    }
    eh.write_json(os.path.join(args.outdir, "meta.json"), meta)
    print(f"[OK] golden dumped to {args.outdir}")
    print(f"  input={tuple(x.shape)} output={tuple(y.shape)}")


if __name__ == "__main__":
    main()
