from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import soundfile as sf
import torch

import export_helpers as eh

try:
    from DCNN.utils.fp8_qat import activation_fp8_mode
except Exception:
    def activation_fp8_mode() -> str:
        return os.environ.get("BCCTN_ACTIVATION_FP8_MODE", "scaled").strip().lower()


DEFAULT_WAV_GLOB = "/raid/R12K41024/BCCTN/Dataset/noisy_testset/snr_0dB/*.wav"
OLD_PATTERN_DIRS = ("golden", "golden_layer_io", "golden_layer_io_full")

LEAF_TYPE_NAMES = {
    "Stft", "IStft",
    "ParametrizedConv2d", "ParametrizedConvTranspose2d",
    "ParametrizedGroupedLinear", "ParametrizedMultiheadAttention",
    "ParametrizedNonDynamicallyQuantizableLinear",
    "Conv2d", "ConvTranspose2d", "BatchNorm2d", "PReLU", "Linear",
    "MultiheadAttention",
}


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _ensure(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _write_raw_bin(path: str | Path, arr: np.ndarray) -> None:
    path = Path(path)
    _ensure(path.parent)
    np.asarray(arr).tofile(path)


def _copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    _ensure(dst.parent)
    shutil.copy2(src, dst)
    return True


def _copy_rel(src_root: Path, dst_root: Path, rel: str | Path) -> bool:
    return _copy_file(src_root / rel, dst_root / rel)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def _write_debug_tensor(prefix: Path, tensor: torch.Tensor) -> Dict[str, Any]:
    t = tensor.detach().cpu().contiguous()
    meta: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "is_complex": bool(t.is_complex()),
    }
    if t.is_complex():
        eh.write_tensor(str(prefix) + ".real.fp32.bin", t.real, torch.float32)
        eh.write_tensor(str(prefix) + ".imag.fp32.bin", t.imag, torch.float32)
        meta["files"] = {
            "real": os.path.basename(str(prefix) + ".real.fp32.bin"),
            "imag": os.path.basename(str(prefix) + ".imag.fp32.bin"),
        }
    else:
        eh.write_tensor(str(prefix) + ".fp32.bin", t, torch.float32)
        meta["files"] = {"value": os.path.basename(str(prefix) + ".fp32.bin")}
    return meta


def _write_debug_tensor_list(call_dir: Path, stem: str, values: list[torch.Tensor]) -> list[Dict[str, Any]]:
    return [_write_debug_tensor(call_dir / f"{stem}_{idx}", value) for idx, value in enumerate(values)]


# ── Block-float activation: e4m3 raw code + per-tensor pow2 shared exponent ──
# The activation descale 2^-k is fused into the weight scale (see fused weight
# descales below), so TB activation files are PURE raw code, no per-activation
# scale. k is the static calibration exponent for the tensor's boundary.
_BF_TABLE: Optional[Dict[str, Any]] = None


def _load_bf_table() -> Dict[str, Any]:
    global _BF_TABLE
    if _BF_TABLE is None:
        path = os.environ.get("BCCTN_ACTIVATION_CALIB_PATH", "").strip() or eh.DEFAULT_ACT_CALIB
        _BF_TABLE = eh.blockfloat_activation_table(path)
    return _BF_TABLE


def _boundary_exp(key: str) -> Optional[int]:
    e = _load_bf_table().get(key)
    return None if e is None else int(e["scale_pow2_exp"])


def _resolve_boundary_exp(name: str, complex_part: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    """(pow2 exponent k, boundary_key) for a hooked leaf's INPUT activation (single-k cases)."""
    m = re.search(r"((encoder|decoder)\.model\.\d+\.0\.(depthwise|pointwise)_[ri])$", name)
    if m:
        return _boundary_exp(m.group(1)), m.group(1)
    if name == "mattn.input_proj" and complex_part:
        key = f"mattn.input_proj.{complex_part}"
        return _boundary_exp(key), key
    if name == "mattn.output_proj" and complex_part:
        key = f"mattn.output_proj.{complex_part}"
        return _boundary_exp(key), key
    return None, None


def _resolve_boundary_candidates(name: str, complex_part: Optional[str]) -> list[tuple[int, str]]:
    """Candidate (k, boundary_key) for a leaf's input activation.

    Decoder split-pointwise sub-convs are called by apply_complex(fr, fi, x) on BOTH
    x.real (cat.real exponent) and x.imag (cat.imag exponent) — the SAME module sees
    two different exponents across its two calls. So return BOTH cat.real and cat.imag
    candidates and let the caller pick per call by fp8 round-trip. All other leaves
    have a single boundary exponent.
    """
    m = re.search(r"(decoder\.model\.\d+\.0)\.pointwise_[ri]_(x|skip)$", name)
    if m:
        out = []
        for part in ("real", "imag"):
            key = f"{m.group(1)}.split_pointwise_cat.{part}"
            k = _boundary_exp(key)
            if k is not None:
                out.append((k, key))
        return out
    k, key = _resolve_boundary_exp(name, complex_part)
    return [(k, key)] if k is not None else []


def _write_blockfloat_real(prefix: Path, tensor: torch.Tensor,
                           candidates: list[tuple[int, str]]) -> Dict[str, Any]:
    t = tensor.detach().to(torch.float32).cpu().contiguous()
    if not candidates:
        return {
            "skipped": True,
            "reason": "not a block-float activation boundary (Stft/iStft Q1.15 DSP, "
                      "BN/PReLU wide affine, MHA-internal qkv, or out_proj wide accumulator)",
            "shape": list(t.shape),
        }
    # The runtime hook applied scale=2^k => value = code * 2^-k. We do NOT know a priori
    # which exponent this particular call used (a decoder pointwise sub-conv is invoked on
    # both the real-slice and the imag-slice), so pick the candidate k whose recovered code
    # lands exactly on the fp8 grid (round-trip ~ 0). This is correct per-call/per-branch.
    best = None
    for k, key in candidates:
        code_f = t * (2.0 ** k)
        code = code_f.to(torch.float8_e4m3fn)
        rt = float((code.to(torch.float32) - code_f).abs().max().item())
        if best is None or rt < best[2]:
            best = (k, key, rt, code)
    k, key, rt, code = best
    code_file = prefix.name + ".e4m3.u8.bin"
    _write_raw_bin(prefix.parent / code_file, code.view(torch.uint8).contiguous().numpy())
    meta = {
        "format": "blockfloat_e4m3_raw_code",
        "shape": list(t.shape),
        "boundary_key": key,
        "scale_pow2_exp": k,           # this call's activation = e4m3(x * 2^k)
        "descale_pow2_exp": -k,        # fused into the layer's weight descale (NOT applied here)
        "fp8_roundtrip_max": rt,       # ~0 confirms the picked exponent matches this call
        "files": {"code_u8": code_file},
        "reconstruction": "raw e4m3 code; effective_value = code * 2^-k. 2^-k is FUSED into "
                          "this layer's weight descale (one post-acc shift), per branch.",
    }
    if len(candidates) > 1:
        meta["candidate_exps"] = [k for k, _ in candidates]
        meta["per_branch_note"] = ("decoder split-pointwise: cat.real and cat.imag exponents may "
                                   "differ (e.g. stage3 real=5,imag=6); exponent picked per call.")
    return meta


def _write_blockfloat_tensor(prefix: Path, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
    t = tensor.detach()
    meta: Dict[str, Any] = {
        "shape": list(t.shape),
        "dtype_before_encoding": str(t.dtype),
        "is_complex": bool(t.is_complex()),
    }
    if t.is_complex():
        meta["real"] = _write_blockfloat_real(prefix.with_name(prefix.name + ".real"), t.real,
                                               _resolve_boundary_candidates(name, "real"))
        meta["imag"] = _write_blockfloat_real(prefix.with_name(prefix.name + ".imag"), t.imag,
                                               _resolve_boundary_candidates(name, "imag"))
        meta["complex_quantization"] = "real and imag use independent per-tensor pow2 exponents"
        return meta
    if not t.is_floating_point():
        meta["skipped"] = True
        meta["reason"] = "non-floating tensor is not an activation"
        return meta
    meta.update(_write_blockfloat_real(prefix, t, _resolve_boundary_candidates(name, None)))
    return meta


def _info_emitted(info: Dict[str, Any]) -> bool:
    """True only if a block-float code file was actually written for this tensor."""
    if info.get("files"):
        return True
    for part in ("real", "imag"):
        sub = info.get(part)
        if isinstance(sub, dict) and sub.get("files"):
            return True
    return False


def _install_capture_hooks(model, debug_layer_dir: Path, tb_act_dir: Path):
    counters: Dict[str, int] = {}
    debug_meta: Dict[str, Any] = {}
    tb_meta: Dict[str, Any] = {}
    hooks = []

    def hook_for(name: str):
        safe = _safe_name(name)

        def _hook(module, inputs, output):
            idx = counters.get(safe, 0)
            counters[safe] = idx + 1
            key = f"{name}__{idx}"
            debug_call_dir = debug_layer_dir / "layers" / f"{safe}__{idx}"
            _ensure(debug_call_dir)

            xs = _collect_tensors(inputs)
            ys = _collect_tensors(output)
            base = {"name": name, "module": type(module).__name__, "call_index": idx}

            # debug_reference: full fp32 input+output (separate folder, untouched)
            debug_item: Dict[str, Any] = dict(base)
            if xs:
                debug_item["inputs"] = _write_debug_tensor_list(debug_call_dir, "input", xs)
                debug_item["input"] = debug_item["inputs"][0]
            if ys:
                debug_item["outputs"] = _write_debug_tensor_list(debug_call_dir, "output", ys)
                debug_item["output"] = debug_item["outputs"][0]
            debug_meta[key] = debug_item

            # tb_pattern: block-float raw e4m3 code for INPUT activations only (the MAC
            # operands). Outputs are pre-quant (BN/PReLU follow) -> debug only. Stft/IStft
            # and non-boundary leaves resolve to k=None -> skipped (F1 fix).
            tb_item: Dict[str, Any] = dict(base)
            tb_item["activation_format"] = "blockfloat_e4m3_raw_code"
            tb_item["note"] = (
                "Input = raw e4m3 code at a block-float boundary; its descale 2^-k is FUSED "
                "into this layer's weight descale (see tb_pattern weights/*fused*). Outputs and "
                "non-arithmetic leaves are in debug_reference only."
            )
            emitted = False
            if xs:
                tb_call_dir = tb_act_dir / f"{safe}__{idx}"
                _ensure(tb_call_dir)
                ins = [_write_blockfloat_tensor(tb_call_dir / f"input_{j}", xt, name)
                       for j, xt in enumerate(xs)]
                emitted = any(_info_emitted(info) for info in ins)
                if emitted:
                    tb_item["inputs"] = ins
                    tb_item["input"] = ins[0]
                else:
                    # nothing written (Stft/iSTFT/BN/PReLU/MHA-internal/out_proj) — drop empty dir
                    try:
                        tb_call_dir.rmdir()
                    except OSError:
                        pass
                    tb_item["skipped"] = True
                    tb_item["reason"] = "no block-float activation boundary for this leaf's inputs"
            else:
                tb_item["skipped"] = True
                tb_item["reason"] = "no tensor inputs"
            tb_meta[key] = tb_item

        return _hook

    for name, module in model.named_modules():
        if type(module).__name__ in LEAF_TYPE_NAMES:
            hooks.append(module.register_forward_hook(hook_for(name)))
    return hooks, debug_meta, tb_meta


def _copy_tb_weight_dirs(export_root: Path, tb_dir: Path) -> Dict[str, int]:
    counts = {"weight_dirs": 0, "block_dirs": 0, "files": 0}
    src_weights = export_root / "weights"
    dst_root = tb_dir

    for rel in ("manifest.json", "layer_config.json"):
        if _copy_rel(export_root, dst_root, rel):
            counts["files"] += 1

    for meta_path in src_weights.rglob("meta.json"):
        rel_dir = meta_path.parent.relative_to(export_root)
        meta = _read_json(meta_path)
        copied = 0
        for rel_file in _tb_files_from_weight_meta(meta):
            if _copy_rel(export_root, dst_root, rel_dir / rel_file):
                copied += 1
        if _copy_rel(export_root, dst_root, rel_dir / "meta.json"):
            copied += 1
        if copied:
            counts["weight_dirs"] += 1
            counts["files"] += copied

    for meta_path in (export_root / "blocks").rglob("meta.json"):
        rel_dir = meta_path.parent.relative_to(export_root)
        copied = 0
        for file_path in meta_path.parent.glob("*.bin"):
            name = file_path.name
            is_hw = (
                "_apot_" in name
                or "_pow2_" in name
                or name in {"bn_bias_r_fused.fp32.bin", "bn_bias_i_fused.fp32.bin"}
            )
            if is_hw and _copy_rel(export_root, dst_root, rel_dir / name):
                copied += 1
        if _copy_rel(export_root, dst_root, rel_dir / "meta.json"):
            copied += 1
        if copied:
            counts["block_dirs"] += 1
            counts["files"] += copied

    for subdir in ("skip_indices", "misc"):
        src = export_root / subdir
        if not src.exists():
            continue
        for file_path in src.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix == ".bin" or file_path.name == "meta.json":
                rel = file_path.relative_to(export_root)
                if _copy_rel(export_root, dst_root, rel):
                    counts["files"] += 1

    return counts


def _emit_fused_weight_descales(export_root: Path, tb_dir: Path) -> Dict[str, Any]:
    """Pre-fuse the activation block-float descale 2^-k into each weight×activation
    layer's APoT descale: fused exponents = weight_a/b_exp − k (uniform integer
    shift; has_second / residual_sign are invariant under a common shift). Emits
    `*_fused.int16.bin` next to the copied weight scales so the TB has ONE descale
    per layer and activations are pure raw code.

    Clean for INT4 convs (own/cat boundary) and the 12 FP8 outliers (cat boundary).
    MHA projections (real/imag boundaries may differ) and the internal MHA are left
    un-fused here — they keep the separate activation exponent (see manifest).
    """
    out = {"fused_layers": 0, "per_branch_layers": 0, "skipped_layers": 0,
           "fused": {}, "skipped": [], "fused_exp_range": [None, None]}
    lo = [None]
    hi = [None]

    def _shift_write(d: Path, a_f: str, b_f: str, k: int, suffix: str) -> list[str]:
        a = np.fromfile(d / a_f, dtype=np.int16).astype(np.int32) - k
        b = np.fromfile(d / b_f, dtype=np.int16).astype(np.int32) - k
        fa = a_f.replace(".int16.bin", f"{suffix}.int16.bin")
        fb = b_f.replace(".int16.bin", f"{suffix}.int16.bin")
        a.astype(np.int16).tofile(d / fa)
        b.astype(np.int16).tofile(d / fb)
        rmin, rmax = int(min(a.min(), b.min())), int(max(a.max(), b.max()))
        lo[0] = rmin if lo[0] is None else min(lo[0], rmin)
        hi[0] = rmax if hi[0] is None else max(hi[0], rmax)
        return [fa, fb]

    for meta_path in (tb_dir / "weights").rglob("meta.json"):
        meta = _read_json(meta_path)
        dotted = meta.get("dotted", "")
        d = meta_path.parent
        a_f, b_f = (("fp8_descale_apot_a_exp.int16.bin", "fp8_descale_apot_b_exp.int16.bin")
                    if meta.get("is_decoder_pw_outlier")
                    else ("scale_apot_a_exp.int16.bin", "scale_apot_b_exp.int16.bin"))
        if not (d / a_f).exists() or not (d / b_f).exists():
            out["skipped_layers"] += 1
            out["skipped"].append(dotted + " (no APoT exp files)")
            continue

        m = re.search(r"(decoder\.model\.\d+\.0)\.pointwise_[ri]_(x|skip)$", dotted)
        if m:
            # apply_complex uses this weight on BOTH the real-slice (cat.real exp) and the
            # imag-slice (cat.imag exp). Emit a PER-BRANCH fused descale for each; HW selects
            # by which activation slice the weight multiplies. (stage3 real=5 != imag=6.)
            kr = _boundary_exp(f"{m.group(1)}.split_pointwise_cat.real")
            ki = _boundary_exp(f"{m.group(1)}.split_pointwise_cat.imag")
            files = {}
            if kr is not None:
                files["act_real"] = {"k": kr, "fused_files": _shift_write(d, a_f, b_f, kr, "_fused_act_kreal")}
            if ki is not None:
                files["act_imag"] = {"k": ki, "fused_files": _shift_write(d, a_f, b_f, ki, "_fused_act_kimag")}
            out["per_branch_layers"] += 1
            out["fused"][dotted] = {
                "type": "per_branch_complex_pointwise",
                "boundary_keys": {"real": f"{m.group(1)}.split_pointwise_cat.real",
                                  "imag": f"{m.group(1)}.split_pointwise_cat.imag"},
                "branches": files,
                "branches_collapse": (kr == ki),
            }
            continue

        k, key = _resolve_boundary_exp(dotted, None)
        if k is None:
            out["skipped_layers"] += 1
            out["skipped"].append(dotted)
            continue
        out["fused_layers"] += 1
        out["fused"][dotted] = {"type": "single", "boundary_key": key, "activation_pow2_exp": k,
                                "fused_exp_offset": -k,
                                "fused_files": _shift_write(d, a_f, b_f, k, "_fused")}

    out["fused_exp_range"] = [lo[0], hi[0]]
    out["rule"] = ("fused_a/b_exp = weight_APoT_a/b_exp - activation_pow2_exp(k); "
                   "has_second/residual_sign unchanged. Activation = raw e4m3 code (no descale). "
                   "Decoder split-pointwise emits TWO variants (_fused_act_kreal / _fused_act_kimag) "
                   "because apply_complex multiplies each weight by both real- and imag-slice "
                   "activations; HW picks the branch matching the activation slice. Other layers: "
                   "one fused descale.")
    out["not_fused_note"] = ("MHA in/out projections and internal MHA keep the separate activation "
                             "pow2 exponent (real/imag boundaries differ; out_proj is W4A16). "
                             "QK/score×V are activation×activation (no weight to fuse into).")
    return out


def _tb_files_from_weight_meta(meta: Dict[str, Any]) -> list[str]:
    files: list[str] = []
    preferred = meta.get("preferred_hw_weight_file")
    if preferred:
        files.append(str(preferred))

    scale_meta = meta.get("scale_meta") or {}
    for name in scale_meta.get("apot_files") or []:
        files.append(str(name))

    for name in meta.get("descale_apot_files") or []:
        files.append(str(name))

    bias = meta.get("bias_file")
    if bias:
        files.append(str(bias))

    # Keep order stable while removing duplicates.
    out: list[str] = []
    seen = set()
    for name in files:
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


def _copy_debug_export(export_root: Path, debug_dir: Path) -> Dict[str, int]:
    dst_root = debug_dir / "export_reference_debug"
    counts = {"files": 0}

    for rel in ("manifest.json", "layer_config.json"):
        if _copy_rel(export_root, dst_root, rel):
            counts["files"] += 1

    for subdir in ("weights", "blocks", "skip_indices", "misc"):
        src = export_root / subdir
        if not src.exists():
            continue
        for file_path in src.rglob("*"):
            if not file_path.is_file():
                continue
            name = file_path.name
            keep = (
                name == "meta.json"
                or name.endswith(".hex")
                or name.endswith(".dec")
                or ".fp32.bin" in name
                or ".fp16.bin" in name
                or ".raw_s8.bin" in name
            )
            if keep:
                rel = file_path.relative_to(export_root)
                if _copy_rel(export_root, dst_root, rel):
                    counts["files"] += 1
    return counts


def _write_readmes(out_root: Path, tb_dir: Path, debug_dir: Path, source_export: Path) -> None:
    readme = f"""# HW Pattern Package

Generated by `software_to_hardware_18/prepare_tb_patterns.py`.

Use `tb_pattern/` for RTL/testbench inputs. Use `debug_reference/` only for
fp32/fp16/readable inspection.

Source export root: `{source_export}`
"""
    (out_root / "README.md").write_text(readme, encoding="utf-8")

    tb_readme = """# Testbench Pattern (block-float deploy)

Compact hardware-facing files only:

- `weights/`: packed INT4/FP8 weight codes + APoT/pow2 weight-scale codes, plus the
  **fused** descale = weight APoT exponents − k (k = the activation block-float exponent).
  - Most layers: one `*_apot_*_fused.int16.bin` (single activation exponent).
  - **Decoder split-pointwise** (`pointwise_*`): TWO variants
    `*_apot_*_fused_act_kreal.int16.bin` and `*_apot_*_fused_act_kimag.int16.bin`,
    because `apply_complex(fr, fi, x)` multiplies each sub-conv weight by BOTH the
    real-slice (cat.real exp) and imag-slice (cat.imag exp) activations. They are
    identical where cat.real==cat.imag (most stages) but DIFFER at stage 3
    (real=2⁻⁵, imag=2⁻⁶). HW must pick the variant matching the activation slice —
    do NOT assume one fused descale per weight here.
  See `weights/fused_descale_summary.json` (`branches_collapse` flags the equal ones).
- `blocks/`: BN APoT scale codes, fused BN bias constants, PReLU pow2 codes.
- `skip_indices/`, `misc/`: deployment constants.
- `activations_fp8/`: per-layer **input** activations as **raw e4m3 uint8 code**
  (block-float). No per-activation scale/descale is stored — the descale 2^-k is
  already fused into the weight descale above. Outputs/non-arithmetic leaves are
  NOT here (debug_reference only). Stft/iSTFT are Q1.15 DSP, not block-float.

Activations are block-float e4m3 (mode `calibrated_pow2_e4m3`): recorded in
`activations_fp8/meta.json`. Exponent table also at
`../export_out_strict18/activation_blockfloat/`.
NOT fused (kept with separate exponent): MHA in/out projections (real/imag k may
differ; out_proj is W4A16) and QK/score×V (activation×activation, no weight).
"""
    (tb_dir / "README.md").write_text(tb_readme, encoding="utf-8")

    debug_readme = """# Debug Reference

This folder is for inspection and numerical comparison:

- `layer_io_fp32/`: fp32/complex64 module input/output tensors from the same
  golden waveform used to generate the block-float testbench activations (incl.
  Stft/iSTFT Q1.15 DSP I/O and all pre-quant outputs, which are NOT in tb_pattern).
- `export_reference_debug/`: readable fp32/fp16/raw_s8/hex/dec views copied from
  the full export.

Do not use this folder as the source of packed testbench weights.
"""
    (debug_dir / "README.md").write_text(debug_readme, encoding="utf-8")


def _remove_old_patterns(export_root: Path) -> list[str]:
    removed: list[str] = []
    for name in OLD_PATTERN_DIRS:
        path = export_root / name
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path))
    return removed


def _run_capture(args, tb_dir: Path, debug_dir: Path) -> Dict[str, Any]:
    wav_path = args.input_wav
    if wav_path is None:
        matches = sorted(glob.glob(args.wav_glob))
        if not matches:
            raise FileNotFoundError(f"no wav matched {args.wav_glob}")
        wav_path = matches[0]

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

    debug_layer_dir = debug_dir / "layer_io_fp32"
    tb_act_dir = tb_dir / "activations_fp8"
    hooks, debug_layer_meta, tb_act_meta = _install_capture_hooks(model, debug_layer_dir, tb_act_dir)

    with torch.no_grad():
        y = model(x)

    for h in hooks:
        h.remove()

    top_debug = {
        "input": _write_debug_tensor(debug_layer_dir / "input_waveform", x),
        "output": _write_debug_tensor(debug_layer_dir / "output_waveform", y),
    }
    top_tb_dir = tb_dir / "top_level_io"
    _ensure(top_tb_dir)
    _write_raw_bin(top_tb_dir / "input_waveform.fp32.bin", x.detach().cpu().contiguous().to(torch.float32).numpy())
    _write_raw_bin(top_tb_dir / "expected_output_waveform.fp32.bin", y.detach().cpu().contiguous().to(torch.float32).numpy())

    common = {
        "source_ckpt": os.path.abspath(args.ckpt),
        "config": os.path.abspath(args.config),
        "input_wav": os.path.abspath(wav_path),
        "sample_rate": int(sr),
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
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
            "fp8_dtype": training_cfg.get("fp8_dtype"),
        },
        "outlier_descale": {
            "delayed_apot_descale": os.environ.get("BCCTN_OUTLIER_DELAYED_DESCALE", "0") == "1",
            "apot_exp_min": int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MIN", "-16")),
            "apot_exp_max": int(os.environ.get("BCCTN_OUTLIER_DESCALE_APOT_MAX", "-6")),
        },
    }
    debug_meta = dict(common)
    debug_meta.update(top_debug)
    debug_meta["layer_io"] = debug_layer_meta
    eh.write_json(str(debug_layer_dir / "meta.json"), debug_meta)

    tb_meta = dict(common)
    mode = activation_fp8_mode()
    tb_meta["activation_mode"] = mode
    if mode in ("plain", "plain_e4m3", "plain_fp8", "raw", "raw_e4m3"):
        tb_meta["activation_format"] = "fp8_e4m3_plain_saturating"
    elif mode in ("calibrated_pow2", "calibrated_pow2_e4m3", "calib_pow2",
                  "pow2", "pow2_e4m3", "blockfloat", "bfp_e4m3"):
        tb_meta["activation_format"] = "blockfloat_e4m3_raw_code"
        tb_meta["activation_note"] = (
            "Block-float: e4m3 raw code + per-tensor pow2 shared exponent (see "
            "../export_out_strict18/activation_blockfloat/). Activation descale 2^-k is "
            "FUSED into the weight descale (weights/*_apot_*_fused.int16.bin); no per-activation "
            "scale is stored. Inputs only; outputs/Stft/iSTFT/BN/PReLU not encoded here."
        )
    else:
        tb_meta["activation_format"] = "fp8_e4m3_per_tensor_scaled"
    tb_meta["top_level_io"] = {
        "input_waveform": {"file": "top_level_io/input_waveform.fp32.bin", "shape": list(x.shape), "dtype": "float32"},
        "expected_output_waveform": {
            "file": "top_level_io/expected_output_waveform.fp32.bin",
            "shape": list(y.shape),
            "dtype": "float32",
        },
    }
    tb_meta["layer_activations"] = tb_act_meta
    eh.write_json(str(tb_act_dir / "meta.json"), tb_meta)
    return {
        "input_wav": wav_path,
        "sample_rate": int(sr),
        "layer_calls": len(tb_act_meta),
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-root", default=os.path.join(eh.PACKAGE_ROOT, "export_out_strict18"))
    ap.add_argument("--out-root", default=os.path.join(eh.PACKAGE_ROOT, "hw_patterns_final"))
    ap.add_argument("--ckpt", default=eh.DEFAULT_CKPT)
    ap.add_argument("--config", default=eh.DEFAULT_CONFIG)
    ap.add_argument("--code-dir", default=eh.DEFAULT_CODE_DIR)
    ap.add_argument("--skip-indices", default=eh.DEFAULT_SKIP_INDICES)
    ap.add_argument("--input-wav", default=None)
    ap.add_argument("--wav-glob", default=DEFAULT_WAV_GLOB)
    ap.add_argument("--input-len", type=int, default=32000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--delete-old-patterns", action="store_true")
    args = ap.parse_args()

    # Block-float deploy defaults: capture runs with the pow2 activation path so the
    # hooked inputs are the deployed block-float values (raw code recoverable exactly).
    os.environ.setdefault("BCCTN_ACTIVATION_FP8_MODE", "calibrated_pow2_e4m3")
    os.environ.setdefault("BCCTN_ACTIVATION_CALIB_PATH", eh.DEFAULT_ACT_CALIB)
    os.environ.setdefault("BCCTN_ACTIVATION_POW2_ROUND", "floor")
    os.environ.setdefault("BCCTN_OUTLIER_DELAYED_DESCALE", "1")
    os.environ.setdefault("BCCTN_OUTLIER_DESCALE_APOT_MIN", "-16")
    os.environ.setdefault("BCCTN_OUTLIER_DESCALE_APOT_MAX", "-6")

    export_root = Path(args.export_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not export_root.exists():
        raise FileNotFoundError(export_root)

    removed = _remove_old_patterns(export_root) if args.delete_old_patterns else []
    if out_root.exists():
        shutil.rmtree(out_root)
    tb_dir = out_root / "tb_pattern"
    debug_dir = out_root / "debug_reference"
    _ensure(tb_dir)
    _ensure(debug_dir)

    tb_counts = _copy_tb_weight_dirs(export_root, tb_dir)
    fused_meta = _emit_fused_weight_descales(export_root, tb_dir)
    eh.write_json(str(tb_dir / "weights" / "fused_descale_summary.json"), fused_meta)
    debug_counts = _copy_debug_export(export_root, debug_dir)
    capture = _run_capture(args, tb_dir, debug_dir)
    _write_readmes(out_root, tb_dir, debug_dir, export_root)

    summary = {
        "out_root": str(out_root),
        "tb_pattern": str(tb_dir),
        "debug_reference": str(debug_dir),
        "source_export_root": str(export_root),
        "activation_mode": os.environ.get("BCCTN_ACTIVATION_FP8_MODE"),
        "removed_old_patterns": removed,
        "tb_copy_counts": tb_counts,
        "fused_weight_descale": fused_meta,
        "debug_copy_counts": debug_counts,
        "capture": capture,
    }
    eh.write_json(str(out_root / "summary.json"), summary)

    print(f"[OK] prepared HW patterns under {out_root}")
    print(f"  tb_pattern={tb_dir}")
    print(f"  debug_reference={debug_dir}")
    print(f"  removed_old_patterns={len(removed)}")
    print(f"  activation_mode={os.environ.get('BCCTN_ACTIVATION_FP8_MODE')}")
    print(f"  layer_calls={capture['layer_calls']}")
    print(f"  tb_weight_dirs={tb_counts['weight_dirs']} block_dirs={tb_counts['block_dirs']} files={tb_counts['files']}")
    print(f"  fused_weight_descales: {fused_meta['fused_layers']} single + "
          f"{fused_meta['per_branch_layers']} per-branch (complex pointwise) "
          f"(exp range {fused_meta['fused_exp_range']}), {fused_meta['skipped_layers']} not fused")


if __name__ == "__main__":
    main()
