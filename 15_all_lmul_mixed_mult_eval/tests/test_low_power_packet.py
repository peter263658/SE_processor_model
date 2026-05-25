#!/usr/bin/env python3

import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from DCNN.utils.complexPyTorch.complexLayers import (  # noqa: E402
    _mha_attention_forward,
    _lmul_matmul,
    _lmul_windowed_qk_matmul,
    _packetize_q08_attention_probs,
    _q08_packet_fixed_matmul,
    _ste_round,
)


def _old_unrestricted_packet(probs, packet_cfg):
    top_k = int(packet_cfg.get("top_k", 4))
    prob_scale = float(packet_cfg.get("prob_scale", 256))
    prob_max_code = int(packet_cfg.get("prob_max_code", 255))
    k = min(max(1, top_k), probs.size(-1))

    topk_probs, topk_idx = torch.topk(probs.clamp(min=0.0), k=k, dim=-1)
    codes = _ste_round(topk_probs * prob_scale).clamp(0.0, float(prob_max_code))
    packet_probs = codes / prob_scale
    return {
        "probs": torch.zeros_like(probs).scatter(-1, topk_idx, packet_probs.to(probs.dtype)),
        "codes": torch.zeros_like(probs).scatter(-1, topk_idx, codes.to(probs.dtype)),
        "indices": topk_idx,
    }


def test_q08_packet_boundary_rows_stay_in_window_and_mid_rows_match_old_path():
    torch.manual_seed(7)
    t = 64
    width = 32
    cfg = {
        "top_k": 4,
        "local_idx_bits": 5,
        "prob_scale": 256,
        "prob_max_code": 255,
        "validate_local_idx": True,
        "renormalize": False,
    }

    row = torch.arange(t).view(t, 1)
    key = torch.arange(t).view(1, t)
    in_window = ((row - key) >= 0) & ((row - key) < width)
    scores = torch.randn(1, 1, t, t).masked_fill(~in_window.view(1, 1, t, t), float("-inf"))
    probs = F.softmax(scores, dim=-1)

    packet = _packetize_q08_attention_probs(probs, cfg)
    old_packet = _old_unrestricted_packet(probs, cfg)

    assert torch.equal(packet["probs"][..., width - 1:, :], old_packet["probs"][..., width - 1:, :])
    assert torch.equal(packet["codes"][..., width - 1:, :], old_packet["codes"][..., width - 1:, :])
    assert torch.equal(packet["indices"][..., width - 1:, :], old_packet["indices"][..., width - 1:, :])

    early_non_window = ~in_window[: width - 1].view(1, 1, width - 1, t)
    assert torch.count_nonzero(packet["probs"][..., : width - 1, :].masked_select(early_non_window)) == 0

    packet_rows = torch.arange(t).view(1, 1, t, 1)
    local_idx = packet_rows - packet["indices"]
    assert torch.all((local_idx >= 0) & (local_idx < width))


def test_q08_packet_fixed_matmul_matches_dense_reconstruction():
    torch.manual_seed(11)
    t = 16
    width = 8
    cfg = {
        "top_k": 4,
        "local_idx_bits": 3,
        "prob_scale": 256,
        "prob_max_code": 255,
        "validate_local_idx": True,
        "renormalize": False,
    }

    row = torch.arange(t).view(t, 1)
    key = torch.arange(t).view(1, t)
    in_window = ((row - key) >= 0) & ((row - key) < width)
    scores = torch.randn(1, 2, t, t).masked_fill(~in_window.view(1, 1, t, t), float("-inf"))
    probs = F.softmax(scores, dim=-1)
    v = torch.randn(1, 2, t, 5)

    packet = _packetize_q08_attention_probs(probs, cfg)
    packet_out = _q08_packet_fixed_matmul(packet, v, cfg)
    dense_out = torch.matmul(packet["codes"].to(torch.float32), v.to(torch.float32)) / cfg["prob_scale"]

    assert torch.allclose(packet_out, dense_out.to(packet_out.dtype), atol=1e-6, rtol=0.0)


def test_windowed_lmul_qk_matches_dense_inside_causal_window():
    torch.manual_seed(19)
    t = 17
    width = 8
    q = torch.randn(1, 2, t, 6)
    k = torch.randn(1, 2, t, 6)
    k_t = k.transpose(-2, -1)

    dense = _lmul_matmul(q, k_t, mantissa_bits=4, bit_exact=True)
    windowed = _lmul_windowed_qk_matmul(
        q, k_t, width, mantissa_bits=4, bit_exact=True, chunk_size=5
    )

    row = torch.arange(t).view(t, 1)
    key = torch.arange(t).view(1, t)
    in_window = ((row - key) >= 0) & ((row - key) < width)
    in_window = in_window.view(1, 1, t, t)

    assert torch.equal(windowed.masked_select(in_window), dense.masked_select(in_window))
    assert torch.count_nonzero(windowed.masked_select(~in_window)) == 0


def test_windowed_lmul_qk_attention_forward_matches_dense_path():
    torch.manual_seed(23)
    batch = 2
    t = 40
    embed_dim = 64
    heads = 8
    x = torch.randn(batch, t, embed_dim)
    mha = torch.nn.MultiheadAttention(embed_dim, heads, batch_first=True)

    mask = torch.triu(torch.full((t, t), float("-inf")), diagonal=1)
    idx = torch.arange(t)
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    mask[dist > 31] = float("-inf")

    attention_datapath = {
        "enabled": True,
        "score_quant": "none",
        "prob_quant": "none",
        "qk_scale_mode": "pow2",
        "lmul_enabled": True,
        "lmul_targets": ["qk"],
        "lmul_mantissa_bits": 4,
        "lmul_bit_exact": True,
        "prob_packet": {"enabled": False},
    }

    windowed = _mha_attention_forward(
        mha, x, mask, top_k=4, fp16_accumulator=True,
        softmax_mode="shift_topk_pow2_clamp8_nomantissa",
        attention_datapath=attention_datapath,
    )

    dense_cfg = dict(attention_datapath)
    dense_cfg["lmul_qk_force_dense"] = True
    dense = _mha_attention_forward(
        mha, x, mask, top_k=4, fp16_accumulator=True,
        softmax_mode="shift_topk_pow2_clamp8_nomantissa",
        attention_datapath=dense_cfg,
    )

    assert torch.equal(windowed, dense)


if __name__ == "__main__":
    test_q08_packet_boundary_rows_stay_in_window_and_mid_rows_match_old_path()
    test_q08_packet_fixed_matmul_matches_dense_reconstruction()
    test_windowed_lmul_qk_matches_dense_inside_causal_window()
    test_windowed_lmul_qk_attention_forward_matches_dense_path()
