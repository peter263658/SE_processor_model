"""Sanity check: custom attention forward (used in top-k mode) must match
nn.MultiheadAttention at FP32 when top_k=None and activation_quantizer=None.

Run BEFORE training to ensure our re-implementation is correct.
"""

import math
import torch
import torch.nn as nn

from DCNN.utils.complexPyTorch.complexLayers import (
    ComplexMultiheadAttention,
    _mha_attention_forward,
)


def build_causal_sliding_mask(T: int, window: int, device) -> torch.Tensor:
    mask = torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)
    idx = torch.arange(T, device=device)
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    mask[dist > window] = float("-inf")
    return mask


def test_real_forward_matches_native():
    torch.manual_seed(0)
    embed_dim, num_heads, T, B = 256, 16, 32, 2
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True).eval()
    x = torch.randn(B, T, embed_dim)
    mask = build_causal_sliding_mask(T, window=7, device=x.device)

    ref, _ = mha(x, x, x, attn_mask=mask)
    ours = _mha_attention_forward(mha, x, mask, top_k=None, activation_quantizer=None)

    diff = (ref - ours).abs().max().item()
    print(f"[real] max |ref − ours| = {diff:.2e}")
    assert diff < 1e-5, f"custom forward does not match native MHA: diff={diff}"
    print("[real] OK")


def test_complex_forward_matches_native():
    torch.manual_seed(0)
    embed_dim, num_heads, T, B = 256, 16, 32, 2
    cmha = ComplexMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).eval()
    x = torch.randn(B, T, embed_dim, dtype=torch.complex64)
    mask = build_causal_sliding_mask(T, window=7, device=x.real.device)

    cmha.use_custom_forward = False
    ref = cmha(x, attn_mask=mask)

    cmha.use_custom_forward = True
    cmha.top_k = None
    cmha.activation_quantizer = None
    ours = cmha(x, attn_mask=mask)

    diff = (ref - ours).abs().max().item()
    print(f"[complex] max |ref − ours| = {diff:.2e}")
    assert diff < 1e-4, f"complex custom forward diverges: diff={diff}"
    print("[complex] OK")


def test_topk_shape():
    torch.manual_seed(0)
    embed_dim, num_heads, T, B = 256, 16, 32, 2
    cmha = ComplexMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True).eval()
    x = torch.randn(B, T, embed_dim, dtype=torch.complex64)
    mask = build_causal_sliding_mask(T, window=7, device=x.real.device)

    cmha.use_custom_forward = True
    cmha.top_k = 4
    cmha.activation_quantizer = None
    out = cmha(x, attn_mask=mask)
    assert out.shape == x.shape
    print(f"[top-k] output shape ok: {out.shape}")


if __name__ == "__main__":
    test_real_forward_matches_native()
    test_complex_forward_matches_native()
    test_topk_shape()
    print("All sanity checks passed.")
