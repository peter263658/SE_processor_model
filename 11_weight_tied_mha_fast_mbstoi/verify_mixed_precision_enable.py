"""Verify that enable_mixed_precision_qat wires the full model without errors
and that a forward pass runs end-to-end.
"""

import torch
from omegaconf import OmegaConf

from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
from DCNN.utils.mixed_precision_qat import enable_mixed_precision_qat


def main():
    cfg = OmegaConf.load("config/config.yaml")
    model_cfg = OmegaConf.to_container(cfg.model)
    model = BinauralAttentionDCNN(**model_cfg).eval()

    num_heads = model_cfg.get("num_heads", 16)
    mp_cfg = {
        "encoder_dw": {"weight_bits": 4, "weight_scale": "per_channel"},
        "encoder_pw": {"weight_bits": 4, "weight_scale": "per_group", "group_size": 256},
        "decoder_dw": {"weight_bits": 4, "weight_scale": "per_channel"},
        "decoder_pw_outlier": {"weight_bits": 8, "weight_format": "fp8_e4m3"},
        "decoder_pw_normal":  {"weight_bits": 4, "weight_scale": "per_group", "group_size": 256},
        "mha": {"weight_bits": 4, "weight_scale": "per_head", "num_heads": num_heads},
    }
    enable_mixed_precision_qat(model, config=mp_cfg, activation_format="e4m3", top_k=4)

    # 1 second of stereo audio @ 16 kHz
    x = torch.randn(1, 2, 16000)
    y = model(x)
    print(f"input  shape: {x.shape}")
    print(f"output shape: {y.shape}")
    assert y.shape[0] == 1 and y.shape[1] == 2

    # Backward pass — ensure STE propagates gradients to all weights.
    loss = y.pow(2).mean()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters())
    print(f"params with non-zero grad: {n_with_grad} / {n_total}")
    assert n_with_grad > n_total * 0.5, "too few params got gradients — STE broken?"
    print("Forward + backward with mixed-precision QAT enabled: OK")


if __name__ == "__main__":
    main()
