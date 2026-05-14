# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# 列出 layer-by-layer：
# 1. 參數量 / dtype / 記憶體占用
# 2. 權重統計 (min‧max‧mean‧std)
# 3. 256-bin histogram，可選擇存成 PNG

# 用法：
#     python model_inspect.py --ckpt path/to/xxx.pt --save-hist
# """
# import argparse, os, torch, math
# from collections import OrderedDict
# import matplotlib.pyplot as plt        # 需要可視化時才用
# from DCNN.models.binaural_attention_model import BinauralAttentionDCNN
# from DCNN.models.model import DCNN

# # ===== 解析參數 =====
# ap = argparse.ArgumentParser()
# ap.add_argument("--ckpt",      type=str, default=None, help="checkpoint 路徑 (.pt / .pth)")
# ap.add_argument("--save-hist", action="store_true",    help="是否存 histogram PNG")
# ap.add_argument("--bins",      type=int, default=256,  help="histogram bin 數")
# args = ap.parse_args()

# # ===== 建立 / 載入模型 =====
# net = BinauralAttentionDCNN().cpu().eval()  # 重要：確定在 CPU、eval 模式
# if args.ckpt:
#     print(f"[+] Load checkpoint: {args.ckpt}")
#     sd = torch.load(args.ckpt, map_location="cpu")
#     # 兼容 lightning / ddp
#     if "state_dict" in sd: sd = sd["state_dict"]
#     net.load_state_dict(sd, strict=False)

# # ===== 統計函式 =====
# def bytes_per_elem(dtype):
#     # Complex: real+imag 各佔 float 大小
#     if dtype == torch.complex64:
#         return 2 * (torch.finfo(torch.float32).bits // 8)   # 8 bytes
#     if dtype == torch.complex128:
#         return 2 * (torch.finfo(torch.float64).bits // 8)   # 16 bytes
#     # Float / BFloat
#     if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
#         return torch.finfo(dtype).bits // 8
#     # Int / UInt / Bool
#     if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
#         return torch.iinfo(dtype).bits // 8
#     raise ValueError(f"Unsupported dtype: {dtype}")

# def summarize_layer(name, module, bins=256, save_hist=False):
#     params = list(module.parameters(recurse=False))
#     if not params:
#         return None
#     w = torch.cat([p.data.view(-1) for p in params]).cpu().float()

#     dtype = params[0].dtype
#     bpe   = bytes_per_elem(dtype)
#     n     = w.numel()
#     mb    = n * bpe / 1024**2
#     mb_q8 = n * 1  / 1024**2

#     stat = OrderedDict(
#         layer = name,
#         cls   = module.__class__.__name__,
#         n     = n,
#         dtype = str(dtype),
#         mb    = mb,
#         mb_q8 = mb_q8,
#         w_min = float(w.min()),
#         w_max = float(w.max()),
#         w_mu  = float(w.mean()),
#         w_sd  = float(w.std(unbiased=False)),
#     )

#     if save_hist:
#         os.makedirs("hist", exist_ok=True)
#         plt.figure()
#         plt.hist(w.numpy(), bins=bins)
#         plt.title(f"{name} ({stat['cls']})")
#         plt.xlabel("weight")
#         plt.ylabel("count")
#         plt.tight_layout()
#         plt.savefig(f"hist/{name.replace('.', '_')}.png")
#         plt.close()

#     return stat

# # ===== 蒐集結果 =====
# table = []
# for name, mod in net.named_modules():
#     if name == "":                       # skip root
#         continue
#     s = summarize_layer(name, mod, bins=args.bins, save_hist=args.save_hist)
#     if s: table.append(s)

# # ===== 列印 =====
# header = f'{"Layer":40}{"Type":20}{"Params":>12}{"DType":>10}{"MB":>9}{"MB@8":>9}{"min":>10}{"max":>10}{"mean":>10}{"std":>10}'
# print(header)
# print("-"*len(header))
# tot_n = tot_mb = tot_mb_q8 = 0
# for r in table:
#     print(f'{r["layer"]:40}{r["cls"]:20}{r["n"]:12,}{r["dtype"]:>10}'
#           f'{r["mb"]:9.2f}{r["mb_q8"]:9.2f}'
#           f'{r["w_min"]:10.3e}{r["w_max"]:10.3e}{r["w_mu"]:10.3e}{r["w_sd"]:10.3e}')
#     tot_n    += r["n"]
#     tot_mb   += r["mb"]
#     tot_mb_q8+= r["mb_q8"]
# print("-"*len(header))
# print(f'{"TOTAL":60}{tot_n:12,}{"":>10}{tot_mb:9.2f}{tot_mb_q8:9.2f}')

# if args.save_hist:
#     print("[+] histograms saved to ./hist/*.png")
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
列出 layer-by-layer：
1. 參數量 / dtype / 記憶體占用 / 維度 (shape)
2. 權重統計 (min‧max‧mean‧std)
3. 256-bin histogram (可選)
4. 儲存成 CSV 檔

用法：
    python model_inspect.py --config config.yaml --ckpt path/to/ckpt.pt --save-hist
"""
import argparse
import os
import yaml
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd

# ===== 解析參數 =====
ap = argparse.ArgumentParser()
ap.add_argument("--config",   type=str, required=True, help="config.yaml 路徑")
ap.add_argument("--ckpt",     type=str, default=None,  help="checkpoint 路徑 (.pt/.pth)")
ap.add_argument("--save-hist",action="store_true",  help="是否存 histogram PNG")
ap.add_argument("--bins",     type=int, default=256,   help="histogram bin 數")
ap.add_argument("--out-csv",  type=str, default="model_inspect.csv", help="輸出 CSV 檔案路徑")
args = ap.parse_args()

# ===== 載入 config & 模型 =====
with open(args.config) as f:
    cfg = yaml.safe_load(f)
from DCNN.models.binaural_attention_model import BinauralAttentionDCNN

net = BinauralAttentionDCNN().cpu().eval()
if args.ckpt:
    print(f"[+] Load checkpoint: {args.ckpt}")
    sd = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    new_sd = OrderedDict()
    for k, v in sd.items():
        key = k.replace('module.', '')
        new_sd[key] = v
    net.load_state_dict(new_sd, strict=False)

# ===== 記憶體單位函式 =====
def bytes_per_elem(dtype):
    if dtype == torch.complex64:
        return 2 * (torch.finfo(torch.float32).bits // 8)
    if dtype == torch.complex128:
        return 2 * (torch.finfo(torch.float64).bits // 8)
    if dtype in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        return torch.finfo(dtype).bits // 8
    if dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
        return torch.iinfo(dtype).bits // 8
    raise ValueError(f"Unsupported dtype: {dtype}")

# ===== 模組統計 =====
def summarize_layer(name, module, bins=256, save_hist=False):
    params = list(module.parameters(recurse=False))
    if not params:
        return None
    w = torch.cat([p.data.view(-1).cpu().float() for p in params])
    dtype = params[0].dtype
    bpe   = bytes_per_elem(dtype)
    n     = w.numel()
    mb    = n * bpe / 1024**2
    mb_q8 = n * 1   / 1024**2

    shapes = ";".join(str(tuple(p.shape)) for p in params)
    return OrderedDict(
        layer = name,
        cls   = module.__class__.__name__,
        n     = n,
        dtype = str(dtype),
        mb    = mb,
        mb_q8 = mb_q8,
        shapes= shapes,
        w_min = float(w.min()),
        w_max = float(w.max()),
        w_mu  = float(w.mean()),
        w_sd  = float(w.std(unbiased=False)),
    )

# ===== 蒐集結果 =====
rows = []
for name, mod in net.named_modules():
    if not name:
        continue
    stat = summarize_layer(name, mod, bins=args.bins, save_hist=args.save_hist)
    if stat:
        rows.append(stat)

# ===== 輸出 CSV =====
df = pd.DataFrame(rows)
df.to_csv(args.out_csv, index=False)
print(f"[+] Saved summary to {args.out_csv}")

# ===== 同步印到終端 =====
print(df.to_string(index=False))
