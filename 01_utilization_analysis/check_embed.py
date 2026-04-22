import torch

def print_state_dict_shapes(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")["model_state_dict"]
    print(f"—— Shapes in {ckpt_path} ——")
    for key, tensor in sd.items():
        print(f"{key:50s} : {tuple(tensor.shape)}")
    print("\n\n")

if __name__ == "__main__":
    print_state_dict_shapes("/raid/R12K41024/BCCTN/outputs/2025-05-31/23-59-43/logs/lightning_logs/version_0/checkpoints/epoch=98-step=95832.ckpt")
    print_state_dict_shapes("/raid/R12K41024/BCCTN/outputs/2025-05-31/03-57-53/logs/lightning_logs/version_0/checkpoints/epoch=96-step=93896.ckpt")
