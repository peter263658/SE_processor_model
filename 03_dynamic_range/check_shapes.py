import torch

def main():
    ckpt_path = '/raid/R12K41024/03_dynamic_range/model.ckpt'
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    
    with open('/raid/R12K41024/03_dynamic_range/weight_keys.txt', 'w') as f:
        for k, v in state_dict.items():
            if 'weight' in k:
                f.write(f"{k}: {v.shape}, dtype: {v.dtype}\n")

if __name__ == '__main__':
    main()
