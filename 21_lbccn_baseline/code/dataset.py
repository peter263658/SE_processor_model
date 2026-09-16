# Same paired-wav dataset as BCCTN/DCNN/datasets/base_dataset.py (index-paired
# sorted rglob) so LBCCN trains on exactly the same (noisy, clean) pairs.
import torch
import torchaudio

from pathlib import Path


SR = 16000


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, noisy_dataset_dir, target_dataset_dir, sr=SR, mono=False):
        self.sr = sr
        self.target_dataset_dir = target_dataset_dir
        self.noisy_dataset_dir = noisy_dataset_dir
        self.mono = mono

        self.noisy_file_paths = self._get_file_paths(noisy_dataset_dir)
        self.target_file_paths = self._get_file_paths(target_dataset_dir)

        assert len(self.noisy_file_paths) == len(self.target_file_paths), (
            f"noisy/clean count mismatch: {len(self.noisy_file_paths)} vs "
            f"{len(self.target_file_paths)}")

    def __len__(self):
        return len(self.noisy_file_paths)

    def __getitem__(self, index):
        clean_signal, _ = torchaudio.load(self.target_file_paths[index])
        noisy_signal, _ = torchaudio.load(self.noisy_file_paths[index])

        if self.mono:
            return (noisy_signal[0], clean_signal[0])
        return (noisy_signal, clean_signal)

    def _get_file_paths(self, dataset_dir):
        dataset_dir = Path(dataset_dir)
        return [str(fp) for fp in sorted(dataset_dir.rglob('*.wav'))]
