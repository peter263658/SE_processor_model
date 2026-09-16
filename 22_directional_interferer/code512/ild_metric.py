import numpy as np
import librosa
from scipy import signal
SAMPLE_RATE=16000
FFT_SIZE=512
WIN_LENGTH=400
HOP_LENGTH=100

def compute_ild_error_with_mask(clean_left, clean_right, enhanced_left, enhanced_right, sr=SAMPLE_RATE):
    """
    Compute ILD error between clean and enhanced binaural signals with
    proper speech activity masking as described in the paper
    
    Args:
        clean_left, clean_right: Clean reference signals
        enhanced_left, enhanced_right: Enhanced signals
        sr: Sample rate
        
    Returns:
        ILD error in dB
    """
    # STFT parameters
    n_fft = FFT_SIZE
    hop_length = HOP_LENGTH
    win_length = WIN_LENGTH
    
    # Compute STFTs
    clean_left_stft = librosa.stft(clean_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    clean_right_stft = librosa.stft(clean_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    enhanced_left_stft = librosa.stft(enhanced_left, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    enhanced_right_stft = librosa.stft(enhanced_right, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    
    # Use a larger epsilon to avoid division by very small numbers
    eps = 1e-8
    
    # Compute magnitudes with epsilon to avoid zeros
    clean_left_mag = np.abs(clean_left_stft) + eps
    clean_right_mag = np.abs(clean_right_stft) + eps
    enhanced_left_mag = np.abs(enhanced_left_stft) + eps
    enhanced_right_mag = np.abs(enhanced_right_stft) + eps
    
    # Compute ILD in dB
    clean_ild = 20 * np.log10(clean_left_mag / clean_right_mag)
    enhanced_ild = 20 * np.log10(enhanced_left_mag / enhanced_right_mag)
    
    # Create speech activity mask as in the paper (T=20dB)
    threshold = 20  # dB below max as specified in paper
    
    # Combine energy from both channels for more robust mask
    combined_energy = np.maximum(clean_left_mag**2, clean_right_mag**2)
    max_energy = np.max(combined_energy)
    
    # Create Ideal Binary Mask (IBM) as described in the paper's Eq. (13-14)
    # For each frequency bin, threshold relative to maximum energy in that frequency
    mask = np.zeros_like(combined_energy, dtype=bool)
    for k in range(combined_energy.shape[0]):
        max_energy_k = np.max(combined_energy[k])
        mask[k] = 20 * np.log10(combined_energy[k]) > (20 * np.log10(max_energy_k) - threshold)
    
    # Compute mean absolute error in active regions
    ild_error = np.abs(clean_ild - enhanced_ild)
    
    # Handle NaN values before masking
    ild_error = np.nan_to_num(ild_error, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply mask and calculate mean
    active_bins = np.sum(mask)
    if active_bins == 0:
        return 0.0  # Return 0 if no active bins
        
    ild_error_masked = ild_error * mask
    mean_ild_error = np.sum(ild_error_masked) / active_bins
    
    return mean_ild_error


# 5. Updated IPD error calculation with speech activity mask
