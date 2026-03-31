"""
Frame runner for streaming audio processing with attention cache management.
Handles STFT/ISTFT processing, causal masking, and per-frame dumping.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from dump_utils import DumpManager


class AttentionCache:
    """Manages K/V cache for attention layers with windowing."""
    
    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self.cache = {}  # layer_name -> {'k': deque, 'v': deque}
    
    def update_cache(self, layer_name: str, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache and return concatenated K/V for attention.
        
        Args:
            layer_name: Name of attention layer
            k: Current frame K tensor [batch, heads, 1, d_k]
            v: Current frame V tensor [batch, heads, 1, d_v]
        
        Returns:
            Tuple of (full_k, full_v) tensors for attention computation
        """
        if layer_name not in self.cache:
            self.cache[layer_name] = {'k': deque(), 'v': deque()}
        
        cache_entry = self.cache[layer_name]
        
        # Add current frame to cache
        cache_entry['k'].append(k)
        cache_entry['v'].append(v)
        
        # Apply window size limit
        if self.window_size is not None:
            while len(cache_entry['k']) > self.window_size:
                cache_entry['k'].popleft()
                cache_entry['v'].popleft()
        
        # Concatenate all cached frames
        full_k = torch.cat(list(cache_entry['k']), dim=2)  # [batch, heads, seq_len, d_k]
        full_v = torch.cat(list(cache_entry['v']), dim=2)  # [batch, heads, seq_len, d_v]
        
        return full_k, full_v
    
    def get_cache_size(self, layer_name: str) -> int:
        """Get current cache size for a layer."""
        if layer_name not in self.cache:
            return 0
        return len(self.cache[layer_name]['k'])
    
    def clear_cache(self, layer_name: Optional[str] = None):
        """Clear cache for specific layer or all layers."""
        if layer_name:
            if layer_name in self.cache:
                self.cache[layer_name]['k'].clear()
                self.cache[layer_name]['v'].clear()
        else:
            self.cache.clear()


class CausalMaskGenerator:
    """Generates causal attention masks with optional windowing."""
    
    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
    
    def generate_mask(self, seq_len: int, current_pos: int, device: torch.device) -> torch.Tensor:
        """
        Generate causal attention mask.
        
        Args:
            seq_len: Sequence length
            current_pos: Current position in sequence
            device: Target device
        
        Returns:
            Attention mask tensor
        """
        if self.window_size is None:
            # Full causal mask
            mask = torch.triu(
                torch.full((seq_len, seq_len), float('-inf'), device=device),
                diagonal=1
            )
        else:
            # Windowed causal mask
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            
            for i in range(seq_len):
                # Allow attention to previous window_size positions and current position
                start_pos = max(0, i - self.window_size + 1)
                mask[i, start_pos:i+1] = 0.0
        
        return mask


class STFTProcessor:
    """Handles STFT/ISTFT operations with frame-by-frame processing."""
    
    # 調整預設為你的模型設定（win_len=128, hop=80, fft_len=128）
    def __init__(self, n_fft: int = 128, hop_length: int = 80, win_length: int = 128):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)
        
        # Overlap-add buffer for ISTFT
        self.istft_buffer = None
        self.frame_count = 0
    
    def frame_stft(self, audio_frame: torch.Tensor) -> torch.Tensor:
        """
        Compute STFT for a single audio frame.
        
        Args:
            audio_frame: Audio frame [channels, samples]
        
        Returns:
            STFT tensor [channels, freq_bins, 1]
        """
        if audio_frame.device != self.window.device:
            self.window = self.window.to(audio_frame.device)
        
        # Apply STFT to each channel
        stft_frames = []
        for ch in range(audio_frame.shape[0]):
            stft = torch.stft(
                audio_frame[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                return_complex=True,
                center=False
            )
            stft_frames.append(stft.unsqueeze(0))  # Add channel dim
        
        return torch.cat(stft_frames, dim=0)  # [channels, freq_bins, time_frames]
    
    def frame_istft(self, stft_frame: torch.Tensor) -> torch.Tensor:
        """
        Compute ISTFT for a single STFT frame using overlap-add.
        
        Args:
            stft_frame: STFT frame [channels, freq_bins, 1]
        
        Returns:
            Audio frame [channels, hop_length]
        """
        if stft_frame.device != self.window.device:
            self.window = self.window.to(stft_frame.device)
        
        audio_frames = []
        for ch in range(stft_frame.shape[0]):
            # ISTFT for single frame
            audio = torch.istft(
                stft_frame[ch],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=False
            )
            audio_frames.append(audio.unsqueeze(0))
        
        return torch.cat(audio_frames, dim=0)


class LayerHookManager:
    """Manages forward hooks for layer interception."""
    
    def __init__(self, dump_manager: DumpManager, layer_order: List[str]):
        self.dump_manager = dump_manager
        self.layer_order = layer_order
        self.hooks = []
        self.frame_idx = 0
        self.layer_outputs = {}
    
    def register_hooks(self, model: nn.Module):
        """Register forward hooks on target layers."""
        layer_name_map = self._build_layer_name_map(model)
        
        for name, module in model.named_modules():
            if self._should_hook_layer(module):
                layer_name = layer_name_map.get(name, name.replace('.', '_'))
                hook = module.register_forward_hook(
                    self._create_hook_fn(layer_name, module)
                )
                self.hooks.append(hook)
    
    def _build_layer_name_map(self, model: nn.Module) -> Dict[str, str]:
        """Build mapping from module names to standardized layer names."""
        name_map = {}
        layer_idx = 0
        
        for name, module in model.named_modules():
            if self._should_hook_layer(module):
                layer_type = module.__class__.__name__.lower()
                friendly_name = name.replace('.', '_')
                standardized_name = f"{layer_idx:02d}_{layer_type}_{friendly_name}"
                name_map[name] = standardized_name
                layer_idx += 1
        
        return name_map
    
    def _should_hook_layer(self, module: nn.Module) -> bool:
        """Determine if layer should be hooked."""
        module_name = module.__class__.__name__.lower()
        
        # Hook conv, linear, attention, batchnorm, activation layers
        hook_patterns = [
            'conv', 'linear', 'attention', 'multihead', 
            'batchnorm', 'prelu', 'relu', 'tanh'
        ]
        
        return any(pattern in module_name for pattern in hook_patterns)
    
    def _create_hook_fn(self, layer_name: str, module: nn.Module):
        """Create hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Dump input
            if input and len(input) > 0 and isinstance(input[0], torch.Tensor):
                self.dump_manager.dump_activations(
                    layer_name, input[0], self.frame_idx, is_input=True
                )
            
            # Dump output
            if isinstance(output, torch.Tensor):
                self.dump_manager.dump_activations(
                    layer_name, output, self.frame_idx, is_input=False
                )
                self.layer_outputs[layer_name] = output
            
            # Special handling for attention layers
            if 'attention' in module.__class__.__name__.lower():
                self._handle_attention_layer(layer_name, module, input, output)
        
        return hook_fn
    
    def _handle_attention_layer(self, layer_name: str, module: nn.Module, 
                               input: Tuple, output: torch.Tensor):
        """Special handling for attention layer outputs."""
        # Try to extract Q, K, V projections and attention weights
        # This is model-specific and may need adjustment
        try:
            # Check if module has accessible Q, K, V projections
            if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                # Extract QKV outputs (this would need model-specific implementation)
                qkv_outputs = {}
                # Note: This is a simplified version - actual implementation 
                # would need to intercept intermediate computations
                
                self.dump_manager.dump_attention_outputs(
                    layer_name, output, qkv_outputs, self.frame_idx
                )
        except Exception as e:
            # Fallback to regular output dumping
            pass
    
    def set_frame_index(self, frame_idx: int):
        """Set current frame index for dumping."""
        self.frame_idx = frame_idx
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


class FrameRunner:
    """Main frame runner for streaming audio processing."""
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 attn_window: Optional[int] = None, 
                 dump_manager: Optional[DumpManager] = None,
                 layer_order: Optional[List[str]] = None):
        self.model = model
        self.device = device
        self.attn_window = attn_window
        self.dump_manager = dump_manager
        
        # Initialize components
        self.attention_cache = AttentionCache(attn_window)
        self.causal_mask_gen = CausalMaskGenerator(attn_window)
        self.stft_processor = STFTProcessor()
        
        # Hook manager for layer interception
        self.hook_manager = None
        if dump_manager and layer_order:
            self.hook_manager = LayerHookManager(dump_manager, layer_order)
            self.hook_manager.register_hooks(model)
        
        # Frame processing state
        self.frame_count = 0
        self.audio_buffer = None
    
    def process_audio(self, audio: torch.Tensor, file_idx: int = 0):
        """
        Process complete audio file frame by frame.
        
        Args:
            audio: Audio tensor [channels, samples]
            file_idx: File index for output naming
        """
        audio = audio.to(self.device)
        
        # Reset state for new file
        self.frame_count = 0
        self.attention_cache.clear_cache()
        
        # Calculate frame parameters
        hop_length = self.stft_processor.hop_length
        total_frames = (audio.shape[1] - self.stft_processor.win_length) // hop_length + 1
        
        print(f"    Processing {total_frames} frames...")
        
        enhanced_frames = []
        
        for frame_idx in range(total_frames):
            # Extract audio frame
            start_sample = frame_idx * hop_length
            end_sample = start_sample + self.stft_processor.win_length
            audio_frame = audio[:, start_sample:end_sample]
            
            # Set frame index for dumping
            if self.hook_manager:
                self.hook_manager.set_frame_index(frame_idx)
            
            # Process frame
            enhanced_frame = self.process_frame(audio_frame, frame_idx)
            enhanced_frames.append(enhanced_frame)
            
            if (frame_idx + 1) % 50 == 0:
                print(f"      Processed {frame_idx + 1}/{total_frames} frames")
        
        # Combine enhanced frames
        if enhanced_frames:
            enhanced_audio = self.combine_frames(enhanced_frames)
            
            # Dump complete enhanced audio
            if self.dump_manager:
                output_file = self.dump_manager.streaming_dir / f"enhanced_audio_file{file_idx:03d}.hex"
                self.dump_manager.dump_activations("enhanced_output", enhanced_audio)
    
    def process_frame(self, audio_frame: torch.Tensor, frame_idx: int) -> torch.Tensor:
        # 直接把「波形」丟進模型；模型內部自行 STFT/ISTFT
        input_batch = audio_frame.unsqueeze(0)  # [1, channels, samples]
        with torch.no_grad():
            enhanced_wave = self._model_forward_with_cache(input_batch, frame_idx)
        # 模型通常輸出 [1, channels, samples]
        return enhanced_wave.squeeze(0)
    
    def _model_forward_with_cache(self, input_batch: torch.Tensor, frame_idx: int) -> torch.Tensor:
        """
        Forward pass through model with attention cache management.
        
        Args:
            input_batch: Input STFT batch [1, channels, freq_bins, 1]
            frame_idx: Current frame index
        
        Returns:
            Enhanced STFT batch
        """
        # This is a simplified version - actual implementation would need
        # to intercept attention layers and manage K/V cache
        
        # For now, just run the model normally
        # In a real implementation, you would need to:
        # 1. Intercept attention layers
        # 2. Update K/V cache
        # 3. Apply causal masking
        # 4. Continue forward pass
        
        return self.model(input_batch)
    
    def combine_frames(self, enhanced_frames: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine enhanced frames using overlap-add.
        
        Args:
            enhanced_frames: List of enhanced audio frames
        
        Returns:
            Combined enhanced audio
        """
        if not enhanced_frames:
            return torch.tensor([])
        
        hop_length = self.stft_processor.hop_length
        total_length = (len(enhanced_frames) - 1) * hop_length + enhanced_frames[0].shape[1]
        
        combined = torch.zeros(enhanced_frames[0].shape[0], total_length, device=self.device)
        
        for i, frame in enumerate(enhanced_frames):
            start_idx = i * hop_length
            end_idx = start_idx + frame.shape[1]
            combined[:, start_idx:end_idx] += frame
        
        return combined
    
    def get_attention_cache_stats(self) -> Dict[str, int]:
        """Get attention cache statistics."""
        stats = {}
        for layer_name in self.attention_cache.cache:
            stats[layer_name] = self.attention_cache.get_cache_size(layer_name)
        return stats
    
    def __del__(self):
        """Cleanup hooks on destruction."""
        if self.hook_manager:
            self.hook_manager.remove_hooks()


# Utility functions for frame processing
def pad_audio_for_framing(audio: torch.Tensor, frame_size: int, hop_length: int) -> torch.Tensor:
    """Pad audio to ensure complete frame processing."""
    total_length = audio.shape[1]
    num_frames = (total_length - frame_size) // hop_length + 1
    required_length = (num_frames - 1) * hop_length + frame_size
    
    if required_length > total_length:
        pad_length = required_length - total_length
        audio = F.pad(audio, (0, pad_length))
    
    return audio


def validate_frame_processing(original_audio: torch.Tensor, 
                            processed_frames: List[torch.Tensor],
                            hop_length: int) -> Dict[str, float]:
    """
    Validate frame processing by comparing lengths and computing basic metrics.
    
    Args:
        original_audio: Original audio [channels, samples]
        processed_frames: List of processed frames
        hop_length: Hop length used
    
    Returns:
        Dictionary of validation metrics
    """
    if not processed_frames:
        return {'error': 'No processed frames'}
    
    # Reconstruct audio from frames
    reconstructed = torch.zeros_like(original_audio)
    for i, frame in enumerate(processed_frames):
        start_idx = i * hop_length
        end_idx = min(start_idx + frame.shape[1], reconstructed.shape[1])
        frame_end = end_idx - start_idx
        reconstructed[:, start_idx:end_idx] += frame[:, :frame_end]
    
    # Compute metrics
    mse = torch.mean((original_audio - reconstructed) ** 2).item()
    mae = torch.mean(torch.abs(original_audio - reconstructed)).item()
    
    return {
        'mse': mse,
        'mae': mae,
        'num_frames': len(processed_frames),
        'length_match': original_audio.shape[1] == reconstructed.shape[1]
    }
