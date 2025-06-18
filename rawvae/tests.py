import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

#from spectralvae.dataset import TestDataset, ToTensor
from rawvae.dataset import TestDataset, ToTensor

import numpy as np
import librosa
from pathlib import Path
import soundfile as sf

def init_test_audio(workdir, test_audio, my_test_audio, sampling_rate, segment_length):
  # Create a set samples to test the network as it trains

  # Create a folder called reconstructions
  audio_log_dir = workdir / 'audio_logs'
  os.makedirs(audio_log_dir, exist_ok=True)

  # List the test audio files from the dataset
  test_files = [f for f in my_test_audio.glob('*.wav')]


  with open( audio_log_dir.joinpath(test_audio+'.txt'), 'w') as test_audio_txt:
    test_audio_txt.writelines( "{}\n".format(test_file) for test_file in test_files)

  init = True
  for test in test_files:
      
    audio_full, _ = librosa.load(test, sr=sampling_rate)

    if init:
      test_dataset_audio = audio_full
      init = False
    else:
      test_dataset_audio = np.concatenate((test_dataset_audio, audio_full ),axis=0)
  
  # Create a dataloader for test dataset
  test_dataset = TestDataset(test_dataset_audio, segment_length = segment_length, sampling_rate = sampling_rate, transform=ToTensor())
  
  sf.write(audio_log_dir.joinpath('test_original.wav'), test_dataset_audio, sampling_rate)
  return test_dataset, audio_log_dir

# Windowing function for audio reconstruction
def apply_window_to_segments(segments, window_type='hann'):
    """
    Apply windowing to reconstructed audio segments for smoother concatenation
    Args:
        segments: tensor of shape [batch_size, segment_length] or [num_segments, segment_length]
        window_type: type of window ('hann', 'hamming', 'blackman')
    Returns:
        windowed_segments: segments with window applied
    """
    if segments.dim() == 1:
        segments = segments.unsqueeze(0)
    
    batch_size, segment_length = segments.shape
    
    # Create window
    if window_type == 'hann':
        window = torch.hann_window(segment_length, device=segments.device)
    elif window_type == 'hamming':
        window = torch.hamming_window(segment_length, device=segments.device)
    elif window_type == 'blackman':
        window = torch.blackman_window(segment_length, device=segments.device)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")
    
    # Apply window to each segment
    windowed_segments = segments * window.unsqueeze(0)
    
    return windowed_segments

def overlap_add_reconstruction(windowed_segments, hop_length):
    """
    Reconstruct audio from windowed segments using overlap-add method
    Args:
        windowed_segments: tensor of shape [num_segments, segment_length]
        hop_length: hop size used for segmentation
    Returns:
        reconstructed_audio: 1D tensor of reconstructed audio
    """
    num_segments, segment_length = windowed_segments.shape
    
    # Calculate output length
    output_length = (num_segments - 1) * hop_length + segment_length
    reconstructed = torch.zeros(output_length, device=windowed_segments.device)
    
    # Overlap-add
    for i, segment in enumerate(windowed_segments):
        start = i * hop_length
        end = start + segment_length
        reconstructed[start:end] += segment
    
    return reconstructed