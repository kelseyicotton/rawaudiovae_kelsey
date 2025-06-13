import os
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader

<<<<<<< HEAD
#from spectralvae.dataset import TestDataset, ToTensor
from rawvae.dataset import TestDataset, ToTensor
=======
from rawvae.dataset import TestDataset, ToTensor 
from rawvae.model import VAE # import new VAE class with the LSTM additions
>>>>>>> windowing-repo/main

import numpy as np
import librosa
from pathlib import Path
import soundfile as sf
<<<<<<< HEAD

def init_test_audio(workdir, test_audio, my_test_audio, sampling_rate, segment_length):
=======
import configparser

def init_test_audio(workdir, test_audio_name, test_audio_path, sampling_rate, segment_length):
>>>>>>> windowing-repo/main
  # Create a set samples to test the network as it trains

  # Create a folder called reconstructions
  audio_log_dir = workdir / 'audio_logs'
  os.makedirs(audio_log_dir, exist_ok=True)

  # List the test audio files from the dataset
<<<<<<< HEAD
  test_files = [f for f in my_test_audio.glob('*.wav')]


  with open( audio_log_dir.joinpath(test_audio+'.txt'), 'w') as test_audio_txt:
=======
  test_files = [f for f in test_audio_path.glob('*.wav')]

  with open(audio_log_dir.joinpath(test_audio_name+'.txt'), 'w') as test_audio_txt:
>>>>>>> windowing-repo/main
    test_audio_txt.writelines( "{}\n".format(test_file) for test_file in test_files)

  init = True
  for test in test_files:
<<<<<<< HEAD
      
=======
>>>>>>> windowing-repo/main
    audio_full, _ = librosa.load(test, sr=sampling_rate)

    if init:
      test_dataset_audio = audio_full
      init = False
    else:
      test_dataset_audio = np.concatenate((test_dataset_audio, audio_full ),axis=0)
  
<<<<<<< HEAD
  # Create a dataloader for test dataset
  test_dataset = TestDataset(test_dataset_audio, segment_length = segment_length, sampling_rate = sampling_rate, transform=ToTensor())
  
  sf.write(audio_log_dir.joinpath('test_original.wav'), test_dataset_audio, sampling_rate)
  return test_dataset, audio_log_dir
=======
  config_path = './default.ini'
  config = configparser.ConfigParser(allow_no_value=True)
  config.read(config_path)
  hop_length = config['audio'].getint('hop_length')
  
  # Store the original length for later reference
  original_length = len(test_dataset_audio)

  # Save the original full audio before any modifications
  test_audio_out = audio_log_dir / 'test_original.wav'
  sf.write(test_audio_out, test_dataset_audio, sampling_rate)
  
  # Use a simple but effective overlapping dataset
  class OverlappingTestDataset(torch.utils.data.Dataset):
    def __init__(self, audio, segment_length, hop_length, transform=None):
      self.audio = audio
      self.segment_length = segment_length
      self.hop_length = hop_length
      self.transform = transform
      
      # Calculate num_segments without truncating audio
      self.num_segments = max(0, (len(audio) - segment_length) // hop_length + 1)
      
      # Store original audio length for scaling during reconstruction
      self.original_length = len(audio)
      
      print(f"OverlappingTestDataset: Audio length: {len(audio)}, Segments: {self.num_segments}")
      print(f"Theoretical reconstruction length: {(self.num_segments - 1) * hop_length + segment_length}")
      
    def __getitem__(self, index):
      # Calculate start and end positions
      start = index * self.hop_length
      end = start + self.segment_length
      
      # Handle boundary cases by padding if needed
      if end > len(self.audio):
        segment = np.zeros(self.segment_length)
        segment_len = len(self.audio) - start
        if segment_len > 0:
          segment[:segment_len] = self.audio[start:]
      else:
        segment = self.audio[start:end]
      
      # Apply Hann window
      window = np.hanning(self.segment_length)
      segment = segment * window
      
      # Convert to float32 explicitly before transform
      segment = segment.astype(np.float32)
      
      if self.transform:
        segment = self.transform(segment)
      return segment
    
    def __len__(self):
      return self.num_segments
    
    def get_original_length(self):
      return self.original_length
    
    def get_expected_length(self):
      return (self.num_segments - 1) * self.hop_length + self.segment_length
  
  # Define a custom ToTensor to ensure float32
  class ToTensor32(object):
    def __call__(self, sample):
      return torch.from_numpy(sample).float()  # Explicitly convert to float32
  
  # Create dataset with overlapping segments
  test_dataset = OverlappingTestDataset(
    test_dataset_audio, 
    segment_length=segment_length,
    hop_length=hop_length,
    transform=ToTensor32()
  )
  
  # Store information for reconstruction
  reconstruction_info = {
    'original_length': test_dataset.get_original_length(),
    'expected_length': test_dataset.get_expected_length(),
    'hop_length': hop_length,
    'segment_length': segment_length
  }
  
  # Return test_dataset, audio_log_dir and reconstruction info
  return test_dataset, audio_log_dir, reconstruction_info
>>>>>>> windowing-repo/main
