# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torchaudio
from torch.utils.data import Dataset, TensorDataset, DataLoader

from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from rawvae.model import VAE, loss_function
from rawvae.tests import init_test_audio
from rawvae.dataset import IterableAudioDataset, AudioDataset, ToTensor

import random
import numpy as np

import os, sys, argparse, time
from pathlib import Path

import librosa
import soundfile as sf
import configparser
import random
import json
import matplotlib.pyplot as plt
import pdb
from itertools import islice
import pdb

from torch.utils.tensorboard import SummaryWriter 

def scan_dataset_adaptive_gpu(audio_folder, sampling_rate, hop_length, device, max_scan_files=200):
    """
    GPU-accelerated adaptive dataset scanning.
    
    Args:
        audio_folder: Path to audio folder
        sampling_rate: Target sampling rate
        hop_length: Hop length for processing
        device: CUDA device for acceleration
        max_scan_files: Max files to scan for large datasets
    
    Returns:
        total_frames: Total frames in dataset
        total_files: Total number of files
        is_estimated: Whether the count is estimated or exact
    """
    
    audio_files = list(audio_folder.glob('*.wav'))
    total_files = len(audio_files)
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_folder}")
    
    print(f"🔍 Found {total_files:,} audio files in dataset")
    
    # Decide whether to do full scan or estimation
    if max_scan_files and total_files > max_scan_files:
        print(f"📊 Large dataset detected. Scanning {max_scan_files} files for estimation...")
        scan_files = audio_files[:max_scan_files]
        is_estimated = True
    else:
        print("📊 Scanning all files for exact count...")
        scan_files = audio_files
        is_estimated = False
    
    total_frames = 0
    processed_files = 0
    
    print(f"🚀 Using GPU acceleration on {device}")
    
    for i, audio_file in enumerate(scan_files):
        try:
            # Load just the metadata for speed (no actual audio data)
            info = torchaudio.info(str(audio_file))
            
            # Calculate frames for this file
            original_frames = info.num_frames
            
            # Account for resampling
            if info.sample_rate != sampling_rate:
                resampled_frames = int(original_frames * sampling_rate / info.sample_rate)
            else:
                resampled_frames = original_frames
            
            # Account for mono conversion (frames stay the same)
            if info.num_channels > 1:
                pass  # Frames don't change, just fewer channels
            
            # Account for padding to hop_length
            if resampled_frames % hop_length != 0:
                padding = hop_length - (resampled_frames % hop_length)
                resampled_frames += padding
            
            total_frames += resampled_frames
            processed_files += 1
            
            if (i + 1) % 100 == 0:
                print(f"  ⚡ Processed {i + 1:,}/{len(scan_files):,} files...")
                
        except Exception as e:
            print(f"  ⚠️  Warning: Could not process {audio_file}: {e}")
            continue
    
    if is_estimated:
        # Scale up the estimate based on sample
        avg_frames_per_file = total_frames / processed_files
        total_frames = int(avg_frames_per_file * total_files)
        print(f"  📈 Estimated total frames: {total_frames:,} (based on {processed_files:,} files)")
    else:
        print(f"  ✅ Exact total frames: {total_frames:,}")
    
    return total_frames, total_files, is_estimated

def create_adaptive_config(config, device, max_scan_files=200):
    """
    Automatically configure training parameters based on dataset scan.
    """
    
    # Get basic parameters from config
    dataset_path = Path(config['dataset']['datapath'])
    audio_folder = dataset_path / 'audio'
    
    sampling_rate = config['audio'].getint('sampling_rate')
    hop_length = config['audio'].getint('hop_length')
    
    epochs = config['training'].getint('epochs')
    batch_size = config['training'].getint('batch_size')
    
    # Try to get desired checkpoints, default to 100
    try:
        desired_checkpoints = config['training'].getint('desired_checkpoints')
    except:
        desired_checkpoints = 100
    
    print("=" * 60)
    print("🤖 ADAPTIVE DATASET CONFIGURATION")
    print("=" * 60)
    
    # Scan the dataset
    start_time = time.time()
    total_frames, total_files, is_estimated = scan_dataset_adaptive_gpu(
        audio_folder, sampling_rate, hop_length, device, max_scan_files
    )
    scan_time = time.time() - start_time
    
    # Calculate training parameters
    total_num_frames = total_frames * epochs
    total_batches = total_num_frames // batch_size
    checkpoint_interval = max(1, total_batches // desired_checkpoints)
    
    # Update config
    config['training']['total_num_frames'] = str(total_num_frames)
    config['training']['checkpoint_interval'] = str(checkpoint_interval)
    
    # Add adaptive info to config
    if not config.has_section('adaptive_info'):
        config.add_section('adaptive_info')
    
    config['adaptive_info']['dataset_files'] = str(total_files)
    config['adaptive_info']['dataset_frames'] = str(total_frames)
    config['adaptive_info']['is_estimated'] = str(is_estimated)
    config['adaptive_info']['scan_time_seconds'] = f"{scan_time:.2f}"
    config['adaptive_info']['calculated_at'] = time.asctime()
    
    # Print results
    print(f"⏱️  Dataset scan completed in {scan_time:.2f} seconds")
    print(f"📁 Files: {total_files:,}")
    print(f"🎵 Frames per epoch: {total_frames:,}")
    print(f"🏋️  Total training frames ({epochs} epochs): {total_num_frames:,}")
    print(f"📦 Total batches: {total_batches:,}")
    print(f"🎯 Checkpoint every {checkpoint_interval:,} batches")
    print(f"📊 Estimated: {'Yes' if is_estimated else 'No'}")
    print("=" * 60)
    
    return total_num_frames, total_batches, checkpoint_interval

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./kelsey_iterable.ini', help='path to the config file')
args = parser.parse_args()

# Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()

# Import audio configs 
sampling_rate = config['audio'].getint('sampling_rate')
hop_length = config['audio'].getint('hop_length')
segment_length = config['audio'].getint('segment_length')

# Dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
  raise FileNotFoundError(dataset.resolve())

run_number = config['dataset'].getint('run_number')

my_audio = dataset / 'audio'

test_audio = config['dataset'].get('test_dataset')
dataset_test_audio = dataset / test_audio

if not dataset_test_audio.exists():
  raise FileNotFoundError(dataset_test_audio.resolve())

generate_test = config['dataset'].get('generate_test')    

# Setup device early for adaptive config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
print('🖥️  Device: {}'.format(device_name))

# ADAPTIVE CONFIGURATION - Check if we need to calculate parameters
needs_calculation = (
    not config.has_option('training', 'total_num_frames') or
    config['training']['total_num_frames'] == 'auto' or
    config['training']['total_num_frames'] == '' or
    config['training']['total_num_frames'] == '0'
)

if needs_calculation:
    print("🤖 Adaptive mode: Calculating dataset parameters...")
    total_num_frames, total_num_batches, checkpoint_interval = create_adaptive_config(config, device)
else:
    # Use existing config values
    total_num_frames = config['training'].getint('total_num_frames')
    checkpoint_interval = config['training'].getint('checkpoint_interval')
    total_num_batches = int(total_num_frames / config['training'].getint('batch_size'))
    print("📋 Using pre-configured values:")
    print(f"   Total frames: {total_num_frames:,}")
    print(f"   Total batches: {total_num_batches:,}")
    print(f"   Checkpoint interval: {checkpoint_interval:,}")

# Training configs
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')

# Model configs
latent_dim = config['VAE'].getint('latent_dim')
n_units = config['VAE'].getint('n_units')
kl_beta = config['VAE'].getfloat('kl_beta')

# etc
example_length = config['extra'].getint('example_length')
normalize_examples = config['extra'].getboolean('normalize_examples')
plot_model = config['extra'].getboolean('plot_model')

desc = config['extra'].get('description')
start_time = time.time()
config['extra']['start'] = time.asctime( time.localtime(start_time) )
config['VAE']['device_name'] = device_name

# Create workspace
run_id = run_number
while True:
    try:
        my_runs = dataset / desc
        run_name = 'run-{:03d}'.format(run_id)
        workdir = my_runs / run_name 
        os.makedirs(workdir)
        break
    except OSError:
        if workdir.is_dir():
            run_id = run_id + 1
            continue
        raise

config['dataset']['workspace'] = str(workdir.resolve())

print("📁 Workspace: {}".format(workdir))

# Set up console logging to file
console_log_path = workdir / 'console_log'

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

# Open log file and redirect stdout
log_file = open(console_log_path, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, log_file)

print("📝 Console logging started - all output will be saved to: {}".format(console_log_path))

# Save the final config (including adaptive calculations)
print("💾 Saving final configuration...")
config_path_final = workdir / 'config.ini'
with open(config_path_final, 'w') as configfile:
  config.write(configfile)

# Create the dataset
print('🎵 Creating the dataset...')

# Count total audio files for reference
audio_files = list(my_audio.glob('*.wav'))
print('Found {} audio files'.format(len(audio_files)))

# Create the dataset using IterableAudioDataset
training_dataset = IterableAudioDataset(
    audio_folder=my_audio, 
    sampling_rate=sampling_rate, 
    hop_size=hop_length,
    dtype=torch.float32,
    device=device,
    shuffle=True
)
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

# Train
model_dir = workdir / "model"
checkpoint_dir = model_dir / 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Logging
log_dir = workdir / 'logs'
os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

if generate_test:
  test_dataset, audio_log_dir = init_test_audio(workdir, test_audio, dataset_test_audio, sampling_rate, segment_length)
  test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# Neural Network
if device.type == "cuda":
  model = VAE(segment_length, n_units, latent_dim).to(device)
else:
  model = VAE(segment_length, n_units, latent_dim)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Some dummy variables to keep track of loss situation
train_loss_prev = 1000000
best_loss = 1000000
final_loss = 1000000

print("🚀 Starting training...")
print(f"   📊 Total batches to process: {total_num_batches:,}")
print(f"   🎯 Checkpoints every {checkpoint_interval:,} batches")
print(f"   🖥️  Using device: {device}")

# Training loop
model.train()
train_loss = 0
batch_id = 0

for data in islice(training_dataloader, total_num_batches):
  
  # Move data to device
  if device.type == "cuda":
    data = data.to(device)
  optimizer.zero_grad()
  recon_batch, mu, logvar = model(data)
  loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
  
  # Log batch loss
  writer.add_scalar('Loss/Batch', loss.item(), batch_id)
  print('====> Batch: {} - Loss: {:.9f}'.format(batch_id, loss.item()))

  loss.backward()
  train_loss += loss.item()
  optimizer.step()

  # Log learning rate
  writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], batch_id)

  # TensorBoard_ModelParameter
  for name, param in model.named_parameters():
    writer.add_histogram(name, param, batch_id)

  # If the checkpoint interval reached, start the process below
  if batch_id % checkpoint_interval == 0 and batch_id != 0: 
    print('🎯 Checkpoint - Batch {}'.format(batch_id))
    state = {
      'batch_id': batch_id,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
  
    if generate_test:
      
      init_test = True
      
      for iterno, test_sample in enumerate(test_dataloader):
        with torch.no_grad():
          if device.type == "cuda":
            test_sample = test_sample.to(device)
          test_pred = model(test_sample)[0]
        
        if init_test:
          test_predictions = test_pred
          init_test = False
        
        else:
          test_predictions = torch.cat([test_predictions, test_pred], 0)
        
      audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format( batch_id))
      test_predictions_np = test_predictions.view(-1).cpu().numpy()
      sf.write( audio_out, test_predictions_np, sampling_rate)
      print('🎵 Audio examples generated: {}'.format(audio_out))
      
      #TensorBoard_ReconstructedAudio 
      writer.add_audio('Reconstructed Audio', test_predictions_np, batch_id, sample_rate=sampling_rate)
  
    torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(batch_id)))
  
    if (train_loss < train_loss_prev):
      
      save_path = workdir.joinpath('model').joinpath('best_model.pt')
      torch.save(model, save_path)
      print('💾 batch_id {:05d}: Saved {}'.format(batch_id, save_path))
      config['training']['best_model'] = str(batch_id)
      best_loss = train_loss
  
    elif (train_loss > train_loss_prev):
      print("📈 Loss did not improve.")

  # This is the end of batch loop
  batch_id += 1

final_loss = train_loss

print('🏁 Last Checkpoint - batch_id {}'.format(batch_id))
state = {
  'batch_id': batch_id,
  'state_dict': model.state_dict(),
  'optimizer': optimizer.state_dict()
}

if generate_test:
      
  init_test = True
  
  for iterno, test_sample in enumerate(test_dataloader):
    with torch.no_grad():
      if device.type == "cuda":
        test_sample = test_sample.to(device)
      test_pred = model(test_sample)[0]
  
    if init_test:
      test_predictions = test_pred
      init_test = False
    
    else:
      test_predictions = torch.cat([test_predictions, test_pred], 0)
    
  audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format(total_num_batches))
  test_predictions_np = test_predictions.view(-1).cpu().numpy()
  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('🎵 Final audio examples generated: {}'.format(audio_out))

  #TensorBoard_ReconstructedAudio 
  writer.add_audio('Reconstructed Audio', test_predictions_np, batch_id, sample_rate=sampling_rate)

# Save the last model as a checkpoint dict
torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(total_num_batches)))

if train_loss > train_loss_prev:
  print("📈 Final loss was not better than the last best model.")
  print("Final Loss: {}".format(final_loss))
  print("Best Loss: {}".format(best_loss))
  
  # Save the last model using torch.save 
  save_path = workdir.joinpath('model').joinpath('last_model.pt')
  torch.save(model, save_path)
  print('💾 Training Finished: Saved the last model')

else:
  print("🏆 The last model is the best model.")

# Update final config with results
config['extra']['end'] = time.asctime(time.localtime(time.time()))
config['extra']['time_elapsed'] = str(time.time() - start_time)

with open(config_path_final, 'w') as configfile:
  config.write(configfile)

writer.close()

# Close the log file and restore stdout
sys.stdout = original_stdout
log_file.close()
print("✅ Training completed. Console log saved to: {}".format(console_log_path)) 