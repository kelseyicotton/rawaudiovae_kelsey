# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
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

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
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

# Training configs
total_num_frames = config['training'].getint('total_num_frames')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')
checkpoint_interval = config['training'].getint('checkpoint_interval')
total_num_batches = int(total_num_frames / batch_size)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = torch.cuda.get_device_name()
print('Device: {}'.format(device_name))
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

print("Workspace: {}".format(workdir))

# Create the dataset
print('creating the dataset...')

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

print("saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
  config.write(configfile)

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
if device == "cuda":
  model = VAE(segment_length, n_units, latent_dim).to(device)
else:
  model = VAE(segment_length, n_units, latent_dim)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Some dummy variables to keep track of loss situation

train_loss_prev = 1000000
best_loss = 1000000
final_loss = 1000000

# Prev. EPOCH START was here

# With Iterable Dataset, there is no epoch anymore. 
model.train()
train_loss = 0
batch_id = 0

for data in islice(training_dataloader, total_num_batches):
  
  # data, = data
  if device == "cuda":
    data = data.to(device)
  optimizer.zero_grad()
  recon_batch, mu, logvar = model(data)
  loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
  
  # Log batch loss
  writer.add_scalar('Loss/Batch', loss.item(), batch_id)  #🪵 Log batch loss
  print('====> Batch: {} - Loss: {:.9f}'.format(batch_id, train_loss))

  loss.backward()
  train_loss += loss.item()
  optimizer.step()

  # Log learning rate
  writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(training_dataloader) + i)  #🪵 Log learning rate

  # TensorBoard_ModelParameter
  for name, param in model.named_parameters():
    writer.add_histogram(name, param, batch_id)

  # If the checkpoint interval reached, start the process below
  if batch_id % checkpoint_interval == 0 and batch_id != 0: 
    print('Checkpoint - Epoch {}'.format(batch_id))
    state = {
      'batch_id': batch_id,
      'state_dict': model.state_dict(),
      'optimizer': optimizer.state_dict()
    }
  
    if generate_test:
      
      init_test = True
      
      for iterno, test_sample in enumerate(test_dataloader):
        with torch.no_grad():
          if device == cuda:
            test_sample = test_sample.to(device)
          test_pred = model(test_sample)[0]
        
        if init_test:
          test_predictions = test_pred
          init_test = False
        
        else:
          test_predictions = torch.cat(test_predictions, test_pred, 0)
        
      audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format( batch_id))
      test_predictions_np = test_predictions.view(-1).cpu().numpy()
      sf.write( audio_out, test_predictions_np, sampling_rate)
      print('Audio examples generated: {}'.format(audio_out))
      
      #TensorBoard_ReconstructedAudio 
      writer.add_audio('Reconstructed Audio', test_predictions_np, batch_id, sample_rate=sampling_rate)
  
    torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(batch_id)))
  
    if (train_loss < train_loss_prev):
      
      save_path = workdir.joinpath('model').joinpath('best_model.pt')
      torch.save(model, save_path)
      print('batch_id {:05d}: Saved {}'.format(batch_id, save_path))
      config['training']['best_model'] = str(batch_id)
      best_loss = train_loss
  
    elif (train_loss > train_loss_prev):
      print("Loss did not improve.")

final_loss = train_loss

print('Last Checkpoint - batch_id {}'.format(batch_id))
state = {
  'batch_id': batch_id,
  'state_dict': model.state_dict(),
  'optimizer': optimizer.state_dict()
}

if generate_test:
      
  init_test = True
  
  for iterno, test_sample in enumerate(test_dataloader):
    with torch.no_grad():
      if device == "cuda":
        test_sample = test_sample.to(device)
      test_pred = model(test_sample)[0]
  
    if init_test:
      test_predictions = test_pred
      init_test = False
    
    else:
      test_predictions = torch.cat(test_predictions, test_pred, 0)
    
  audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format(total_num_batches))
  test_predictions_np = test_predictions.view(-1).cpu().numpy()
  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Audio examples generated: {}'.format(audio_out))

  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Last Audio examples generated: {}'.format(audio_out))
  #TensorBoard_ReconstructedAudio 
  writer.add_audio('Reconstructed Audio', test_predictions_np, batch_id, sample_rate=sampling_rate)

  # This is the end of batch loop
  batch_id += 1

# Save the last model as a checkpoint dict
torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(total_num_batches)))

if train_loss > train_loss_prev:
  print("Final loss was not better than the last best model.")
  print("Final Loss: {}".format(final_loss))
  print("Best Loss: {}".format(best_loss))
  
  # Save the last model using torch.save 
  save_path = workdir.joinpath('model').joinpath('last_model.pt')
  torch.save(model, save_path)
  print('Training Finished: Saved the last model')

else:
  print("The last model is the best model.")

with open(config_path, 'w') as configfile:
  config.write(configfile)

writer.close()
