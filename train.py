# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
<<<<<<< HEAD

=======
>>>>>>> windowing-repo/main
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from rawvae.model import VAE, loss_function
from rawvae.tests import init_test_audio
from rawvae.dataset import AudioDataset, ToTensor

import random
import numpy as np
<<<<<<< HEAD

=======
>>>>>>> windowing-repo/main
import os, sys, argparse, time
from pathlib import Path

import librosa
import soundfile as sf
import configparser
<<<<<<< HEAD
import random
import json
import matplotlib.pyplot as plt
import pdb

from torch.utils.tensorboard import SummaryWriter 

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
=======
import json
import matplotlib.pyplot as plt
import pdb
import logging
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from datetime import datetime
import librosa.display

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./default.ini', help='path to the config file')
>>>>>>> windowing-repo/main
args = parser.parse_args()

# Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
<<<<<<< HEAD
try: 
  config.read(config_path)
except FileNotFoundError:
  print('Config File Not Found at {}'.format(config_path))
  sys.exit()

# Import audio configs 
=======
try:
    config.read(config_path)
except FileNotFoundError:
    print('Config File Not Found at {}'.format(config_path))
    sys.exit()

# Import audio configs
>>>>>>> windowing-repo/main
sampling_rate = config['audio'].getint('sampling_rate')
hop_length = config['audio'].getint('hop_length')
segment_length = config['audio'].getint('segment_length')

# Dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
<<<<<<< HEAD
  raise FileNotFoundError(dataset.resolve())

run_number = config['dataset'].getint('run_number')

my_audio = dataset / 'audio'

=======
    raise FileNotFoundError(dataset.resolve())

run_number = config['dataset'].getint('run_number')
my_audio = dataset / 'audio'
>>>>>>> windowing-repo/main
test_audio = config['dataset'].get('test_dataset')
dataset_test_audio = dataset / test_audio

if not dataset_test_audio.exists():
<<<<<<< HEAD
  raise FileNotFoundError(dataset_test_audio.resolve())

generate_test = config['dataset'].get('generate_test')    
=======
    raise FileNotFoundError(dataset_test_audio.resolve())

generate_test = config['dataset'].get('generate_test')
>>>>>>> windowing-repo/main

# Training configs
epochs = config['training'].getint('epochs')
learning_rate = config['training'].getfloat('learning_rate')
batch_size = config['training'].getint('batch_size')
checkpoint_interval = config['training'].getint('checkpoint_interval')
save_best_model_after = config['training'].getint('save_best_model_after')

# Model configs
latent_dim = config['VAE'].getint('latent_dim')
n_units = config['VAE'].getint('n_units')
kl_beta = config['VAE'].getfloat('kl_beta')
<<<<<<< HEAD
=======
device = config['VAE'].get('device')
>>>>>>> windowing-repo/main

# etc
example_length = config['extra'].getint('example_length')
normalize_examples = config['extra'].getboolean('normalize_examples')
plot_model = config['extra'].getboolean('plot_model')

desc = config['extra'].get('description')
start_time = time.time()
<<<<<<< HEAD
config['extra']['start'] = time.asctime( time.localtime(start_time) )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
=======
config['extra']['start'] = time.asctime(time.localtime(start_time))

device = torch.device(device)
>>>>>>> windowing-repo/main
device_name = torch.cuda.get_device_name()
print('Device: {}'.format(device_name))
config['VAE']['device_name'] = device_name

<<<<<<< HEAD
# Create workspace
=======
# Create workspace and configure console logging
>>>>>>> windowing-repo/main
run_id = run_number
while True:
    try:
        my_runs = dataset / desc
        run_name = 'run-{:03d}'.format(run_id)
<<<<<<< HEAD
        workdir = my_runs / run_name 
        os.makedirs(workdir)

=======
        workdir = my_runs / run_name
        os.makedirs(workdir)
>>>>>>> windowing-repo/main
        break
    except OSError:
        if workdir.is_dir():
            run_id = run_id + 1
            continue
        raise

config['dataset']['workspace'] = str(workdir.resolve())

<<<<<<< HEAD
print("Workspace: {}".format(workdir))

# Create the dataset
print('creating the dataset...')
training_array = []
new_loop = True

for f in my_audio.glob('*.wav'): 
  print('adding-> %s' % f.stem)
  new_array, _ = librosa.load(f, sr=sampling_rate)

  if new_loop:
      training_array = new_array
      new_loop = False
  else:
      training_array = np.concatenate((training_array, new_array), axis=0)

total_frames = len(training_array) // segment_length
print('Total number of audio frames: {}'.format(total_frames))
config['dataset']['total_frames'] = str(total_frames)

# Create the dataset
training_dataset = AudioDataset(training_array, segment_length = segment_length, sampling_rate = sampling_rate, hop_size = hop_length, transform=ToTensor())
training_dataloader = DataLoader(training_dataset, batch_size = batch_size, shuffle=True)

print("saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
  config.write(configfile)
=======
# Set up logging to file and console
log_file = workdir / 'print_console.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler(log_file),
    logging.StreamHandler()
])

print("Workspace: {}".format(workdir))
logging.info("Workspace: {}".format(workdir))

# TensorBoard Writer
writer = SummaryWriter(workdir / "tensorboard-logging")

# Create the dataset
logging.info('Creating the dataset...')

# Get list of audio file paths
file_paths = list(my_audio.glob('*.wav'))
logging.info(f'Found {len(file_paths)} audio files.')

# Create the dataset
training_dataset = AudioDataset(file_paths, segment_length=segment_length, hop_size=hop_length, sampling_rate=sampling_rate, transform=ToTensor())

# Create the DataLoader
training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

logging.info('Starting training with AudioIterableDataset...')
logging.info(f'Total segments available: {len(training_dataset)}')
logging.info(f'Segment length: {segment_length}')
logging.info(f'Hop size: {hop_length}')

logging.info("Saving initial configs...")
config_path = workdir / 'config.ini'
with open(config_path, 'w') as configfile:
    config.write(configfile)
>>>>>>> windowing-repo/main

# Train
model_dir = workdir / "model"
checkpoint_dir = model_dir / 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

<<<<<<< HEAD
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
=======
log_dir = workdir / 'logs'
os.makedirs(log_dir, exist_ok=True)

# TensorBoard Writer
writer = SummaryWriter(log_dir=log_dir)

if generate_test:
    test_dataset, audio_log_dir = init_test_audio(workdir, test_audio, dataset_test_audio, sampling_rate, segment_length)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Neural Network
model = VAE(segment_length, n_units, latent_dim).to(device)

# Print and log the number of parameters
num_params = sum(p.numel() for p in model.parameters())
logging.info(f"Number of parameters: {num_params}")
>>>>>>> windowing-repo/main

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Some dummy variables to keep track of loss situation
<<<<<<< HEAD

=======
>>>>>>> windowing-repo/main
train_loss_prev = 1000000
best_loss = 1000000
final_loss = 1000000

for epoch in range(epochs):
<<<<<<< HEAD
  
  print('Epoch {}/{}'.format(epoch, epochs - 1))
  print('-' * 10)

  model.train()
  train_loss = 0
  
  for i, data in enumerate(training_dataloader):
    
    # data, = data
    if device == "cuda":
      data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
    
    # Log batch loss
    writer.add_scalar('Loss/Batch', loss.item(), epoch * len(training_dataloader) + i)  #🪵 Log batch loss
    
    loss.backward()
    train_loss += loss.item()
    optimizer.step()

    # Log learning rate
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(training_dataloader) + i)  #🪵 Log learning rate
  
  print('====> Epoch: {} - Total loss: {} - Average loss: {:.9f}'.format(epoch, train_loss, train_loss / len(training_dataset)))
  writer.add_scalar('Loss/train_total', train_loss, epoch) # 🪵Log the loss 
  writer.add_scalar('Loss/train_average', train_loss / len(training_dataset), epoch) # 🪵Log the loss 

  # TensorBoard_ModelParameter
  for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)
  
  if epoch % checkpoint_interval == 0 and epoch != 0: 
    print('Checkpoint - Epoch {}'.format(epoch))
    state = {
      'epoch': epoch,
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
        
      audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format( epoch))
      test_predictions_np = test_predictions.view(-1).cpu().numpy()
      sf.write( audio_out, test_predictions_np, sampling_rate)
      print('Audio examples generated: {}'.format(audio_out))
      
      #TensorBoard_ReconstructedAudio 
      writer.add_audio('Reconstructed Audio', test_predictions_np, epoch, sample_rate=sampling_rate)

    torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epoch)))
  
    if (train_loss < train_loss_prev) and (epoch > save_best_model_after):
      
      save_path = workdir.joinpath('model').joinpath('best_model.pt')
      torch.save(model, save_path)
      print('Epoch {:05d}: Saved {}'.format(epoch, save_path))
      config['training']['best_epoch'] = str(epoch)
      best_loss = train_loss

    elif (train_loss > train_loss_prev):
      print("Loss did not improve.")
  
  final_loss = train_loss

print('Last Checkpoint - Epoch {}'.format(epoch))
state = {
  'epoch': epoch,
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
    
  audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format(epochs))
  test_predictions_np = test_predictions.view(-1).cpu().numpy()
  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Audio examples generated: {}'.format(audio_out))

  sf.write( audio_out, test_predictions_np, sampling_rate)
  print('Last Audio examples generated: {}'.format(audio_out))
  #TensorBoard_ReconstructedAudio 
  writer.add_audio('Reconstructed Audio', test_predictions_np, epoch, sample_rate=sampling_rate)
=======
    logging.info('Epoch {}/{}'.format(epoch, epochs - 1))
    logging.info('-' * 10)

    model.train()
    train_loss = 0
    batch_count = 0

    for i, data in enumerate(training_dataloader):
        if i == 0:  # First batch of each epoch
            logging.info(f'First batch shape: {data.shape}')
            logging.info(f'Batch range: [{data.min():.3f}, {data.max():.3f}]')

        batch_count += 1

        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)

        # Log batch loss
        writer.add_scalar('Loss/Batch', loss.item(), epoch * len(training_dataloader) + i)

        loss.backward()

        # Gradient clipping because loss was exploding # NEW
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # NEW

        train_loss += loss.item()

        optimizer.step()

        # Log learning rate
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch * len(training_dataloader) + i)

    logging.info(f'Processed {batch_count} batches in epoch {epoch}')

    average_loss = train_loss / len(training_dataloader)  # Calculate average loss
    logging.info('====> Epoch: {} - Total loss: {} - Average loss: {:.9f}'.format(
        epoch, train_loss, average_loss))

    writer.add_scalar('Loss/train', average_loss, epoch)

    # TensorBoard_TrainingLoss
    writer.add_scalar('Loss/training', train_loss / len(training_dataloader), epoch)

    # TensorBoard_ModelParameters
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

    if epoch % checkpoint_interval == 0 and epoch != 0:
        logging.info('Checkpoint - Epoch {}'.format(epoch))

        # Save our checkpoint first
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epoch)))

        # Test and save audio examples
        if generate_test:
            init_test = True

            for iterno, test_sample in enumerate(test_dataloader):
                with torch.no_grad():
                    test_sample = test_sample.to(device)
                    test_pred = model(test_sample)[0]

                if init_test:
                    test_predictions = test_pred
                    init_test = False
                else:
                    test_predictions = torch.cat((test_predictions, test_pred), 0)

            audio_out = audio_log_dir.joinpath('test_reconst_{:05d}.wav'.format(epoch))
            test_predictions_np = test_predictions.view(-1).cpu().numpy()
            sf.write(audio_out, test_predictions_np, sampling_rate)
            logging.info('Audio examples generated: {}'.format(audio_out))

            # TensorBoard_ReconstructedAudio
            writer.add_audio('Reconstructed Audio', test_predictions_np, epoch, sample_rate=sampling_rate)

        # torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epoch)))

        # Save if best model so far
        if (train_loss < train_loss_prev) and (epoch > save_best_model_after):
            save_path = workdir.joinpath('model').joinpath('best_model.pt')
            torch.save(model, save_path)
            logging.info('Epoch {:05d}: Saved {}'.format(epoch, save_path))
            config['training']['best_epoch'] = str(epoch)
            best_loss = train_loss

        elif (train_loss > train_loss_prev):
            logging.info("Average loss did not improve.")

    final_loss = train_loss
>>>>>>> windowing-repo/main

# Save the last model as a checkpoint dict
torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epochs)))

if train_loss > train_loss_prev:
<<<<<<< HEAD
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
=======
    logging.info("Final loss was not better than the last best model.")
    logging.info("Final Loss: {}".format(final_loss))
    logging.info("Best Loss: {}".format(best_loss))

    # Save the last model using torch.save
    save_path = workdir.joinpath('model').joinpath('last_model.pt')
    torch.save(model, save_path)
    logging.info('Training Finished: Saved the last model')

else:
    logging.info("The last model is the best model.")

with open(config_path, 'w') as configfile:
    config.write(configfile)

# TensorBoard_Close
writer.close()
>>>>>>> windowing-repo/main
