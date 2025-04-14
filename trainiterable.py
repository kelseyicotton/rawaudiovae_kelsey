# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.multiprocessing as mp
from torch.multiprocessing import freeze_support
import fcntl
import errno

from rawvae.model import VAE, loss_function
from rawvae.tests import init_test_audio
from rawvae.iterabledataset import AudioDataset, ToTensor

import random
import numpy as np
import os, sys, argparse, time
from pathlib import Path

import librosa
import soundfile as sf
import configparser
import json
import matplotlib.pyplot as plt
import pdb
import logging
from torch.utils.tensorboard import SummaryWriter
from collections import Counter
from datetime import datetime
import librosa.display

def acquire_lock():
    lock_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.training.lock')
    try:
        fd = os.open(lock_file, os.O_CREAT | os.O_WRONLY)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd
    except OSError as e:
        if e.errno == errno.EACCES or e.errno == errno.EAGAIN:
            print("Another instance is already running")
            sys.exit(1)
        raise

def release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    except Exception as e:
        print(f"Error releasing lock: {e}")

def main():
    # Acquire lock at the start
    lock_fd = acquire_lock()
    
    try:
        # Add process identification logging
        pid = os.getpid()
        print(f"Starting main process with PID: {pid}")
        
        try:
            # Set multiprocessing start method
            mp.set_start_method('spawn', force=True)
        except RuntimeError as e:
            print(f"Process {pid}: Start method already set: {e}")
            pass

        # Parse arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, default='./default.ini', help='path to the config file')
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
        epochs = config['training'].getint('epochs')
        learning_rate = config['training'].getfloat('learning_rate')
        batch_size = config['training'].getint('batch_size')
        checkpoint_interval = config['training'].getint('checkpoint_interval')
        save_best_model_after = config['training'].getint('save_best_model_after')

        # Model configs
        latent_dim = config['VAE'].getint('latent_dim')
        n_units = config['VAE'].getint('n_units')
        kl_beta = config['VAE'].getfloat('kl_beta')
        device = config['VAE'].get('device')

        # etc
        example_length = config['extra'].getint('example_length')
        normalize_examples = config['extra'].getboolean('normalize_examples')
        plot_model = config['extra'].getboolean('plot_model')

        desc = config['extra'].get('description')
        start_time = time.time()
        config['extra']['start'] = time.asctime(time.localtime(start_time))

        device = torch.device(device)
        device_name = torch.cuda.get_device_name()
        print('Device: {}'.format(device_name))
        config['VAE']['device_name'] = device_name

        # Create workspace and configure console logging
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
        training_dataset = AudioDataset(file_paths, hop_size=hop_length, segment_length=segment_length, 
                                      sampling_rate=sampling_rate, transform=ToTensor(device), 
                                      device=device)

        # Create the DataLoader with CUDA-optimized settings
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1, #reduced for big dataset
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=1
        )

        logging.info('Starting training with AudioIterableDataset...')
        logging.info(f'Total segments available: {len(training_dataset)}')
        logging.info(f'Segment length: {segment_length}')
        logging.info(f'Hop size: {hop_length}')

        logging.info("Saving initial configs...")
        config_path = workdir / 'config.ini'
        with open(config_path, 'w') as configfile:
            config.write(configfile)

        # Train
        model_dir = workdir / "model"
        checkpoint_dir = model_dir / 'checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        log_dir = workdir / 'logs'
        os.makedirs(log_dir, exist_ok=True)

        if generate_test:
            test_dataset, audio_log_dir = init_test_audio(workdir, test_audio, dataset_test_audio, sampling_rate, segment_length)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Neural Network
        torch.cuda.empty_cache()  # Clear any existing GPU memory
        model = VAE(segment_length, n_units, latent_dim).to(device)
        
        # Ensure model is on the correct device
        # print(f"Model device: {next(model.parameters()).device}")
        # print(f"CUDA available: {torch.cuda.is_available()}")
        # print(f"Current CUDA device: {torch.cuda.current_device()}")
        
        # Move model to GPU explicitly
        if not next(model.parameters()).is_cuda:
            logging.warning("Model not on CUDA! Forcing move to GPU...")
            model = model.cuda()
        
        # Print and log the number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Number of parameters: {num_params}")

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Some dummy variables to keep track of loss situation
        train_loss_prev = 1000000
        best_loss = 1000000
        final_loss = 1000000

        for epoch in range(epochs):
            logging.info('Epoch {}/{}'.format(epoch, epochs - 1))
            logging.info('-' * 10)

            model.train()
            train_loss = 0
            batch_count = 0

            # Reset coverage at start of epoch
            if epoch > 0:  # Don't reset on first epoch
                training_dataset.reset_coverage()
                logging.info("Reset coverage statistics for new epoch")

            # Calculate batch limits for this epoch
            stats = training_dataset.get_processing_stats()
            if stats['segments_per_second'] > 0:  # Check if we have valid processing stats
                segments_per_second = stats['segments_per_second']
                
                # Calculate time-based limit (5 minutes worth of batches)
                time_based_limit = int(segments_per_second * 300)  # 300 seconds = 5mins
                
                # Conservative batch limit calculation based on segments processed
                max_batches = min(
                    time_based_limit,  # 20 minutes worth of batches
                    20000           # Hard cap at 20k batches
                )
                max_batches = max(max_batches, 1000)  # Ensure at least 1000 batches
                logging.info(f"Batch limit for epoch {epoch}:")
                logging.info(f"  - Time-based limit (5 min): {time_based_limit:,}")
                logging.info(f"  - Hard cap: 20,000")
                logging.info(f"  - Final limit: {max_batches:,}")
            else:
                max_batches = 20000  # Default limit for first epoch
                logging.info(f"No processing stats yet - using default batch limit of {max_batches:,} for epoch {epoch}")

            # Add warmup computation to initialize CUDA
            # logging.info("Performing CUDA warmup...")
            # warmup_tensor = torch.randn(1, 1, segment_length, device=device)
            # for _ in range(3):  # Run a few warmup iterations
            #     with torch.amp.autocast('cuda'):
            #         _ = model(warmup_tensor)
            #     torch.cuda.synchronize()
            # logging.info("CUDA warmup complete")

            for i, data in enumerate(training_dataloader):
                
                batch_count += 1 #increment first

                # Then check limit
                if batch_count >= max_batches:
                    logging.info(f"Reached batch limit of {max_batches}. Ending epoch early.")
                    break

                if i == 0:  # First batch logging
                    # logging.info(f'First batch shape: {data.shape}')
                    # logging.info(f'First batch device before transfer: {data.device}')
                    logging.info(f'Batch range: [{data.min():.3f}, {data.max():.3f}]')

                # Force immediate GPU transfer and computation
                data = data.to(device, non_blocking=False)
                torch.cuda.synchronize()
                
                if i == 0:
                    # logging.info(f'First batch device after transfer: {data.device}')
                    # logging.info(f'Model device: {next(model.parameters()).device}')
                    # Force computation and verify
                    test_compute = torch.nn.functional.relu(data).sum()
                    torch.cuda.synchronize()
                    # logging.info(f'Test computation result: {test_compute.item():.3f} on {test_compute.device}')
                    logging.info(f'Current GPU memory after first batch: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB')

                optimizer.zero_grad()
                
                # Force immediate computation
                with torch.amp.autocast('cuda'):
                    recon_batch, mu, logvar = model(data)
                    torch.cuda.synchronize()  # Sync after forward pass
                    loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
                    torch.cuda.synchronize()  # Sync after loss
                
                # Increment batch count before potential early stopping
                batch_count += 1

                # Log stats every 1000 batches
                if i % 1000 == 0:
                    logging.info(f'Batch {i}, Loss: {loss.item():.6f}')
                    # Log processing time statistics
                    training_dataset.log_processing_stats()
                    
                    if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
                        logging.debug(f'GPU Memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB')
                        logging.debug(f'GPU Memory cached: {torch.cuda.memory_reserved(device) / 1024**2:.2f} MB')
                        # logging.debug(f'Current CUDA device: {torch.cuda.current_device()}')

                # Force immediate backward pass
                loss.backward()
                torch.cuda.synchronize()  # Sync after backward
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                torch.cuda.synchronize()  # Sync after optimizer

                train_loss += loss.item()

                # # Force computation every few batches
                # if i % 5 == 0:
                #     # Force some computation to keep GPU active
                #     _ = torch.nn.functional.relu(data).sum()
                #     torch.cuda.synchronize()

            logging.info(f'Processed {batch_count} batches in epoch {epoch}')

            # Calculate average loss using actual batch count instead of dataloader length
            average_loss = train_loss / batch_count if batch_count > 0 else float('inf')
            logging.info('====> Epoch: {} - Total loss: {} - Average loss: {:.9f}'.format(
                epoch, train_loss, average_loss))

            writer.add_scalar('Loss/train', average_loss, epoch)

            # TensorBoard_TrainingLoss - use actual batch count
            writer.add_scalar('Loss/training', train_loss / batch_count if batch_count > 0 else float('inf'), epoch)

            # Update train_loss_prev for next epoch comparison
            train_loss_prev = train_loss

            # TensorBoard_ModelParameters
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, epoch)

            if epoch % checkpoint_interval == 0 and epoch != 0:
                logging.info('Checkpoint - Epoch {}'.format(epoch))

                # Save our checkpoint first
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': train_loss,
                    'best_loss': best_loss
                }
                checkpoint_path = checkpoint_dir.joinpath('ckpt_{:05d}'.format(epoch))
                torch.save(state, checkpoint_path)
                logging.info(f'Saved checkpoint: {checkpoint_path}')

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

        # Save the last model as a checkpoint dict
        torch.save(state, checkpoint_dir.joinpath('ckpt_{:05d}'.format(epochs)))

        if train_loss > train_loss_prev:
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

    finally:
        # Release lock in finally block to ensure it's always released
        release_lock(lock_fd)

if __name__ == '__main__':
    print(f"Initializing script with PID: {os.getpid()}")
    freeze_support()
    main()
    print(f"Main execution completed for PID: {os.getpid()}")

