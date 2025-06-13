#!/usr/bin/env python
# finetune_vae.py

"""
This script fine-tunes a pre-trained VAE model on a dataset of your choice.

It adapts the model to better reconstruct a specific sound profile while maintaining 
its ability to reconstruct other sounds from the original training set.
"""

import os
import sys
import argparse
import time
import configparser
import logging
from pathlib import Path
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from torch import optim
import torchaudio
import librosa
import soundfile as sf
from torch.utils.tensorboard import SummaryWriter

# Import your existing VAE model and data processing classes
from rawvae.model import VAE, loss_function
from rawvae.iterabledataset import AudioDataset, TestDataset, ToTensor

def setup_logging(log_dir):
    """Set up logging to file and console"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'finetune.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_config(config_path):
    """Load configuration from the provided config file"""
    config = configparser.ConfigParser(allow_no_value=True)
    try:
        config.read(config_path)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f'Config file not found at {config_path}')
        sys.exit(1)

def load_model(model_path, segment_length, n_units, latent_dim, device):
    """Load the pre-trained VAE model"""
    logging.info(f"Loading model from {model_path}")
    
    try:
        # Initialize new model first
        model = VAE(segment_length, n_units, latent_dim).to(device)
        
        if os.path.exists(model_path):
            # Load the saved file
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if it's a complete model or a checkpoint dictionary
            if isinstance(checkpoint, dict):
                logging.info("Loaded checkpoint dictionary")
                
                # If it contains a state_dict, load it
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    logging.info("Loaded model state from checkpoint dictionary")
                    
                    # Log epoch info if available
                    if 'epoch' in checkpoint:
                        logging.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
                    if 'loss' in checkpoint:
                        logging.info(f"Checkpoint loss: {checkpoint['loss']}")
                else:
                    # If it's a dictionary but doesn't have state_dict, it might be a direct state dict
                    try:
                        model.load_state_dict(checkpoint)
                        logging.info("Loaded model from direct state dictionary")
                    except Exception as e:
                        logging.error(f"Error loading state dictionary: {e}")
                        logging.error(f"Keys in checkpoint: {checkpoint.keys()}")
                        raise
            else:
                # If it's a complete model, use it directly
                model = checkpoint
                logging.info("Loaded complete model")
        else:
            # If the exact path doesn't exist, try with different extensions
            logging.warning(f"File not found at {model_path}, trying alternative extensions")
            found_file = False
            
            for ext in ['.pt', '.pth']:
                state_dict_path = model_path.replace('.pt', ext).replace('.pth', ext)
                if os.path.exists(state_dict_path):
                    checkpoint = torch.load(state_dict_path, map_location=device)
                    
                    # Similar handling as above
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            model.load_state_dict(checkpoint['state_dict'])
                        else:
                            try:
                                model.load_state_dict(checkpoint)
                            except Exception as e:
                                logging.error(f"Error loading from {state_dict_path}: {e}")
                                continue
                    else:
                        model = checkpoint
                    
                    logging.info(f"Loaded model from {state_dict_path}")
                    found_file = True
                    break
            
            if not found_file:
                logging.warning("No model file found with any extension, using fresh model")
        
        # Ensure model is on the correct device and using float32
        model = model.to(device)
        
        # Convert model parameters to float32 to avoid dtype mismatches
        model = model.float()
        logging.info("Ensuring model parameters are float32")
        
        return model
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        logging.error(f"Model path: {model_path}")
        # Print traceback for better debugging
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

class VoiceDataset(Dataset):
    """Dataset class for voice recordings"""
    
    def __init__(self, voice_dir, segment_length, hop_length, sampling_rate, transform=None):
        self.voice_dir = voice_dir
        self.segment_length = segment_length
        self.hop_length = hop_length
        self.sampling_rate = sampling_rate
        self.transform = transform
        
        # Create hanning window for audio windowing
        self.window = np.hanning(self.segment_length)
        
        # Get all audio files
        self.file_paths = []
        for ext in ['.wav', '.mp3', '.flac', '.ogg']:
            self.file_paths.extend(list(Path(voice_dir).glob(f'*{ext}')))
        
        logging.info(f"Found {len(self.file_paths)} voice files in {voice_dir}")
        
        # Load and segment all files
        self.segments = []
        
        for file_path in self.file_paths:
            try:
                audio, _ = librosa.load(file_path, sr=self.sampling_rate)
                
                # # Normalize audio # commented out to prevent audio distortion #nonoisebitches
                # if np.max(np.abs(audio)) > 0: # commented out to prevent audio distortion #nonoisebitches
                #     audio = audio / np.max(np.abs(audio)) # commented out to prevent audio distortion #nonoisebitches
                
                # Pad if needed to match hop size
                if len(audio) % self.hop_length != 0:
                    padding = self.hop_length - (len(audio) % self.hop_length)
                    audio = np.pad(audio, (0, padding), 'constant')
                
                # Segment the audio
                for i in range(0, len(audio) - self.segment_length + 1, self.hop_length):
                    segment = audio[i:i + self.segment_length]
                    if len(segment) == self.segment_length:
                        # Apply window
                        segment = segment * self.window
                        self.segments.append(segment)
                
                logging.info(f"Processed {file_path.name}: {len(audio)/self.sampling_rate:.2f}s, {len(audio)} samples")
                
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
        
        logging.info(f"Created dataset with {len(self.segments)} segments")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        
        if self.transform:
            segment = self.transform(segment)
        else:
            # Explicitly specify float32 to avoid dtype mismatches
            segment = torch.tensor(segment, dtype=torch.float32)
        
        # Ensure the data is float32, not float64 (double)
        if segment.dtype != torch.float32:
            segment = segment.float()
            
        return segment

def create_voice_dataset(voice_dir, config, device):
    """Create a dataset from voice recordings"""
    sampling_rate = config['audio'].getint('sampling_rate')
    hop_length = config['audio'].getint('hop_length')
    segment_length = config['audio'].getint('segment_length')
    
    logging.info(f"Creating voice dataset from {voice_dir}")
    logging.info(f"Audio config: sampling_rate={sampling_rate}, hop_length={hop_length}, segment_length={segment_length}")
    
    # Modified ToTensor transform that ensures float32 dtype
    class Float32ToTensor:
        def __init__(self, device):
            self.device = device
            
        def __call__(self, sample):
            # Convert numpy array to tensor and ensure float32 dtype
            tensor = torch.tensor(sample, dtype=torch.float32)
            return tensor.to(self.device)
    
    transform = Float32ToTensor(device)
    
    dataset = VoiceDataset(
        voice_dir=voice_dir,
        segment_length=segment_length,
        hop_length=hop_length,
        sampling_rate=sampling_rate,
        transform=transform
    )
    
    return dataset

def create_original_dataset_sample(original_data_dir, config, device, max_files=100):
    """Create a sample dataset from original training data"""
    sampling_rate = config['audio'].getint('sampling_rate')
    hop_length = config['audio'].getint('hop_length')
    segment_length = config['audio'].getint('segment_length')
    
    logging.info(f"Creating original data sample from {original_data_dir}")
    
    # Get all audio file paths
    file_paths = []
    for ext in ['.wav', '.mp3', '.flac', '.ogg']:
        file_paths.extend(list(Path(original_data_dir).glob(f'**/*{ext}')))
    
    if len(file_paths) > max_files:
        logging.info(f"Limiting to {max_files} files from {len(file_paths)} available")
        file_paths = random.sample(file_paths, max_files)
    
    # Create dataset
    return AudioDataset(
        file_paths=file_paths,
        sampling_rate=sampling_rate,
        hop_size=hop_length,
        segment_length=segment_length,
        device=device,
        transform=ToTensor(device)
    )

def create_mixed_dataset(original_dataset, voice_dataset, voice_ratio=0.7, batch_size=32):
    """
    This function is a placeholder. In practice, with our existing code structure,
    we would need to implement a custom sampler or collate function to mix the datasets.
    
    Since your IterableDataset is designed differently from the VoiceDataset,
    we'll have different handling in the training loop instead.
    """
    return None  # We'll handle mixing during training

def overlap_add(segments, hop_length, original_length=None):
    """Combine overlapping segments using overlap-add synthesis"""
    segment_length = segments.shape[1]
    
    if original_length is None:
        # Calculate expected output length
        original_length = (len(segments) - 1) * hop_length + segment_length
    
    # Initialize output buffer and normalization buffer
    output = np.zeros(original_length)
    norm = np.zeros(original_length)
    
    # Create hanning window for overlap-add
    window = np.hanning(segment_length)
    
    # Overlap-add segments
    for i, segment in enumerate(segments):
        pos = i * hop_length
        end = min(pos + segment_length, original_length)
        seg_end = end - pos
        
        # Apply window again to ensure smooth transitions
        windowed_segment = segment[:seg_end] * window[:seg_end]
        
        # Add to output buffer
        output[pos:end] += windowed_segment
        
        # Add window weights to normalization buffer
        norm[pos:end] += window[:seg_end]
    
    # Normalize by the summed window weights
    mask = norm > 1e-10
    output[mask] /= norm[mask]
    
    return output

def plot_reconstructions(model, test_samples, output_dir, epoch, config):
    """
    Generate and save audio reconstructions from the model to monitor progress.
    
    Args:
        model: The VAE model
        test_samples: Audio samples to reconstruct
        output_dir: Directory to save reconstructions
        epoch: Current epoch number
        config: Configuration object with audio settings
    """
    sampling_rate = config['audio'].getint('sampling_rate')
    segment_length = config['audio'].getint('segment_length')
    hop_length = int(segment_length * 0.25)  # Use 25% of segment length as hop size
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        # Process the test samples through the model
        recon_samples, _, _ = model(test_samples)
        
        # Convert to numpy for saving
        originals = test_samples.cpu().numpy()
        reconstructions = recon_samples.cpu().numpy()
        
        # Log reconstruction statistics
        logging.info(f"Reconstruction stats - Original range: [{np.min(originals):.3f}, {np.max(originals):.3f}], "
                     f"Reconstructed range: [{np.min(reconstructions):.3f}, {np.max(reconstructions):.3f}]")
        
        # Check for NaN or infinity values
        if np.any(np.isnan(reconstructions)) or np.any(np.isinf(reconstructions)):
            logging.warning("Found NaN or Inf values in reconstructions. Replacing with zeros.")
            reconstructions = np.nan_to_num(reconstructions, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Save individual segments
        for i in range(min(5, originals.shape[0])):
            # Ensure audio is within [-1, 1] range for saving
            original_audio = originals[i].copy()
            if np.max(np.abs(original_audio)) > 0:
                original_audio = np.clip(original_audio / np.max(np.abs(original_audio)), -1.0, 1.0)
            
            recon_audio = reconstructions[i].copy()
            if np.max(np.abs(recon_audio)) > 0:
                recon_audio = np.clip(recon_audio / np.max(np.abs(recon_audio)), -1.0, 1.0)
            
            # Convert to float32 for soundfile
            original_audio = original_audio.astype(np.float32)
            recon_audio = recon_audio.astype(np.float32)
            
            # Check if audio contains only zeros
            if np.all(original_audio == 0):
                logging.warning(f"Original audio sample {i} contains only zeros!")
            if np.all(recon_audio == 0):
                logging.warning(f"Reconstructed audio sample {i} contains only zeros!")
            
            # Save original audio
            orig_path = os.path.join(output_dir, f"epoch_{epoch}_original_{i}.wav")
            try:
                sf.write(orig_path, original_audio, sampling_rate)
                logging.info(f"Saved original audio to {orig_path}, shape: {original_audio.shape}")
            except Exception as e:
                logging.error(f"Error saving original audio: {e}")
            
            # Save reconstruction
            recon_path = os.path.join(output_dir, f"epoch_{epoch}_recon_{i}.wav")
            try:
                sf.write(recon_path, recon_audio, sampling_rate)
                logging.info(f"Saved reconstructed audio to {recon_path}, shape: {recon_audio.shape}")
            except Exception as e:
                logging.error(f"Error saving reconstructed audio: {e}")
            
            # Plot spectrograms
            plt.figure(figsize=(12, 6))
            
            # Original spectrogram
            plt.subplot(1, 2, 1)
            plt.title("Original")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(original_audio)), ref=np.max)
            librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            # Reconstruction spectrogram
            plt.subplot(1, 2, 2)
            plt.title("Reconstruction")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(recon_audio)), ref=np.max)
            librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            # Save figure
            plt.tight_layout()
            spec_path = os.path.join(output_dir, f"epoch_{epoch}_spectrogram_{i}.png")
            plt.savefig(spec_path)
            plt.close()
            logging.info(f"Saved spectrogram to {spec_path}")
        
        # Calculate how many segments we need to create 10 seconds of audio
        # For 10 seconds at 44.1kHz, we need 10 * 44100 = 441,000 samples
        target_samples = 10 * sampling_rate
        
        # Calculate how many segments we need to loop
        samples_per_segment_with_hop = hop_length  # Each new segment adds hop_length samples
        num_loops_needed = target_samples // samples_per_segment_with_hop + 1
        
        logging.info(f"Creating long audio reconstruction (target: 10 seconds, {target_samples} samples)")
        logging.info(f"Need approximately {num_loops_needed} segment repetitions")
        
        # Create longer audio by repeating and concatenating segments using overlap-add
        # We'll use all available segments and repeat them as needed
        available_segments = originals.shape[0]
        
        # Create arrays that will hold all the repeated segments
        num_total_segments = num_loops_needed
        logging.info(f"Creating array with {num_total_segments} segments")
        
        # Create extended arrays by repeating the segments we have
        extended_originals = np.zeros((num_total_segments, segment_length))
        extended_reconstructions = np.zeros((num_total_segments, segment_length))
        
        for i in range(num_total_segments):
            segment_idx = i % available_segments  # Loop through available segments
            extended_originals[i] = originals[segment_idx]
            extended_reconstructions[i] = reconstructions[segment_idx]
        
        # Apply overlap-add to create long audio streams
        logging.info(f"Applying overlap-add to create long audio streams")
        long_original = overlap_add(extended_originals, hop_length)
        long_recon = overlap_add(extended_reconstructions, hop_length)
        
        # Calculate actual duration
        original_duration_sec = len(long_original) / sampling_rate
        recon_duration_sec = len(long_recon) / sampling_rate
        
        logging.info(f"Generated long audio examples - Original: {original_duration_sec:.2f}s, Reconstructed: {recon_duration_sec:.2f}s")
        
        # Ensure audio is within [-1, 1] range for saving
        if np.max(np.abs(long_original)) > 0:
            long_original = long_original / np.max(np.abs(long_original))
        if np.max(np.abs(long_recon)) > 0:
            long_recon = long_recon / np.max(np.abs(long_recon))
        
        # Convert to float32 for soundfile
        long_original = long_original.astype(np.float32)
        long_recon = long_recon.astype(np.float32)
        
        # Save longer audio files
        long_orig_path = os.path.join(output_dir, f"epoch_{epoch}_long_original.wav")
        long_recon_path = os.path.join(output_dir, f"epoch_{epoch}_long_recon.wav")
        
        try:
            sf.write(long_orig_path, long_original, sampling_rate)
            sf.write(long_recon_path, long_recon, sampling_rate)
            logging.info(f"Saved long audio examples: {long_orig_path} and {long_recon_path}")
            
            # Plot spectrograms for longer audio
            plt.figure(figsize=(14, 8))
            
            # Original spectrogram
            plt.subplot(2, 1, 1)
            plt.title(f"Long Original Audio ({original_duration_sec:.2f}s)")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(long_original)), ref=np.max)
            librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            # Reconstruction spectrogram
            plt.subplot(2, 1, 2)
            plt.title(f"Long Reconstructed Audio ({recon_duration_sec:.2f}s)")
            D = librosa.amplitude_to_db(np.abs(librosa.stft(long_recon)), ref=np.max)
            librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            
            # Save figure
            plt.tight_layout()
            long_spec_path = os.path.join(output_dir, f"epoch_{epoch}_long_spectrogram.png")
            plt.savefig(long_spec_path)
            plt.close()
            logging.info(f"Saved long spectrogram to {long_spec_path}")
            
        except Exception as e:
            logging.error(f"Error saving longer audio examples: {e}")
    
    model.train()
    logging.info(f"Saved reconstructions for epoch {epoch}")

def freeze_layers(model, num_frozen_layers):
    """Freeze a specified number of layers in the model"""
    if num_frozen_layers <= 0:
        return
    
    total_layers = len(list(model.parameters()))
    layers_to_freeze = min(num_frozen_layers, total_layers)
    
    logging.info(f"Freezing {layers_to_freeze} layers out of {total_layers} total layers")
    
    # Get all named parameters
    named_params = list(model.named_parameters())
    
    # Freeze the first layers_to_freeze layers
    for i, (name, param) in enumerate(named_params):
        if i < layers_to_freeze:
            param.requires_grad = False
            logging.info(f"Freezing layer: {name}")
        else:
            param.requires_grad = True

def finetune_model(model, voice_loader, validation_loader, output_dir, writer, 
                  config, epochs=50, frozen_layers=0, use_original_data=False, 
                  original_loader=None, voice_ratio=0.7, learning_rate=None):
    """
    Fine-tune the VAE model on voice data.
    
    Args:
        model: The pre-trained VAE model
        voice_loader: DataLoader for voice samples
        validation_loader: DataLoader for validation
        output_dir: Directory to save results
        writer: TensorBoard writer
        config: Configuration object
        epochs: Number of epochs to fine-tune
        frozen_layers: Number of layers to freeze during training
        use_original_data: Whether to mix in original data
        original_loader: DataLoader for original data
        voice_ratio: Ratio of voice data to use in each epoch
        learning_rate: Override learning rate (if None, use config value)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get configuration parameters
    segment_length = config['audio'].getint('segment_length')
    kl_beta = config['VAE'].getfloat('kl_beta')
    
    # Set learning rate
    if learning_rate is None:
        learning_rate = config['training'].getfloat('learning_rate') * 0.1  # Lower learning rate for fine-tuning
    
    # Freeze specified layers
    if frozen_layers > 0:
        freeze_layers(model, frozen_layers)
    
    # Ensure model is in float32 mode
    model = model.float()
    
    # Create optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Get batch size
    batch_size = config['training'].getint('batch_size')
    
    # Training loop
    best_loss = float('inf')
    best_epoch = 0
    
    logging.info(f"Starting fine-tuning for {epochs} epochs with learning rate {learning_rate}")
    logging.info(f"Using voice ratio: {voice_ratio:.1f}")
    
    # Set model to training mode
    model.train()
    
    # Define reconstruction samples for visualization
    # Get a batch from validation loader
    if validation_loader:
        for val_batch in validation_loader:
            # Ensure visualization samples are float32
            val_batch = val_batch.float()
            # Get more samples (15-20) so we can create longer reconstructions
            vis_samples = val_batch[:20].to(next(model.parameters()).device)
            logging.info(f"Using {len(vis_samples)} validation samples for reconstruction monitoring")
            break
    
    for epoch in range(epochs):
        logging.info(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        batch_count = 0
        
        # Reset voice and original data iterators if needed
        if use_original_data and original_loader:
            voice_iter = iter(voice_loader)
            original_iter = iter(original_loader)
        
        # Training loop
        for batch_idx, voice_batch in enumerate(voice_loader):
            # If using mixed data, combine with original data
            if use_original_data and original_loader:
                if random.random() < voice_ratio:
                    # Use voice data
                    data = voice_batch
                else:
                    # Use original data
                    try:
                        data = next(original_iter)
                    except StopIteration:
                        # Reset original data iterator if exhausted
                        original_iter = iter(original_loader)
                        data = next(original_iter)
            else:
                # Use only voice data
                data = voice_batch
            
            # Ensure the data is float32 before moving to device
            data = data.float()
            
            # Move data to device
            data = data.to(next(model.parameters()).device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Calculate loss
            loss = loss_function(recon_batch, data, mu, logvar, kl_beta, segment_length)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logging.info(f"Batch {batch_idx}/{len(voice_loader)}, Loss: {loss.item():.6f}")
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        logging.info(f"Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.6f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/finetune_train', avg_epoch_loss, epoch)
        
        # Generate reconstructions
        if epoch % 5 == 0 or epoch == epochs - 1:
            recon_dir = os.path.join(output_dir, 'reconstructions')
            plot_reconstructions(model, vis_samples, recon_dir, epoch, config)
        
        # Save checkpoint
        save_path = os.path.join(output_dir, f'vae_checkpoint_epoch_{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, save_path)
        logging.info(f"Saved checkpoint at {save_path}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_epoch = epoch + 1
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            torch.save(model, best_model_path)
            logging.info(f"New best model saved at {best_model_path}")
    
    logging.info(f"Fine-tuning complete. Best model was from epoch {best_epoch} with loss {best_loss:.6f}")
    return model, best_loss, best_epoch

def progressive_finetuning(model, voice_dataset, validation_dataset, 
                          output_dir, writer, config, use_original_data=False, 
                          original_dataset=None, voice_ratio=0.7):
    """
    Implement progressive fine-tuning in multiple stages.
    
    Args:
        model: The pre-trained VAE model
        voice_dataset: Dataset of voice samples
        validation_dataset: Dataset for validation
        output_dir: Directory to save results
        writer: TensorBoard writer
        config: Configuration object
        use_original_data: Whether to use original data
        original_dataset: Sample of original training data (optional)
        voice_ratio: Ratio of voice to original data
    
    Returns:
        The fine-tuned model and best performance metrics
    """
    logging.info("Starting progressive fine-tuning...")
    
    # Get configuration parameters from config file
    has_finetune_config = 'finetune' in config
    
    # Get stage epochs from config if available
    if has_finetune_config:
        stage1_epochs = config['finetune'].getint('stage1_epochs', 15)
        stage2_epochs = config['finetune'].getint('stage2_epochs', 25)
        stage3_epochs = config['finetune'].getint('stage3_epochs', 10)
        learning_rate_decay = config['finetune'].getfloat('learning_rate_decay', 0.1)
    else:
        stage1_epochs = 15
        stage2_epochs = 25
        stage3_epochs = 10
        learning_rate_decay = 0.1
    
    # Get base learning rate from config
    base_learning_rate = config['training'].getfloat('learning_rate')
    
    # Calculate learning rates for each stage
    stage1_lr = base_learning_rate * learning_rate_decay
    stage2_lr = base_learning_rate * learning_rate_decay * 0.1
    stage3_lr = base_learning_rate * learning_rate_decay * 0.01
    
    logging.info(f"Progressive fine-tuning configuration:")
    logging.info(f"  - Stage 1: {stage1_epochs} epochs, LR: {stage1_lr:.6f}, Voice ratio: 0.6")
    logging.info(f"  - Stage 2: {stage2_epochs} epochs, LR: {stage2_lr:.6f}, Voice ratio: 0.8")
    logging.info(f"  - Stage 3: {stage3_epochs} epochs, LR: {stage3_lr:.6f}, Voice ratio: 0.9")
    
    # Ensure model is in float32 mode
    model = model.float()
    logging.info("Ensuring model is using float32 precision")
    
    # Get batch size from config
    batch_size = config['training'].getint('batch_size')
    
    # Create data loaders
    voice_loader = DataLoader(
        voice_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    if validation_dataset:
        validation_loader = DataLoader(
            validation_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            drop_last=False
        )
    else:
        validation_loader = None
    
    if use_original_data and original_dataset:
        original_loader = DataLoader(
            original_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
    else:
        original_loader = None
    
    # Stage 1: Initial adaptation with frozen encoder
    logging.info("Stage 1: Initial adaptation (frozen encoder)")
    stage1_dir = os.path.join(output_dir, 'stage1')
    os.makedirs(stage1_dir, exist_ok=True)
    
    # In first stage, freeze the encoder (first half of the model)
    # For your VAE, this corresponds roughly to freezing fc1, fc21, and fc22
    model_params = list(model.named_parameters())
    encoder_params = ['fc1.', 'fc21.', 'fc22.'] 
    
    # Count parameters before freezing
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Freeze encoder parameters
    for name, param in model.named_parameters():
        if any(name.startswith(ep) for ep in encoder_params):
            param.requires_grad = False
    
    # Count parameters after freezing
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    logging.info(f"Stage 1 - Frozen parameters: {frozen_params:,} / {total_params:,} total parameters")
    
    # First stage: higher learning rate, focus more on original data to prevent catastrophic forgetting
    stage1_model, stage1_loss, stage1_epoch = finetune_model(
        model=model,
        voice_loader=voice_loader,
        validation_loader=validation_loader,
        output_dir=stage1_dir,
        writer=writer,
        config=config,
        epochs=stage1_epochs,  # From config
        frozen_layers=0,  # Already manually frozen above
        use_original_data=use_original_data,
        original_loader=original_loader,
        voice_ratio=0.6,  # Less voice focus in first stage
        learning_rate=stage1_lr  # From config
    )
    
    # Stage 2: Refinement with all layers unfrozen
    logging.info("Stage 2: Refinement (all layers unfrozen)")
    stage2_dir = os.path.join(output_dir, 'stage2')
    os.makedirs(stage2_dir, exist_ok=True)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Recount trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"Stage 2 - Trainable parameters: {trainable_params:,} / {total_params:,} total parameters")
    
    # Second stage: lower learning rate, focus more on voice data
    stage2_model, stage2_loss, stage2_epoch = finetune_model(
        model=model,
        voice_loader=voice_loader,
        validation_loader=validation_loader,
        output_dir=stage2_dir,
        writer=writer,
        config=config,
        epochs=stage2_epochs,  # From config
        frozen_layers=0,
        use_original_data=use_original_data,
        original_loader=original_loader,
        voice_ratio=0.8,  # More voice focus in second stage
        learning_rate=stage2_lr  # From config
    )
    
    # Final stage: Specialized voice fine-tuning (mostly voice data)
    logging.info("Stage 3: Specialized voice fine-tuning")
    stage3_dir = os.path.join(output_dir, 'stage3')
    os.makedirs(stage3_dir, exist_ok=True)
    
    # Third stage: very low learning rate, heavy voice focus
    stage3_model, stage3_loss, stage3_epoch = finetune_model(
        model=model,
        voice_loader=voice_loader,
        validation_loader=validation_loader,
        output_dir=stage3_dir,
        writer=writer,
        config=config,
        epochs=stage3_epochs,  # From config
        frozen_layers=0,
        use_original_data=use_original_data,
        original_loader=original_loader,
        voice_ratio=0.9,  # Almost exclusively voice data
        learning_rate=stage3_lr  # From config
    )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_finetuned_model.pt')
    torch.save(model, final_model_path)
    logging.info(f"Saved final fine-tuned model to {final_model_path}")
    
    # Calculate and report best loss
    best_loss = min(stage1_loss, stage2_loss, stage3_loss)
    best_stage = 1
    if stage2_loss < stage1_loss and stage2_loss <= stage3_loss:
        best_stage = 2
    elif stage3_loss < stage1_loss and stage3_loss < stage2_loss:
        best_stage = 3
    
    logging.info(f"Progressive fine-tuning complete:")
    logging.info(f"  - Stage 1 loss: {stage1_loss:.6f}")
    logging.info(f"  - Stage 2 loss: {stage2_loss:.6f}")
    logging.info(f"  - Stage 3 loss: {stage3_loss:.6f}")
    logging.info(f"  - Best loss: {best_loss:.6f} (Stage {best_stage})")
    
    # Return final model and best performance
    return model, best_loss

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fine-tune VAE model on new data')
    parser.add_argument('--config', type=str, default='default-alvis.ini', help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to pre-trained model')
    
    # Hardcoded default output directory - modify this path to your preferred location
    default_output_dir = "/mimer/NOBACKUP/groups/x_kelco_musai_2024/finetuned_models"
    
    # Make output_dir optional since we have a hardcoded default
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to save output (overrides default location)')
    parser.add_argument('--finetune_name', type=str, default=None,
                        help='Custom name for the fine-tuning run (used in directory naming)')
    
    # The following arguments are now optional as they can be specified in the config
    parser.add_argument('--voice_dir', type=str, help='Directory containing voice recordings (overrides config)')
    parser.add_argument('--use_original_data', action='store_true', help='Whether to mix in original training data (overrides config)')
    parser.add_argument('--original_data_dir', type=str, help='Directory containing original training data (overrides config)')
    parser.add_argument('--progressive', action='store_true', help='Use progressive fine-tuning (overrides config)')
    parser.add_argument('--epochs', type=int, help='Number of epochs for standard fine-tuning (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration first so we can access description and other naming info
    config = load_config(args.config)
    
    # Determine output directory - use command line if provided, otherwise use hardcoded default
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create a timestamped and descriptive subdirectory within the default location
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        
        # Get model identifier from the model path
        try:
            model_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_path)))
        except:
            model_name = "unknown_model"
            
        # Get description from config if available
        model_description = ""
        if 'extra' in config and 'description' in config['extra']:
            model_description = config['extra'].get('description')
            # Clean the description to make it filesystem-friendly
            model_description = model_description.replace(' ', '_').replace('/', '-')
            model_description = ''.join(c for c in model_description if c.isalnum() or c in '_-')
            if model_description:
                model_description = f"_{model_description}"
        
        # Get fine-tune dataset name
        finetune_dataset_name = ""
        if args.finetune_name:
            finetune_dataset_name = f"_{args.finetune_name}"
        elif 'finetune' in config and 'finetune_dataset' in config['finetune']:
            # Extract the last part of the path as the dataset name
            finetune_path = config['finetune'].get('finetune_dataset')
            finetune_dataset_name = os.path.basename(finetune_path.rstrip('/'))
            finetune_dataset_name = f"_{finetune_dataset_name}"
        
        # Construct directory name with all components
        output_dir = os.path.join(
            default_output_dir, 
            f"{model_name}{model_description}_finetuned{finetune_dataset_name}_{timestamp}"
        )
        logging.info(f"Using default output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    setup_logging(output_dir)
    
    # Start time
    start_time = time.time()
    logging.info(f"Fine-tuning started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Using model: {args.model_path}")
    logging.info(f"Output directory: {output_dir}")
    
    # Get fine-tuning dataset path from config or command line
    if args.voice_dir:
        finetune_data_dir = args.voice_dir
        logging.info(f"Using fine-tuning dataset from command line: {finetune_data_dir}")
    elif 'finetune' in config and 'finetune_dataset' in config['finetune']:
        finetune_data_dir = config['finetune'].get('finetune_dataset')
        logging.info(f"Using fine-tuning dataset from config: {finetune_data_dir}")
    else:
        logging.error("No fine-tuning dataset specified in config or command line")
        sys.exit(1)
    
    # Get other fine-tuning parameters from config or command line
    if args.use_original_data:
        use_original_data = True
    elif 'finetune' in config and 'use_original_data' in config['finetune']:
        use_original_data = config['finetune'].getboolean('use_original_data')
    else:
        use_original_data = False
    
    if args.original_data_dir:
        original_data_dir = args.original_data_dir
    elif use_original_data and 'dataset' in config:
        # Use the main dataset path if no specific original data path is specified
        original_data_dir = os.path.join(config['dataset'].get('datapath'), 'audio')
    else:
        original_data_dir = None
    
    if args.progressive:
        use_progressive = True
    elif 'finetune' in config and 'progressive' in config['finetune']:
        use_progressive = config['finetune'].getboolean('progressive')
    else:
        use_progressive = False
    
    if args.epochs:
        epochs = args.epochs
    elif 'finetune' in config and 'epochs' in config['finetune']:
        epochs = config['finetune'].getint('epochs')
    else:
        epochs = 50  # Default
    
    # Get original data ratio from config
    if 'finetune' in config and 'original_data_ratio' in config['finetune']:
        original_data_ratio = config['finetune'].getfloat('original_data_ratio')
    else:
        original_data_ratio = 0.3  # Default
    
    voice_ratio = 1.0 - original_data_ratio  # Voice ratio is complement of original data ratio
    
    # Get device from config
    device_str = config['VAE'].get('device')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load model parameters from config
    segment_length = config['audio'].getint('segment_length')
    n_units = config['VAE'].getint('n_units')
    latent_dim = config['VAE'].getint('latent_dim')
    
    # Log fine-tuning settings
    logging.info("Fine-tuning settings:")
    logging.info(f"  - Fine-tuning dataset: {finetune_data_dir}")
    logging.info(f"  - Use original data: {use_original_data}")
    if use_original_data:
        logging.info(f"  - Original data directory: {original_data_dir}")
        logging.info(f"  - Voice ratio: {voice_ratio:.2f}")
    logging.info(f"  - Progressive fine-tuning: {use_progressive}")
    logging.info(f"  - Epochs: {epochs}")
    
    # Load pre-trained model
    model = load_model(args.model_path, segment_length, n_units, latent_dim, device)
    
    # Create TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    
    # Create datasets
    voice_dataset = create_voice_dataset(finetune_data_dir, config, device)
    
    # Split voice dataset into training and validation (80/20 split)
    dataset_size = len(voice_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size
    
    # Use random_split to create the splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        voice_dataset, [train_size, val_size]
    )
    
    logging.info(f"Voice dataset split: {train_size} training, {val_size} validation samples")
    
    # Load original dataset if specified
    original_dataset = None
    if use_original_data and original_data_dir:
        original_dataset = create_original_dataset_sample(
            original_data_dir, config, device, max_files=100
        )
    
    # Create data loaders
    batch_size = config['training'].getint('batch_size')
    
    voice_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True
    )
    
    validation_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    if original_dataset:
        original_loader = DataLoader(
            original_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
    else:
        original_loader = None
    
    # Fine-tune the model
    if use_progressive:
        logging.info("Using progressive fine-tuning approach")
        
        final_model, best_loss = progressive_finetuning(
            model=model,
            voice_dataset=train_dataset,
            validation_dataset=val_dataset,
            output_dir=output_dir,
            writer=writer,
            config=config,
            use_original_data=use_original_data,
            original_dataset=original_dataset,
            voice_ratio=voice_ratio
        )
    else:
        logging.info("Using standard fine-tuning approach")
        final_model, best_loss, best_epoch = finetune_model(
            model=model,
            voice_loader=voice_loader,
            validation_loader=validation_loader,
            output_dir=output_dir,
            writer=writer,
            config=config,
            epochs=epochs,
            frozen_layers=0,
            use_original_data=use_original_data,
            original_loader=original_loader,
            voice_ratio=voice_ratio
        )
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    torch.save(final_model, final_model_path)
    logging.info(f"Saved final model to {final_model_path}")
    
    # End time
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f"Fine-tuning completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    logging.info(f"Best loss: {best_loss:.6f}")
    logging.info(f"Final model saved to {final_model_path}")
    
    # Write fine-tuning results to config
    if 'finetune_results' not in config:
        config.add_section('finetune_results')
    
    config['finetune_results']['best_loss'] = str(best_loss)
    config['finetune_results']['training_time'] = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    config['finetune_results']['completion_date'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Save updated config
    config_output_path = os.path.join(output_dir, 'finetune_config.ini')
    with open(config_output_path, 'w') as configfile:
        config.write(configfile)
    
    # Close TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main() 