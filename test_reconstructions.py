#!/usr/bin/env python
"""
Enhanced test script to verify audio reconstructions from a VAE model.
This script processes longer audio files by dividing them into overlapping segments,
reconstructing each segment, and then stitching them back together.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import argparse
import logging
import configparser
from pathlib import Path

# Import the VAE model
from rawvae.model import VAE

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
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
    """Load a pre-trained or fine-tuned VAE model"""
    logging.info(f"Loading model from {model_path}")
    
    try:
        # Initialize new model
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
                else:
                    # Try loading as direct state dict
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
                logging.error("No model file found with any extension")
                sys.exit(1)
        
        # Ensure model is using float32
        model = model.float()
        logging.info("Ensuring model parameters are float32")
        
        return model.to(device)
    
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

def load_audio_file(audio_path, sampling_rate):
    """Load a full audio file for processing"""
    try:
        logging.info(f"Loading audio from {audio_path}")
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        
        # Log audio statistics
        duration = len(audio) / sampling_rate
        logging.info(f"Loaded audio file: {len(audio)} samples, {duration:.2f} seconds")
        
        return audio
    
    except Exception as e:
        logging.error(f"Error loading audio: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)

def generate_random_audio(duration_seconds, sampling_rate):
    """Generate a random audio sample of specified duration"""
    num_samples = int(duration_seconds * sampling_rate)
    logging.info(f"Generating random audio: {duration_seconds} seconds ({num_samples} samples)")
    
    # Generate white noise
    audio = np.random.uniform(-0.5, 0.5, num_samples)
    
    return audio

def segment_audio(audio, segment_length, hop_length):
    """Split audio into overlapping segments"""
    # Create hanning window for audio windowing
    window = np.hanning(segment_length)
    
    # Pad audio if needed to ensure we can extract segments
    if len(audio) < segment_length:
        audio = np.pad(audio, (0, segment_length - len(audio)), 'constant')
    
    # Calculate number of segments
    num_segments = max(1, (len(audio) - segment_length) // hop_length + 1)
    logging.info(f"Splitting audio into {num_segments} segments (length={segment_length}, hop={hop_length})")
    
    segments = []
    for i in range(num_segments):
        start = i * hop_length
        end = start + segment_length
        
        if end <= len(audio):
            segment = audio[start:end]
            # Apply window
            segment = segment * window
            segments.append(segment)
    
    return np.array(segments)

def reconstruct_segments(model, segments, device):
    """Process segments through the model to get reconstructions"""
    logging.info(f"Reconstructing {len(segments)} segments")
    
    # Convert to tensor
    segments_tensor = torch.tensor(segments, dtype=torch.float32).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process in batches to avoid memory issues
    batch_size = 32
    reconstructions = []
    
    with torch.no_grad():
        for i in range(0, len(segments), batch_size):
            batch = segments_tensor[i:i+batch_size]
            
            # Forward pass through model
            recon_batch, _, _ = model(batch)
            
            # Convert to numpy and store
            recon_batch_np = recon_batch.cpu().numpy()
            reconstructions.append(recon_batch_np)
            
            if i % 100 == 0 and i > 0:
                logging.info(f"Processed {i}/{len(segments)} segments")
    
    # Concatenate all batches
    all_reconstructions = np.concatenate(reconstructions) if len(reconstructions) > 1 else reconstructions[0]
    
    logging.info(f"Reconstruction complete: {all_reconstructions.shape}")
    return all_reconstructions

def overlap_add(segments, hop_length, original_length):
    """Combine overlapping segments using overlap-add synthesis"""
    logging.info(f"Performing overlap-add synthesis with {len(segments)} segments")
    
    segment_length = segments.shape[1]
    
    # Calculate the expected output length
    output_length = min(original_length, (len(segments) - 1) * hop_length + segment_length)
    
    # Initialize output buffer and normalization buffer
    output = np.zeros(output_length)
    norm = np.zeros(output_length)
    
    # Create hanning window for overlap-add
    window = np.hanning(segment_length)
    
    # Overlap-add segments
    for i, segment in enumerate(segments):
        pos = i * hop_length
        end = min(pos + segment_length, output_length)
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
    
    logging.info(f"Overlap-add synthesis complete: {len(output)} samples")
    return output

def save_audio_and_plot(original, reconstruction, output_dir, sampling_rate, name="audio"):
    """Save original and reconstructed audio, and plot spectrograms"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize audio for better listening experience
    if np.max(np.abs(original)) > 0:
        original_norm = original / np.max(np.abs(original))
    else:
        original_norm = original
        
    if np.max(np.abs(reconstruction)) > 0:
        recon_norm = reconstruction / np.max(np.abs(reconstruction))
    else:
        recon_norm = reconstruction
    
    # Ensure float32 for soundfile
    original_norm = original_norm.astype(np.float32)
    recon_norm = recon_norm.astype(np.float32)
    
    # Save audio files
    orig_path = os.path.join(output_dir, f"{name}_original.wav")
    recon_path = os.path.join(output_dir, f"{name}_reconstructed.wav")
    
    sf.write(orig_path, original_norm, sampling_rate)
    sf.write(recon_path, recon_norm, sampling_rate)
    
    logging.info(f"Saved original audio: {orig_path}")
    logging.info(f"Saved reconstructed audio: {recon_path}")
    
    # Calculate durations
    orig_duration = len(original) / sampling_rate
    recon_duration = len(reconstruction) / sampling_rate
    
    logging.info(f"Original audio: {len(original)} samples, {orig_duration:.2f} seconds")
    logging.info(f"Reconstructed audio: {len(reconstruction)} samples, {recon_duration:.2f} seconds")
    
    # Plot spectrograms
    plt.figure(figsize=(14, 8))
    
    # Original spectrogram
    plt.subplot(2, 1, 1)
    plt.title(f"Original Audio Spectrogram ({orig_duration:.2f}s)")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original_norm)), ref=np.max)
    librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    # Reconstruction spectrogram
    plt.subplot(2, 1, 2)
    plt.title(f"Reconstructed Audio Spectrogram ({recon_duration:.2f}s)")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(recon_norm)), ref=np.max)
    librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    
    # Save figure
    plt.tight_layout()
    spec_path = os.path.join(output_dir, f"{name}_spectrogram.png")
    plt.savefig(spec_path)
    plt.close()
    
    logging.info(f"Saved spectrogram comparison: {spec_path}")

def test_single_segment(model, segment, output_dir, sampling_rate, device):
    """Test reconstruction of a single segment (for comparison with full overlap-add)"""
    logging.info("Testing single segment reconstruction")
    
    # Convert to tensor
    segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass through model
        recon_segment, _, _ = model(segment_tensor)
        
        # Convert to numpy
        recon_segment_np = recon_segment.cpu().numpy()[0]
    
    # Save original and reconstructed segment
    save_audio_and_plot(segment, recon_segment_np, output_dir, sampling_rate, name="single_segment")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test audio reconstructions from a VAE model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model file')
    parser.add_argument('--audio_path', type=str, help='Path to audio file (optional)')
    parser.add_argument('--output_dir', type=str, default='./test_reconstructions', help='Directory to save output')
    parser.add_argument('--duration', type=float, default=3.0, help='Duration in seconds for random audio (default: 3.0)')
    parser.add_argument('--hop_ratio', type=float, default=0.25, help='Hop size as a ratio of segment length (default: 0.25)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get model parameters from config
    segment_length = config['audio'].getint('segment_length')
    sampling_rate = config['audio'].getint('sampling_rate')
    n_units = config['VAE'].getint('n_units')
    latent_dim = config['VAE'].getint('latent_dim')
    
    # Calculate hop length (default: 25% of segment length)
    hop_length = int(segment_length * args.hop_ratio)
    logging.info(f"Using segment length: {segment_length}, hop length: {hop_length}")
    
    # Set up device
    device_str = config['VAE'].get('device', 'cuda:0')
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.model_path, segment_length, n_units, latent_dim, device)
    
    # Get audio input
    if args.audio_path and os.path.exists(args.audio_path):
        # Load audio file
        audio = load_audio_file(args.audio_path, sampling_rate)
        audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    else:
        # Generate random audio
        audio = generate_random_audio(args.duration, sampling_rate)
        audio_name = f"random_{args.duration}s"
    
    # Segment the audio
    segments = segment_audio(audio, segment_length, hop_length)
    
    # Test a single segment for comparison
    test_single_segment(model, segments[0], args.output_dir, sampling_rate, device)
    
    # Reconstruct all segments
    reconstructed_segments = reconstruct_segments(model, segments, device)
    
    # Combine segments using overlap-add
    reconstructed_audio = overlap_add(reconstructed_segments, hop_length, len(audio))
    
    # Save results
    save_audio_and_plot(audio, reconstructed_audio, args.output_dir, sampling_rate, name=audio_name)
    
    # Calculate and report reconstruction error
    mse = np.mean((audio[:len(reconstructed_audio)] - reconstructed_audio) ** 2)
    logging.info(f"Reconstruction MSE: {mse:.6f}")
    
    logging.info("Test completed successfully!")

if __name__ == "__main__":
    main() 