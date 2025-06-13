#!/usr/bin/env python3
"""
Example of adaptive configuration for rawaudiovae training.
This shows how the training script could automatically calculate dataset parameters.
"""

import torch
import torchaudio
from pathlib import Path
import configparser
import time

def scan_dataset_adaptive(audio_folder, sampling_rate, hop_length, max_scan_files=None):
    """
    Adaptively scan dataset to determine total frames.
    Can do full scan or estimate from sample.
    
    Args:
        audio_folder: Path to audio folder
        sampling_rate: Target sampling rate
        hop_length: Hop length for processing
        max_scan_files: If set, only scan this many files and estimate total
    
    Returns:
        total_frames: Total frames in dataset
        total_files: Total number of files
        is_estimated: Whether the count is estimated or exact
    """
    
    audio_files = list(audio_folder.glob('*.wav'))
    total_files = len(audio_files)
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {audio_folder}")
    
    print(f"Found {total_files} audio files in dataset")
    
    # Decide whether to do full scan or estimation
    if max_scan_files and total_files > max_scan_files:
        # ESTIMATION MODE: Sample subset of files
        print(f"Large dataset detected. Scanning {max_scan_files} files for estimation...")
        scan_files = audio_files[:max_scan_files]
        is_estimated = True
    else:
        # FULL SCAN MODE: Count all files
        print("Scanning all files for exact count...")
        scan_files = audio_files
        is_estimated = False
    
    total_frames = 0
    processed_files = 0
    
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
            
            # Account for mono conversion (take max channel)
            if info.num_channels > 1:
                # Frames stay the same, just fewer channels
                pass
            
            # Account for padding to hop_length
            if resampled_frames % hop_length != 0:
                padding = hop_length - (resampled_frames % hop_length)
                resampled_frames += padding
            
            total_frames += resampled_frames
            processed_files += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(scan_files)} files...")
                
        except Exception as e:
            print(f"  Warning: Could not process {audio_file}: {e}")
            continue
    
    if is_estimated:
        # Scale up the estimate based on sample
        avg_frames_per_file = total_frames / processed_files
        total_frames = int(avg_frames_per_file * total_files)
        print(f"  Estimated total frames: {total_frames:,} (based on {processed_files} files)")
    else:
        print(f"  Exact total frames: {total_frames:,}")
    
    return total_frames, total_files, is_estimated

def create_adaptive_training_config(config, max_scan_files=200):
    """
    Automatically configure training parameters based on dataset scan.
    
    Args:
        config: ConfigParser object with basic settings
        max_scan_files: Max files to scan for large datasets (None = scan all)
    
    Returns:
        Updated config with calculated training parameters
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
    print("ADAPTIVE DATASET CONFIGURATION")
    print("=" * 60)
    
    # Scan the dataset
    start_time = time.time()
    total_frames, total_files, is_estimated = scan_dataset_adaptive(
        audio_folder, sampling_rate, hop_length, max_scan_files
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
    print(f"Dataset scan completed in {scan_time:.2f} seconds")
    print(f"Files: {total_files:,}")
    print(f"Frames per epoch: {total_frames:,}")
    print(f"Total training frames ({epochs} epochs): {total_num_frames:,}")
    print(f"Total batches: {total_batches:,}")
    print(f"Checkpoint every {checkpoint_interval} batches")
    print(f"Estimated: {'Yes' if is_estimated else 'No'}")
    print("=" * 60)
    
    return config

# Example usage in training script:
def adaptive_training_example():
    """
    Example of how this would be integrated into train_iterable.py
    """
    
    # Load config file (like normal)
    config = configparser.ConfigParser()
    config.read('kelsey_iterable.ini')
    
    # Check if we need to calculate adaptive parameters
    needs_calculation = (
        not config.has_option('training', 'total_num_frames') or
        config['training']['total_num_frames'] == 'auto' or
        config['training']['total_num_frames'] == ''
    )
    
    if needs_calculation:
        print("Adaptive mode: Calculating dataset parameters...")
        config = create_adaptive_training_config(config)
        
        # Optionally save the calculated config
        with open('kelsey_iterable_calculated.ini', 'w') as f:
            config.write(f)
        print("Saved calculated parameters to kelsey_iterable_calculated.ini")
    
    # Continue with normal training using calculated values
    total_num_frames = config['training'].getint('total_num_frames')
    checkpoint_interval = config['training'].getint('checkpoint_interval')
    
    print(f"Training with: {total_num_frames:,} total frames")
    print(f"Checkpoints every: {checkpoint_interval} batches")

if __name__ == "__main__":
    adaptive_training_example() 