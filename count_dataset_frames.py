#!/usr/bin/env python3
"""
Script to count total frames in an audio dataset for rawaudiovae training configuration.
"""

import librosa
import torchaudio
from pathlib import Path
import argparse

def count_dataset_frames(dataset_path, sampling_rate=44100, hop_length=128):
    """
    Count total frames in a dataset directory.
    
    Args:
        dataset_path: Path to dataset (should contain 'audio' folder)
        sampling_rate: Target sampling rate
        hop_length: Hop length for frame calculation
    
    Returns:
        total_frames: Total number of frames in the dataset
    """
    
    audio_folder = Path(dataset_path) / 'audio'
    
    if not audio_folder.exists():
        raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
    
    audio_files = list(audio_folder.glob('*.wav'))
    
    if not audio_files:
        raise FileNotFoundError(f"No .wav files found in {audio_folder}")
    
    print(f"Found {len(audio_files)} audio files")
    print("Counting frames...")
    
    total_frames = 0
    
    for i, audio_file in enumerate(audio_files):
        try:
            # Load audio file
            audio_np, audio_sr = torchaudio.load(audio_file)
            
            # Resample if needed
            if audio_sr != sampling_rate:
                audio_np = torchaudio.functional.resample(audio_np, audio_sr, sampling_rate)
            
            # Convert to mono if stereo
            if audio_np.shape[0] > 1:
                audio_np = audio_np[0:1, :]
            
            # Get length in samples
            audio_length = audio_np.shape[1]
            
            # Pad if needed (to make divisible by hop_length)
            if audio_length % hop_length != 0:
                num_zeros = hop_length - (audio_length % hop_length)
                audio_length += num_zeros
            
            # Calculate frames for this file
            file_frames = audio_length
            total_frames += file_frames
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(audio_files)} files...")
                
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    return total_frames, len(audio_files)

def main():
    parser = argparse.ArgumentParser(description='Count frames in audio dataset')
    parser.add_argument('--dataset', type=str, required=True, 
                        help='Path to dataset directory (should contain audio/ folder)')
    parser.add_argument('--sampling_rate', type=int, default=44100,
                        help='Target sampling rate (default: 44100)')
    parser.add_argument('--hop_length', type=int, default=128,
                        help='Hop length (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs for training calculation (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for calculation (default: 4096)')
    parser.add_argument('--desired_checkpoints', type=int, default=100,
                        help='Desired number of checkpoints (default: 100)')
    
    args = parser.parse_args()
    
    print(f"Analyzing dataset: {args.dataset}")
    print(f"Sampling rate: {args.sampling_rate}")
    print(f"Hop length: {args.hop_length}")
    print("-" * 50)
    
    try:
        total_frames, num_files = count_dataset_frames(
            args.dataset, 
            args.sampling_rate, 
            args.hop_length
        )
        
        print(f"\n{'='*50}")
        print(f"DATASET ANALYSIS RESULTS")
        print(f"{'='*50}")
        print(f"Number of audio files: {num_files:,}")
        print(f"Total frames in dataset: {total_frames:,}")
        
        # Calculate training parameters
        total_frames_for_training = total_frames * args.epochs
        total_batches = total_frames_for_training // args.batch_size
        checkpoint_interval = total_batches // args.desired_checkpoints
        
        print(f"\n{'='*50}")
        print(f"TRAINING CONFIGURATION")
        print(f"{'='*50}")
        print(f"For {args.epochs} epochs:")
        print(f"total_num_frames = {total_frames_for_training:,}")
        print(f"total_batches = {total_batches:,}")
        print(f"checkpoint_interval = {checkpoint_interval}")
        
        print(f"\n{'='*50}")
        print(f"CONFIG FILE VALUES")
        print(f"{'='*50}")
        print(f"[training]")
        print(f"epochs = {args.epochs}")
        print(f"total_num_frames = {total_frames_for_training}")
        print(f"batch_size = {args.batch_size}")
        print(f"checkpoint_interval = {checkpoint_interval}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 