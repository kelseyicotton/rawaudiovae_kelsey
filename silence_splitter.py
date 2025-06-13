# silence_splitter.py

# This script splits audio files into multiple smaller files, all random length.
# It creates segments based on random time intervals and saves them as separate files.

import os
import random
import subprocess
import argparse
from pathlib import Path
import ffmpeg

def split_audio_file(input_file, output_dir, min_length=1, max_length=45, max_segments=1825):
    """
    Split a single audio file into segments of random durations and save to output directory
    
    Args:
        input_file (str): Path to the input audio file
        output_dir (str): Directory to save the output segments
        min_length (int): Minimum segment length in seconds
        max_length (int): Maximum segment length in seconds
        max_segments (int): Maximum number of segments to create
    
    Returns:
        int: Number of segments created
    """
    # Get the base filename without extension for naming segments
    base_filename = os.path.splitext(os.path.basename(input_file))[0]
    
    # Get the duration of the input file
    try:
        probe = ffmpeg.probe(input_file)
        duration = float(probe['format']['duration'])
        print(f"Processing file: {input_file}")
        print(f"Total audio duration: {duration:.2f} seconds")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
        return 0
    
    # Generate random split points
    current_time = 0
    segment_number = 1
    split_points = []
    
    while current_time < duration and len(split_points) < max_segments:
        # Generate a random segment length
        segment_length = random.uniform(min_length, max_length)
        
        # Make sure we don't exceed the total duration
        end_time = min(current_time + segment_length, duration)
        
        split_points.append((current_time, end_time))
        current_time = end_time
        segment_number += 1
        
        # Check if we've reached the limit of segments
        if len(split_points) >= max_segments:
            print(f"Reached maximum segment limit of {max_segments}")
            break
    
    # Extract each segment
    segments_created = 0
    for i, (start_time, end_time) in enumerate(split_points):
        segment_length = end_time - start_time
        # Use the input filename as the base for naming segments
        output_file = os.path.join(output_dir, f"{base_filename}_{i+1:04d}.wav")
        
        print(f"Creating segment {i+1}: {start_time:.2f}s to {end_time:.2f}s (duration: {segment_length:.2f}s)")
        
        try:
            # Use ffmpeg to extract the segment
            (
                ffmpeg
                .input(input_file, ss=start_time, t=segment_length)
                .output(output_file)
                .run(quiet=True, overwrite_output=True)
            )
            segments_created += 1
        except Exception as e:
            print(f"Error creating segment {i+1}: {e}")
    
    print(f"Completed {input_file}: Created {segments_created} segments")
    return segments_created

def process_folder(input_folder, output_dir, min_length=1, max_length=45, max_segments=1825):
    """
    Process all audio files in a folder and split them into segments
    
    Args:
        input_folder (str): Path to the folder containing input audio files
        output_dir (str): Directory to save the output segments
        min_length (int): Minimum segment length in seconds
        max_length (int): Maximum segment length in seconds
        max_segments (int): Maximum number of segments to create per file
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files in the input folder
    audio_extensions = ['.wav', '.mp3', '.flac', '.aac', '.ogg', '.m4a']
    audio_files = []
    
    for extension in audio_extensions:
        audio_files.extend(list(Path(input_folder).glob(f'*{extension}')))
    
    if not audio_files:
        print(f"No audio files found in {input_folder}")
        return
    
    print(f"Found {len(audio_files)} audio files in {input_folder}")
    
    # Process each audio file
    total_segments = 0
    for file_path in audio_files:
        segments = split_audio_file(
            str(file_path), 
            output_dir,
            min_length, 
            max_length, 
            max_segments
        )
        total_segments += segments
    
    print(f"\nProcessing complete!")
    print(f"Total audio files processed: {len(audio_files)}")
    print(f"Total segments created: {total_segments}")
    print(f"Output directory: {output_dir}")

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Split audio files into random length segments')
    parser.add_argument('input_path', help='Path to the input audio file or folder containing audio files')
    parser.add_argument('output_dir', help='Directory to save the output segments')
    parser.add_argument('--min-length', type=float, default=1, help='Minimum segment length in seconds')
    parser.add_argument('--max-length', type=float, default=45, help='Maximum segment length in seconds')
    parser.add_argument('--max-segments', type=int, default=1825, help='Maximum number of segments to create per file')
    
    args = parser.parse_args()
    
    # Check if input_path is a file or a directory
    if os.path.isfile(args.input_path):
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        # Process single file
        split_audio_file(args.input_path, args.output_dir, args.min_length, args.max_length, args.max_segments)
    elif os.path.isdir(args.input_path):
        # Process folder
        process_folder(args.input_path, args.output_dir, args.min_length, args.max_length, args.max_segments)
    else:
        print(f"Error: {args.input_path} is not a valid file or directory")

if __name__ == "__main__":
    main()





