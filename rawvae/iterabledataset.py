from itertools import chain, islice, cycle
from torch.utils.data import IterableDataset, DataLoader
import random
import numpy as np
import torch, torchaudio
import librosa
import logging
import os
import time
import math
from multiprocessing import Value, Array, Lock
import pathlib
from pathlib import Path
import json

class AudioDataset(IterableDataset):
    """
    This is the main class. It's designed to load and process audio files in streaming fashion. 
    We like this because it allows us to train on large datasets that don't fit in memory!
    #chonkyboiswelcomehere

    # Iterable Dataset class structure source:
    Source: https://medium.com/speechmatics/how-to-build-a-streaming-dataloader-with-pytorch-a66dd891d9dd

    The current loading mechanism shuffles the audio file list, but not the audio windows! #timeseriesdatababe 
    """

    def __init__(self, file_paths, sampling_rate, hop_size, segment_length, device, shuffle=True, transform=None):
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.segment_length = segment_length
        self.device = device
        self.shuffle = shuffle
        self.file_paths = file_paths
        self.transform = transform
        self.device = device

        self.file_paths_list = [file_path for file_path in self.file_paths if file_path.suffix == '.wav']
        self.num_files = len(self.file_paths_list)
        
        # Initialize shared counters using multiprocessing.Value
        self._total_processing_time = Value('d', 0.0)
        self._total_files_processed = Value('i', 0)
        self._total_segments_processed = Value('i', 0)
        
        # Initialize file coverage tracking
        self._initialize_tracking()
        
        # Create a lock for thread-safe operations
        self._lock = Lock()
        
        # Log initial setup
        logging.info(f"Initialized dataset with {self.num_files} files")

    def _initialize_tracking(self):
        """Initialize file coverage tracking using simple dictionaries"""
        self.file_coverage = {}
        for fp in self.file_paths_list:
            self.file_coverage[str(fp.resolve())] = {
                'count': 0,
                'last_used': None,
                'total_segments': 0,
                'segments_used': 0,
                'total_processing_time': 0.0,
                'num_processed': 0,
                'avg_processing_time': 0.0
            }

    def _update_coverage(self, file_key, processing_time, segments):
        """Thread-safe update of coverage statistics"""
        with self._lock:
            # Update file-specific stats
            if file_key not in self.file_coverage:
                logging.warning(f"Creating new coverage entry for {file_key}")
                self.file_coverage[file_key] = {
                    'count': 0,
                    'last_used': None,
                    'total_segments': 0,
                    'segments_used': 0,
                    'total_processing_time': 0.0,
                    'num_processed': 0,
                    'avg_processing_time': 0.0
                }
            
            coverage_data = self.file_coverage[file_key]
            coverage_data['count'] += 1
            coverage_data['last_used'] = time.time()
            if coverage_data['total_segments'] == 0:
                coverage_data['total_segments'] = len(segments)
            coverage_data['segments_used'] += len(segments)
            coverage_data['total_processing_time'] += processing_time
            coverage_data['num_processed'] += 1
            coverage_data['avg_processing_time'] = (
                coverage_data['total_processing_time'] / 
                coverage_data['num_processed']
            )
            
            # Update global counters
            self._total_processing_time.value += processing_time
            self._total_files_processed.value += 1
            self._total_segments_processed.value += len(segments)

    def get_processing_stats(self):
        """Returns statistics about file processing times."""
        with self._lock:
            stats = {
                'total_processing_time': self._total_processing_time.value,
                'total_files_processed': self._total_files_processed.value,
                'total_segments_processed': self._total_segments_processed.value,
            }
            
            if stats['total_files_processed'] > 0:
                stats['avg_time_per_file'] = stats['total_processing_time'] / stats['total_files_processed']
                stats['avg_segments_per_file'] = stats['total_segments_processed'] / stats['total_files_processed']
                stats['avg_time_per_segment'] = stats['total_processing_time'] / stats['total_segments_processed']
                stats['segments_per_second'] = stats['total_segments_processed'] / stats['total_processing_time']
                stats['files_per_second'] = stats['total_files_processed'] / stats['total_processing_time']
            else:
                stats.update({
                    'avg_time_per_file': 0,
                    'avg_segments_per_file': 0,
                    'avg_time_per_segment': 0,
                    'segments_per_second': 0,
                    'files_per_second': 0
                })
        
        return stats

    def reset_coverage(self):
        """Reset coverage statistics for a new epoch"""
        with self._lock:
            # Reset global counters
            self._total_processing_time.value = 0.0
            self._total_files_processed.value = 0
            self._total_segments_processed.value = 0
            
            # Reset file-specific stats while preserving total_segments
            for file_key in self.file_coverage:
                total_segments = self.file_coverage[file_key]['total_segments']
                self.file_coverage[file_key] = {
                    'count': 0,
                    'last_used': None,
                    'total_segments': total_segments,  # Preserve this value
                    'segments_used': 0,
                    'total_processing_time': 0.0,
                    'num_processed': 0,
                    'avg_processing_time': 0.0
                }
            
            logging.debug("Coverage statistics reset for new epoch")

    def get_coverage_stats(self):
        """Returns statistics about file coverage during training."""
        with self._lock:
            stats = {
                'total_files': self.num_files,
                'files_used': len([f for f in self.file_coverage.values() if f['count'] > 0]),
                'total_segments': sum(f['total_segments'] for f in self.file_coverage.values()),
                'segments_used': sum(f['segments_used'] for f in self.file_coverage.values())
            }
            
            if stats['total_segments'] > 0:
                stats['coverage_percentage'] = (stats['segments_used'] / stats['total_segments']) * 100
            else:
                stats['coverage_percentage'] = 0
                
        return stats

    def shuffled_data_list(self):
        return random.sample(self.file_paths_list, len(self.file_paths_list))

    def process_data(self, audio): # Pad audio to fit the hop size, because we use a sliding window and hop_size matters
        # Normalize audio to [-1, 1] range
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
            
        # Pad if needed
        if len(audio) % self.hop_size != 0:
            num_zeros = self.hop_size - (len(audio) % self.hop_size)
            # logging.warning(f"Padding audio with {num_zeros} zeros to match hop size {self.hop_size}")
            audio = np.pad(audio, (0, num_zeros), 'constant', constant_values=(0, 0))
        return audio
    
    def apply_window(self, segments): # 🆕
        """
        Apply Hann window to audio segments for better audio reconstructions
        """

        window = np.hanning(self.segment_length) # 🆕

        windowed_segments = [seg * window for seg in segments] # 🆕

        return windowed_segments # 🆕
    
    def log_processing_stats(self):
        """Log processing time statistics."""
        stats = self.get_processing_stats()
        if stats['total_files_processed'] > 0:
            logging.info("\nProcessing Time Statistics:")
            logging.info(f"Total files processed: {stats['total_files_processed']:,}")
            logging.info(f"Total segments processed: {stats['total_segments_processed']:,}")
            logging.info(f"Total processing time: {stats['total_processing_time']:.2f}s")
            logging.info(f"Average time per file: {stats['avg_time_per_file']:.3f}s")
            logging.info(f"Average time per segment: {stats['avg_time_per_segment']*1000:.2f}ms")
            logging.info(f"Throughput: {stats['segments_per_second']:.1f} segments/s ({stats['files_per_second']:.2f} files/s)")
            
            if 'fastest_file' in stats:
                logging.info(f"\nFile Processing Times:")
                logging.info(f"Fastest file: {stats['fastest_file']} ({stats['fastest_time']*1000:.1f}ms)")
                logging.info(f"Slowest file: {stats['slowest_file']} ({stats['slowest_time']*1000:.1f}ms)")

    """
    def log_coverage(self):
        # Commented out since we're not using coverage tracking
        with self._lock:
            stats = self.get_coverage_stats()
            logging.info("\nCoverage Statistics:")
            logging.info(f"Files used: {stats['files_used']} / {stats['total_files']} ({stats['coverage_percentage']:.1f}%)")
            
            # Get sorted files by usage count
            sorted_files = sorted(
                [(k, v['count']) for k, v in self.file_coverage.items()],
                key=lambda x: x[1],
                reverse=True
            )
            
            if sorted_files:
                logging.info("\nTop 5 most used files:")
                for file_path, count in sorted_files[:5]:
                    logging.info(f"{Path(file_path).name}: {count} times")
                
                logging.info("\nBottom 5 least used files:")
                for file_path, count in sorted_files[-5:]:
                    logging.info(f"{Path(file_path).name}: {count} times")
            
            total_segments = stats['total_segments']
            used_segments = stats['segments_used']
            logging.info(f"\nSegment Coverage: {used_segments:,} / {total_segments:,} ({(used_segments/total_segments)*100:.1f}%)")
    """

    def get_stream(self, file_paths_list):
        """
        This is our generator function
        It loads an audio file, processes it, and then yields segments of the audio based on the segment_length
        We use itertools.chain.from_iterable to flatten the list of lists of segments into a single list of segments
        This is what we iterate over in our training loop, and it allows us to train on an infinite stream of audio data
        """

        def load_and_process(file_path): # Load audio file and process it
            start_time = time.time()
            
            # Convert file_path to string and normalize for consistent key usage
            file_key = str(file_path.resolve())
            
            # Debug logging for file path handling
            logging.debug(f"Processing file: {file_key}")
            logging.debug(f"Available keys: {list(self.file_coverage.keys())}")
            
            audio, _ = librosa.load(file_path, sr=self.sampling_rate)
            audio = self.process_data(audio)
            segments = [audio[i:i+self.segment_length] for i in range(0, len(audio), self.hop_size)]
            segments = [seg for seg in segments if len(seg) == self.segment_length] #ensure segments are all same length
            

            # Apply windowing to segments # 🆕
            segments = self.apply_window(segments) # 🆕

            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Debug logging for coverage tracking
            if file_key not in self.file_coverage:
                logging.warning(f"File key not found in coverage tracking: {file_key}")
                # Add it to tracking if missing
                self.file_coverage[file_key] = {
                    'count': 0,
                    'last_used': None,
                    'total_segments': 0,
                    'segments_used': 0,
                    'total_processing_time': 0.0,
                    'num_processed': 0,
                    'avg_processing_time': 0.0
                }
            
            # Update coverage and processing time tracking
            self._update_coverage(file_key, processing_time, segments)
            
            # Convert segments to tensors (on CPU)
            segments = [torch.tensor(seg, dtype=torch.float32).unsqueeze(0) for seg in segments]

            # Use logging instead of print for consistent output handling
            if self.file_coverage[file_key]['count'] == 1:  # Only log first time file is processed
                logging.info(f"First load of file: {file_path.name}, Audio shape: {audio.shape}, Segments: {len(segments)}")
            
            return segments

        return chain.from_iterable(map(load_and_process, cycle(file_paths_list)))

    def __iter__(self):
        """
        This is the function that returns an iterator over the audio segments.
        Handles multiple workers by splitting the file list among workers.
        """
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()
        
        # If we have multiple workers, split the file list
        if worker_info is not None:
            # Split workload
            per_worker = int(math.ceil(len(self.file_paths_list) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.file_paths_list))
            
            # Get this worker's file list
            file_paths = self.file_paths_list[iter_start:iter_end]
            logging.info(f"Worker {worker_id}: Processing {len(file_paths)} files ({iter_start} to {iter_end-1})")
        else:
            file_paths = self.file_paths_list
            logging.info(f"Single worker mode: Processing all {len(file_paths)} files")
        
        # Create the stream with the appropriate file list
        # Important: Use the worker's file_paths for cycling, but shuffle it first if needed
        if self.shuffle:
            file_paths = random.sample(file_paths, len(file_paths))
        
        stream = self.get_stream(cycle(file_paths))
        for item in stream:
            yield item  # Keep data on CPU, let DataLoader handle device placement

    def __len__(self):
        """
        Returns a theoretical length based on file sizes.
        For IterableDataset with cycle, this is just an estimate used for progress bars.
        We avoid loading audio files and just use file sizes to approximate.
        """
        if not hasattr(self, '_approx_length'):
            total_size = 0
            # Use file sizes to approximate (44.1kHz * 2 bytes per sample)
            bytes_per_second = self.sampling_rate * 2
            
            for file_path in self.file_paths_list:
                total_size += os.path.getsize(file_path)
            
            # Approximate number of samples (file_size / 2 since 16-bit audio)
            total_samples = (total_size // 2)
            
            # Calculate segments based on hop_size
            self._approx_length = total_samples // self.hop_size
            
            logging.info(f"Estimated total segments: {self._approx_length} (based on file sizes)")
        
        return self._approx_length


class ToTensor(object):
    """Convert ndarrays in sample to Tensors and move to specified device."""
    
    def __init__(self, device):
        self.device = device

    def __call__(self, sample):
        return torch.from_numpy(sample).to(self.device)

class TestDataset(torch.utils.data.Dataset):
    """
    This is the main class that calculates the spectrogram and returns the
    spectrogram, audio pair.
    """

    def __init__(self, audio_np, segment_length, sampling_rate, transform=None):
        
        self.transform = transform
        self.sampling_rate = sampling_rate
        self.segment_length = segment_length
        
        if len(audio_np) % segment_length != 0: # Pad audio to fit segment length
            num_zeros = segment_length - (len(audio_np) % segment_length)
            # logging.warning(f"Padding audio with {num_zeros} zeros to match segment length {segment_length}")
            audio_np = np.pad(audio_np, (0, num_zeros), 'constant', constant_values=(0,0))

        self.audio_np = audio_np
        
    def __getitem__(self, index):
        
        # Take segment
        seg_start = index * self.segment_length
        seg_end = (index * self.segment_length) + self.segment_length
        sample = self.audio_np[ seg_start : seg_end ]
        
        if self.transform:
            sample = self.transform(sample)
        
        sample_tensor = torch.tensor(sample, dtype=torch.float32) # Ensure float32 for PyTorch compatibility

        # Log sample information at debug level
        logging.debug(f"TestDataset sample - Index: {index}, Shape: {sample_tensor.shape}, Type: {type(sample_tensor)}")

        return sample_tensor

    def __len__(self):
        return len(self.audio_np) // self.segment_length