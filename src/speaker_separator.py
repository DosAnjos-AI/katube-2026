"""
Speaker separation module to create individual audio files per speaker.
"""
import pandas as pd
import numpy as np
from pydub import AudioSegment
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

from config import Config

logger = logging.getLogger(__name__)

class SpeakerSeparator:
    def __init__(self, min_duration: float = 2.0, max_gap: float = 1.5, max_segment_duration: float = 20.0):
        self.min_duration = min_duration
        self.max_gap = max_gap  # Maximum gap between segments to merge
        self.max_segment_duration = max_segment_duration
        self.sample_rate = Config.SAMPLE_RATE
    
    def load_diarization_dataframe(self, rttm_path: Path) -> pd.DataFrame:
        """Load diarization results from RTTM file."""
        try:
            # Define RTTM columns
            columns = [
                "TYPE", "FILENAME", "CHANNEL", "START", "DURATION",
                "ORTHO", "SUBTYPE", "SPEAKER", "CONFIDENCE", "MISC"
            ]
            
            # Read RTTM file
            rows = []
            with open(rttm_path, 'r') as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) >= len(columns):
                        rows.append(fields[:len(columns)])
            
            if not rows:
                logger.warning(f"No valid data in RTTM file: {rttm_path}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            
            # Convert numeric columns
            df["START"] = pd.to_numeric(df["START"], errors="coerce")
            df["DURATION"] = pd.to_numeric(df["DURATION"], errors="coerce")
            df["END"] = df["START"] + df["DURATION"]
            
            # Remove invalid rows
            df = df.dropna(subset=["START", "DURATION"])
            
            logger.info(f"Loaded {len(df)} segments from RTTM")
            return df
            
        except Exception as e:
            logger.error(f"Error loading RTTM file {rttm_path}: {e}")
            return pd.DataFrame()
    
    def merge_consecutive_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge consecutive segments from the same speaker.
        
        Args:
            df: DataFrame with diarization results
            
        Returns:
            DataFrame with merged segments
        """
        if df.empty:
            return df
        
        # Sort by speaker and start time
        df = df.sort_values(['SPEAKER', 'START']).reset_index(drop=True)
        
        merged_segments = []
        
        for speaker in df['SPEAKER'].unique():
            speaker_df = df[df['SPEAKER'] == speaker].copy()
            
            i = 0
            while i < len(speaker_df):
                current_start = speaker_df.iloc[i]['START']
                current_end = speaker_df.iloc[i]['END']
                
                # Merge consecutive segments
                while (i + 1 < len(speaker_df) and 
                       speaker_df.iloc[i + 1]['START'] - current_end <= self.max_gap and
                       current_end - current_start <= self.max_segment_duration):
                    i += 1
                    current_end = speaker_df.iloc[i]['END']
                
                # Only keep segments longer than minimum duration
                duration = current_end - current_start
                if duration >= self.min_duration:
                    merged_segments.append({
                        'START': current_start,
                        'END': current_end,
                        'DURATION': duration,
                        'SPEAKER': speaker
                    })
                
                i += 1
        
        merged_df = pd.DataFrame(merged_segments)
        if not merged_df.empty:
            merged_df = merged_df.sort_values('START').reset_index(drop=True)
            logger.info(f"Merged {len(df)} segments into {len(merged_df)} segments")
        
        return merged_df
    
    def enhance_audio_segment(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio enhancement to improve quality."""
        try:
            # Normalize
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # Apply high-pass filter to remove low-frequency noise
            nyquist = self.sample_rate / 2
            low_cutoff = 80.0  # Hz
            high_cutoff = min(8000.0, nyquist - 100)  # Hz
            
            # Design filters
            b_high, a_high = butter(4, low_cutoff / nyquist, btype='high')
            b_low, a_low = butter(4, high_cutoff / nyquist, btype='low')
            
            # Apply filters
            audio = filtfilt(b_high, a_high, audio)
            audio = filtfilt(b_low, a_low, audio)
            
            # Normalize again
            audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            return audio
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            return audio
    
    
    
    def extract_speaker_segments(self, audio_path: Path, df: pd.DataFrame, output_dir: Path, 
                                enhance: bool = True, segment_offset: float = 0.0) -> Dict[str, List[Path]]:
        """
        Extract audio segments for each speaker.
        
        Args:
            audio_path: Path to original audio file
            df: DataFrame with speaker segments
            output_dir: Output directory
            enhance: Whether to apply audio enhancement
            segment_offset: Absolute start time of this segment in the original audio (in seconds)
            
        Returns:
            Dictionary mapping speaker IDs to lists of segment file paths
        """
        if df.empty:
            logger.warning("No segments to extract")
            return {}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded audio: {len(audio) / sr:.2f}s at {sr}Hz")
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            return {}
        
        # Create speaker directories
        speaker_files = {}
        
        for speaker in df['SPEAKER'].unique():
            speaker_dir = output_dir / f"speaker_{speaker}"
            speaker_dir.mkdir(exist_ok=True)
            speaker_files[speaker] = []
        
        # Extract segments
        audio_name = audio_path.stem
        
        for idx, row in df.iterrows():
            try:
                speaker = row['SPEAKER']
                start_time = row['START']  # Relative to segment
                end_time = row['END']      # Relative to segment
                
                # Calculate absolute timestamps
                absolute_start = segment_offset + start_time
                absolute_end = segment_offset + end_time
                
                # Convert to sample indices (relative to this segment)
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                
                # Validate bounds
                start_sample = max(0, start_sample)
                end_sample = min(len(audio), end_sample)
                
                if start_sample >= end_sample:
                    logger.warning(f"Invalid segment bounds: {start_sample}-{end_sample}")
                    continue
                
                # Extract segment
                segment_audio = audio[start_sample:end_sample]
                
                # Apply enhancement if requested
                if enhance:
                    segment_audio = self.enhance_audio_segment(segment_audio)
                
                # Create filename with absolute timestamps
                filename = f"{audio_name}_{speaker}_{absolute_start:.2f}_{absolute_end:.2f}.{Config.AUDIO_FORMAT}"
                output_path = output_dir / f"speaker_{speaker}" / filename
                
                # Save segment
                sf.write(output_path, segment_audio, sr)
                speaker_files[speaker].append(output_path)
                
                logger.debug(f"Extracted {speaker}: relative={start_time:.2f}-{end_time:.2f}s, absolute={absolute_start:.2f}-{absolute_end:.2f}s -> {filename}")
                
            except Exception as e:
                logger.error(f"Failed to extract segment {idx}: {e}")
                continue
        
        # Log statistics
        for speaker, files in speaker_files.items():
            total_duration = sum(
                sf.info(file).duration for file in files
            )
            logger.info(f"Speaker {speaker}: {len(files)} segments, {total_duration:.2f}s total")
        
        return speaker_files
    
    def create_speaker_compilation(self, speaker_files: Dict[str, List[Path]], 
                                 output_dir: Path, max_duration: float = 300.0) -> Dict[str, Path]:
        """
        Create compilation files for each speaker (useful for STT).
        
        Args:
            speaker_files: Dictionary of speaker ID to file paths
            output_dir: Output directory
            max_duration: Maximum duration per compilation file
            
        Returns:
            Dictionary mapping speaker IDs to compilation file paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        compilation_files = {}
        
        for speaker, files in speaker_files.items():
            if not files:
                continue
            
            try:
                # Sort files by timestamp (extracted from filename)
                sorted_files = sorted(files, key=lambda f: self._extract_start_time(f))
                
                # Create compilation
                compilation_audio = []
                current_duration = 0.0
                compilation_idx = 0
                
                for file_path in sorted_files:
                    # Load segment
                    audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
                    segment_duration = len(audio) / sr
                    
                    # Check if we need to start a new compilation
                    if current_duration + segment_duration > max_duration and compilation_audio:
                        # Save current compilation
                        self._save_compilation(compilation_audio, speaker, compilation_idx, 
                                             output_dir, compilation_files)
                        compilation_idx += 1
                        compilation_audio = []
                        current_duration = 0.0
                    
                    # Add segment
                    compilation_audio.extend(audio)
                    current_duration += segment_duration
                    
                    # Add small pause between segments (0.2 seconds)
                    pause = np.zeros(int(0.2 * sr))
                    compilation_audio.extend(pause)
                    current_duration += 0.2
                
                # Save final compilation if not empty
                if compilation_audio:
                    # Convert list to numpy array
                    compilation_array = np.array(compilation_audio, dtype=np.float32)
                    self._save_compilation([compilation_array], speaker, compilation_idx, 
                                         output_dir, compilation_files)
                
            except Exception as e:
                logger.error(f"Failed to create compilation for speaker {speaker}: {e}")
        
        return compilation_files
    
    def _extract_start_time(self, file_path: Path) -> float:
        """Extract start time from filename."""
        try:
            # Filename format: {audio_name}_{speaker}_{start_time}_{end_time}.flac
            parts = file_path.stem.split('_')
            if len(parts) >= 3:
                return float(parts[-2])
        except:
            pass
        return 0.0
    
    def _save_compilation(self, audio_data: List[np.ndarray], speaker: str, idx: int, 
                         output_dir: Path, compilation_files: Dict[str, Path]):
        """Save compilation audio file."""
        try:
            # Check if audio_data is empty or contains only empty arrays
            if not audio_data or all(len(arr) == 0 for arr in audio_data):
                logger.warning(f"No audio data to save for speaker {speaker}")
                return
                
            # Filter out empty arrays before concatenation
            valid_audio_data = [arr for arr in audio_data if len(arr) > 0]
            if not valid_audio_data:
                logger.warning(f"No valid audio data to save for speaker {speaker}")
                return
                
            compilation_audio = np.concatenate(valid_audio_data)
            filename = f"speaker_{speaker}_compilation_{idx:02d}.{Config.AUDIO_FORMAT}"
            output_path = output_dir / filename
            
            sf.write(output_path, compilation_audio, self.sample_rate)
            
            if speaker not in compilation_files:
                compilation_files[speaker] = []
            if not isinstance(compilation_files[speaker], list):
                compilation_files[speaker] = [compilation_files[speaker]]
            compilation_files[speaker].append(output_path)
            
            duration = len(compilation_audio) / self.sample_rate
            logger.info(f"Saved compilation: {filename} ({duration:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to save compilation: {e}")
    
    def process_audio_file(self, audio_path: Path, rttm_path: Path, output_dir: Path, 
                          enhance: bool = True, create_compilations: bool = True,
                          segment_offset: float = 0.0) -> Dict[str, any]:
        """
        Complete processing pipeline for a single audio file.
        
        Args:
            audio_path: Path to audio file
            rttm_path: Path to RTTM diarization file
            output_dir: Output directory
            enhance: Apply audio enhancement
            create_compilations: Create speaker compilation files
            segment_offset: Absolute start time of this segment in the original audio (in seconds)
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing {audio_path.name}")
        
        # Load diarization results
        df = self.load_diarization_dataframe(rttm_path)
        if df.empty:
            return {'error': 'No diarization data available'}
        
        # Merge consecutive segments
        merged_df = self.merge_consecutive_segments(df)
        if merged_df.empty:
            return {'error': 'No segments after merging'}
        
        # Extract speaker segments with offset
        speaker_files = self.extract_speaker_segments(
            audio_path, merged_df, output_dir, enhance, segment_offset
        )
        
        results = {
            'speaker_files': speaker_files,
            'num_speakers': len(speaker_files),
            'total_segments': sum(len(files) for files in speaker_files.values()),
            'segment_offset': segment_offset  # Include offset in results
        }
        
        # Create compilations if requested
        if create_compilations:
            compilation_files = self.create_speaker_compilation(speaker_files, output_dir)
            results['compilation_files'] = compilation_files
        
        return results


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    separator = SpeakerSeparator()
    # results = separator.process_audio_file(
    #     Path("input.flac"), 
    #     Path("input.rttm"), 
    #     Path("output/speakers/")
    # )
    # print(f"Separation results: {results}")
