"""
Intelligent audio segmentation that preserves word boundaries and speech patterns.
"""
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import webrtcvad
from scipy.signal import find_peaks
import pyloudnorm as pyln

from config import Config
from naming_utils import extract_base_name, generate_standard_name

logger = logging.getLogger(__name__)

class AudioSegmenter:
    def __init__(self, min_duration: float = None, max_duration: float = None):
        self.min_duration = min_duration or Config.SEGMENT_MIN_DURATION
        self.max_duration = max_duration or Config.SEGMENT_MAX_DURATION
        self.sample_rate = Config.SAMPLE_RATE
        self.overlap = Config.SEGMENT_OVERLAP
        
        # VAD for speech detection - optimized for high-quality segmentation
        self.vad = webrtcvad.Vad(2)  # Level 2: Balanced sensitivity for better segmentation
        
        # Enhanced VAD parameters for natural speech pause detection
        self.vad_frame_duration = 30  # 30ms frames for better stability
        self.silence_threshold_db = -35  # Balanced silence detection for natural pauses
        self.min_silence_duration = 0.2  # Detect even short natural pauses
        self.max_silence_duration = 2.0  # Allow longer natural breaks (end of sentences, thoughts)
        self.speech_continuity_threshold = 0.3  # Balanced threshold for natural speech flow
        
        # Energy-based segmentation parameters
        self.energy_threshold = 0.01  # Minimum energy threshold
        self.energy_window_size = 0.1  # 100ms energy analysis window
        self.spectral_centroid_threshold = 1000  # Hz threshold for speech detection
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio using loudness normalization."""
        try:
            meter = pyln.Meter(self.sample_rate)
            loudness = meter.integrated_loudness(audio)
            # Normalize to -23 LUFS (broadcast standard)
            audio_normalized = pyln.normalize.loudness(audio, loudness, -23.0)
            return audio_normalized
        except:
            # Fallback to simple normalization
            return audio / np.max(np.abs(audio))
    
    def analyze_energy(self, audio: np.ndarray) -> np.ndarray:
        """Analyze energy levels in audio for better segmentation."""
        window_size = int(self.energy_window_size * self.sample_rate)
        hop_size = window_size // 2
        
        energy = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i + window_size]
            energy.append(np.mean(window ** 2))
        
        return np.array(energy)
    
    def analyze_spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Analyze spectral centroid to detect speech characteristics."""
        hop_length = 512
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate, hop_length=hop_length)[0]
        return spectral_centroids
    
    def detect_speech_regions(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Detect speech regions using combined VAD, energy, and spectral analysis."""
        # Get VAD results
        vad_frames = self.detect_speech_activity(audio)
        
        # Get energy analysis
        energy = self.analyze_energy(audio)
        energy_window_size = int(self.energy_window_size * self.sample_rate)
        
        # Get spectral analysis
        spectral_centroids = self.analyze_spectral_centroid(audio)
        hop_length = 512
        
        # Convert to sample indices
        speech_regions = []
        in_speech = False
        speech_start = 0
        
        for i, is_speech in enumerate(vad_frames):
            sample_idx = i * int(self.vad_frame_duration * self.sample_rate / 1000)
            
            # Check energy threshold
            energy_idx = min(sample_idx // energy_window_size, len(energy) - 1)
            has_energy = energy[energy_idx] > self.energy_threshold
            
            # Check spectral centroid
            spectral_idx = min(sample_idx // hop_length, len(spectral_centroids) - 1)
            has_speech_spectrum = spectral_centroids[spectral_idx] > self.spectral_centroid_threshold
            
            # Combined decision
            is_speech_region = is_speech and has_energy and has_speech_spectrum
            
            if is_speech_region and not in_speech:
                speech_start = sample_idx
                in_speech = True
            elif not is_speech_region and in_speech:
                speech_regions.append((speech_start, sample_idx))
                in_speech = False
        
        if in_speech:
            speech_regions.append((speech_start, len(audio)))
        
        return speech_regions
    
    def detect_speech_activity(self, audio: np.ndarray, frame_duration: int = None) -> List[bool]:
        """Detect speech activity using WebRTC VAD with enhanced precision."""
        frame_duration = frame_duration or self.vad_frame_duration
        
        # Convert to 16-bit PCM
        audio_16bit = (audio * 32767).astype(np.int16)
        
        frame_size = int(self.sample_rate * frame_duration / 1000)  # frame_duration in ms
        frames = []
        
        for i in range(0, len(audio_16bit), frame_size):
            frame = audio_16bit[i:i+frame_size]
            if len(frame) == frame_size:
                frames.append(frame.tobytes())
        
        # Apply VAD with enhanced processing
        speech_frames = []
        for frame in frames:
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                speech_frames.append(is_speech)
            except:
                speech_frames.append(False)
        
        # Apply smoothing to reduce false positives/negatives
        speech_frames = self._smooth_speech_detection(speech_frames)
        
        return speech_frames
    
    def _smooth_speech_detection(self, speech_frames: List[bool], window_size: int = 7) -> List[bool]:
        """Smooth speech detection to reduce false positives/negatives."""
        if len(speech_frames) < window_size:
            return speech_frames
        
        smoothed = []
        for i in range(len(speech_frames)):
            start = max(0, i - window_size // 2)
            end = min(len(speech_frames), i + window_size // 2 + 1)
            window = speech_frames[start:end]
            
            # Require stronger majority for speech detection
            speech_count = sum(window)
            smoothed.append(speech_count > len(window) * 0.6)  # 60% threshold
        
        return smoothed
    
    def detect_silence(self, audio: np.ndarray, threshold_db: float = None) -> np.ndarray:
        """Detect silence regions in audio with enhanced precision."""
        threshold_db = threshold_db or self.silence_threshold_db
        
        # Convert to dB
        audio_db = librosa.amplitude_to_db(np.abs(audio))
        
        # Enhanced smoothing for better silence detection
        hop_length = 512
        from scipy.ndimage import uniform_filter1d
        # Use more conservative smoothing to reduce over-segmentation
        smoothing_size = max(hop_length // 4, 64)  # More conservative smoothing
        audio_db = uniform_filter1d(audio_db, size=smoothing_size)
        
        # Find silence (below threshold)
        silence_mask = audio_db < threshold_db
        
        # Apply morphological operations to clean up silence regions
        silence_mask = self._clean_silence_regions(silence_mask)
        
        return silence_mask
    
    def _clean_silence_regions(self, silence_mask: np.ndarray) -> np.ndarray:
        """Clean up silence regions using morphological operations."""
        # Remove only very short silence regions (likely noise), allow natural pauses
        min_silence_samples = int(0.2 * self.sample_rate / 512)  # 0.2 second minimum (natural pauses)
        
        # Find silence regions
        silence_starts = []
        silence_ends = []
        in_silence = False
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_starts.append(i)
                in_silence = True
            elif not is_silent and in_silence:
                silence_ends.append(i)
                in_silence = False
        
        if in_silence:
            silence_ends.append(len(silence_mask))
        
        # Clean up only very short silence regions
        cleaned_mask = silence_mask.copy()
        for start, end in zip(silence_starts, silence_ends):
            if end - start < min_silence_samples:
                # Mark very short silence as speech
                cleaned_mask[start:end] = False
        
        return cleaned_mask
    
    def find_optimal_cut_points(self, audio: np.ndarray) -> List[int]:
        """Find optimal points to cut audio based on natural silence detection."""
        # Detect silence regions
        silence_mask = self.detect_silence(audio)
        
        # Find silence regions
        silence_starts = []
        silence_ends = []
        in_silence = False
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_starts.append(i)
                in_silence = True
            elif not is_silent and in_silence:
                silence_ends.append(i)
                in_silence = False
        
        if in_silence:
            silence_ends.append(len(silence_mask))
        
        # Convert silence regions to sample indices and filter by duration
        silence_regions = []
        for start, end in zip(silence_starts, silence_ends):
            start_sample = start * 512  # Convert back to sample index
            end_sample = end * 512
            duration = (end_sample - start_sample) / self.sample_rate
            
            # Only consider silence regions of exactly 1 second for cutting
            if duration >= self.min_silence_duration:  # At least 1 second of silence
                silence_regions.append((start_sample, end_sample, duration))
        
        logger.info(f"Found {len(silence_regions)} suitable silence regions for cutting")
        if silence_regions:
            durations = [d for _, _, d in silence_regions]
            logger.info(f"Silence durations: min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
        
        # Create segments based on silence regions with intelligent selection
        cut_points = [0]  # Start of audio
        
        current_pos = 0
        max_segment_duration = 18.0  # 1 minuto máximo conforme solicitado
        
        while current_pos < len(audio):
            best_cut_point = None
            best_score = float('inf')
            
            # Look for the BEST natural silence within acceptable range
            for silence_start, silence_end, silence_duration in silence_regions:
                if silence_start <= current_pos:
                    continue
                    
                segment_duration = (silence_start - current_pos) / self.sample_rate
                
                # Skip if segment is too short
                if segment_duration < self.min_duration:
                    continue
                
                # Force cut if approaching 1 minute limit
                if segment_duration > max_segment_duration:
                    break  # Stop looking, we'll force a cut
                
                # Natural scoring - prefer longer pauses (more natural breaks)
                # Longer silences = better natural breaks
                silence_quality_score = 1.0 / (silence_duration + 0.1)  # Lower score for longer silences
                
                # Prefer segments that are reasonable length but prioritize natural breaks
                if segment_duration >= self.min_duration:
                    duration_penalty = 0  # No penalty for valid durations
                else:
                    duration_penalty = (self.min_duration - segment_duration) * 10  # High penalty for too short
                
                # Combined score (lower is better) - prioritize natural long pauses
                total_score = silence_quality_score + duration_penalty
                
                if total_score < best_score:
                    best_score = total_score
                    best_cut_point = (silence_start + silence_end) // 2
            
            if best_cut_point is not None:
                cut_points.append(best_cut_point)
                current_pos = best_cut_point
                logger.debug(f"Found silence cut at {current_pos/self.sample_rate:.2f}s")
            else:
                # No suitable silence found, force cut at 1 minute limit
                current_pos += int(max_segment_duration * self.sample_rate)
                if current_pos < len(audio):
                    cut_points.append(current_pos)
                    logger.debug(f"Forced cut at {current_pos/self.sample_rate:.2f}s (reached 1min limit, no suitable silence found)")
        
        # Add end of audio
        cut_points.append(len(audio))
        
        # Remove duplicates and sort
        cut_points = sorted(list(set(cut_points)))
        
        logger.info(f"Created {len(cut_points)-1} segments using intelligent silence detection")
        return cut_points
    
    def _find_silence_based_cuts(self, audio: np.ndarray) -> List[int]:
        """Fallback method using silence detection."""
        silence_mask = self.detect_silence(audio)
        
        # Find silence regions
        silence_starts = []
        silence_ends = []
        in_silence = False
        
        for i, is_silent in enumerate(silence_mask):
            if is_silent and not in_silence:
                silence_starts.append(i)
                in_silence = True
            elif not is_silent and in_silence:
                silence_ends.append(i)
                in_silence = False
        
        if in_silence:
            silence_ends.append(len(silence_mask))
        
        # Find silence regions suitable for cutting
        min_silence_samples = int(self.min_silence_duration * self.sample_rate / 512)
        max_silence_samples = int(self.max_silence_duration * self.sample_rate / 512)
        
        good_cut_points = []
        for start, end in zip(silence_starts, silence_ends):
            silence_duration_samples = end - start
            
            if min_silence_samples <= silence_duration_samples <= max_silence_samples:
                cut_point = (start + end) // 2 * 512
                good_cut_points.append(cut_point)
        
        return sorted(good_cut_points)
    
    
    def segment_audio(self, audio_path: Path, output_dir: Path) -> List[Tuple[Path, float, float]]:
        """
        Segment audio file intelligently based on speech patterns.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save segments
            
        Returns:
            List of tuples (segment_path, absolute_start_time, absolute_end_time)
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        logger.info(f"Loaded audio: {len(audio) / sr:.2f}s at {sr}Hz")
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Find optimal cut points
        cut_points = self.find_optimal_cut_points(audio)
        
        logger.info(f"Found {len(cut_points) - 1} potential segments")
        
        # Initialize variables
        segments_with_timestamps = []
        segment_idx = 0
        
        # Process segments based on intelligent cut points
        for i in range(len(cut_points) - 1):
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]
            
            # Calculate duration
            duration = (end_sample - start_sample) / self.sample_rate
            
            # Skip segments that are too short
            if duration < self.min_duration:
                    continue
            
            # Handle segments that are too long
            if duration > 60.0:  # More than 60s
                # Split long segments into smaller chunks
                chunk_duration = self.max_duration
                chunk_samples = int(chunk_duration * self.sample_rate)
                
                chunk_start = start_sample
                chunk_idx = 0
                
                while chunk_start < end_sample:
                    chunk_end = min(chunk_start + chunk_samples, end_sample)
                    chunk_audio = audio[chunk_start:chunk_end]
                    chunk_duration_actual = len(chunk_audio) / self.sample_rate
                    
                    # Only save chunks that meet minimum duration
                    if chunk_duration_actual >= self.min_duration:
                        # Use standardized naming
                        base_name = extract_base_name(audio_path)
                        standard_name = generate_standard_name(base_name, "chunk", chunk_idx)
                        filename = f"{standard_name}.flac"
                        segment_path = output_dir / filename
                        
                        sf.write(segment_path, chunk_audio, self.sample_rate)
                        
                        # Calculate absolute timestamps
                        absolute_start = chunk_start / self.sample_rate
                        absolute_end = chunk_end / self.sample_rate
                        
                        segments_with_timestamps.append((segment_path, absolute_start, absolute_end))
                        
                        logger.debug(f"Segment {segment_idx} chunk {chunk_idx}: {chunk_duration_actual:.2f}s ({absolute_start:.2f}s - {absolute_end:.2f}s) -> {segment_path}")
                        chunk_idx += 1
                    
                    chunk_start = chunk_end
                
                segment_idx += 1
            else:
                # Normal segment processing
                segment_audio = audio[start_sample:end_sample]
                duration = len(segment_audio) / self.sample_rate
            
                # Save segment
                # Use standardized naming
                base_name = extract_base_name(audio_path)
                standard_name = generate_standard_name(base_name, "segment", segment_idx)
                filename = f"{standard_name}.flac"
                segment_path = output_dir / filename
                
                sf.write(segment_path, segment_audio, self.sample_rate)
                
                # Calculate absolute timestamps
                absolute_start = start_sample / self.sample_rate
                absolute_end = end_sample / self.sample_rate
                
                segments_with_timestamps.append((segment_path, absolute_start, absolute_end))
                
                logger.debug(f"Segment {segment_idx}: {duration:.2f}s ({absolute_start:.2f}s - {absolute_end:.2f}s) -> {segment_path}")
                segment_idx += 1
        
        logger.info(f"Created {len(segments_with_timestamps)} segments with timestamps")
        return segments_with_timestamps
    def segment_with_timestamps(self, audio_path: Path, output_dir: Path) -> List[Tuple[Path, float, float]]:
        """
        Segment audio and return with timestamps.
        
        Returns:
            List of (path, start_time, end_time) tuples
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Normalize audio
        audio = self.normalize_audio(audio)
        
        # Find optimal cut points
        cut_points = self.find_optimal_cut_points(audio)
        cut_points = [0] + cut_points + [len(audio)]
        cut_points = sorted(list(set(cut_points)))
        
        segments_with_timestamps = []
        i = 0
        segment_idx = 0
        
        while i < len(cut_points) - 1:
            start_sample = cut_points[i]
            end_sample = cut_points[i + 1]
            duration = (end_sample - start_sample) / self.sample_rate
            
            if duration < self.min_duration and i < len(cut_points) - 2:
                continue
                
            if duration > self.max_duration:
                sub_cuts = [cp for cp in cut_points if start_sample < cp < end_sample]
                if sub_cuts:
                    target_sample = start_sample + int(self.max_duration * self.sample_rate)
                    best_cut = min(sub_cuts, key=lambda x: abs(x - target_sample))
                    end_sample = best_cut
                else:
                    end_sample = start_sample + int(self.max_duration * self.sample_rate)
            
            segment_audio = audio[start_sample:end_sample]
            duration = len(segment_audio) / self.sample_rate
            
            if duration >= self.min_duration:
                # Use standardized naming
                base_name = extract_base_name(audio_path)
                standard_name = generate_standard_name(base_name, "segment", segment_idx)
                filename = f"{standard_name}.{Config.AUDIO_FORMAT}"
                segment_path = output_dir / filename
                
                sf.write(segment_path, segment_audio, self.sample_rate)
                
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                
                segments_with_timestamps.append((segment_path, start_time, end_time))
                segment_idx += 1
            
            # Move to next
            overlap_samples = int(self.overlap * self.sample_rate)
            next_start = max(start_sample + 1, end_sample - overlap_samples)
            
            i += 1
            while i < len(cut_points) and cut_points[i] <= next_start:
                i += 1
                
            if i >= len(cut_points):
                break
        
        return segments_with_timestamps


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    segmenter = AudioSegmenter()
    # segments = segmenter.segment_audio(Path("input.flac"), Path("segments/"))
    # print(f"Created {len(segments)} segments")
