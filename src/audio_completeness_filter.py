"""
Audio Completeness Filter
Detects and filters cut/incomplete audio segments
"""
import librosa
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

class AudioCompletenessFilter:
    """
    Audio completeness filter to detect and remove cut/incomplete audio segments.
    """
    
    def __init__(self, 
                 completeness_threshold: float = 0.8,
                 energy_continuity_threshold: float = 0.9,
                 transition_threshold_db: float = -15):
        """
        Initialize audio completeness filter.
        
        Args:
            completeness_threshold: Minimum completeness score (0.0-1.0)
            energy_continuity_threshold: Energy continuity threshold
            transition_threshold_db: Transition threshold in dB
        """
        self.completeness_threshold = completeness_threshold
        self.energy_continuity_threshold = energy_continuity_threshold
        self.transition_threshold_db = transition_threshold_db
        
        logger.info(f"AudioCompletenessFilter initialized:")
        logger.info(f"   - Completeness threshold: {self.completeness_threshold}")
        logger.info(f"   - Energy continuity threshold: {self.energy_continuity_threshold}")
        logger.info(f"   - Transition threshold: {self.transition_threshold_db} dB")
    
    def analyze_audio_completeness(self, audio_path: Path) -> Dict[str, Any]:
        """
        Analyze audio segment for completeness indicators.
        
        Args:
            audio_path: Path to audio file to analyze
            
        Returns:
            Dictionary with completeness analysis results
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = len(audio) / sr
            
            # Analyze energy continuity
            energy_continuity = self._analyze_energy_continuity(audio, sr)
            
            # Analyze spectral continuity
            spectral_continuity = self._analyze_spectral_continuity(audio, sr)
            
            # Analyze transition smoothness
            transition_smoothness = self._analyze_transition_smoothness(audio, sr)
            
            # Calculate overall completeness score
            completeness_score = (
                energy_continuity * 0.4 +
                spectral_continuity * 0.3 +
                transition_smoothness * 0.3
            )
            
            # Determine if segment is complete
            is_complete = completeness_score >= self.completeness_threshold
            
            # Collect filter reasons if not complete
            filter_reasons = []
            if not is_complete:
                if energy_continuity < self.energy_continuity_threshold:
                    filter_reasons.append(f"Low energy continuity: {energy_continuity:.2f}")
                if spectral_continuity < 0.7:
                    filter_reasons.append(f"Low spectral continuity: {spectral_continuity:.2f}")
                if transition_smoothness < 0.7:
                    filter_reasons.append(f"Poor transition smoothness: {transition_smoothness:.2f}")
            
            return {
                'is_complete': is_complete,
                'completeness_score': completeness_score,
                'energy_continuity': energy_continuity,
                'spectral_continuity': spectral_continuity,
                'transition_smoothness': transition_smoothness,
                'duration': duration,
                'filter_reasons': filter_reasons
            }
            
        except Exception as e:
            logger.error(f"Error analyzing completeness for {audio_path}: {e}")
            return {
                'is_complete': False,
                'completeness_score': 0.0,
                'energy_continuity': 0.0,
                'spectral_continuity': 0.0,
                'transition_smoothness': 0.0,
                'duration': 0.0,
                'filter_reasons': [f"Analysis error: {str(e)}"]
            }
    
    def _analyze_energy_continuity(self, audio: np.ndarray, sr: int) -> float:
        """Analyze energy continuity throughout the audio."""
        try:
            # Calculate RMS energy in windows
            window_size = int(0.1 * sr)  # 100ms windows
            hop_size = window_size // 2
            
            energy_values = []
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                energy_values.append(energy)
            
            if len(energy_values) < 2:
                return 1.0
            
            # Calculate energy continuity (how smooth the energy changes are)
            energy_diff = np.diff(energy_values)
            energy_continuity = 1.0 - np.std(energy_diff) / (np.mean(energy_values) + 1e-8)
            
            return max(0.0, min(1.0, energy_continuity))
            
        except Exception as e:
            logger.error(f"Error in energy continuity analysis: {e}")
            return 0.0
    
    def _analyze_spectral_continuity(self, audio: np.ndarray, sr: int) -> float:
        """Analyze spectral continuity throughout the audio."""
        try:
            # Calculate spectral centroid in windows
            hop_length = 512
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, sr=sr, hop_length=hop_length
            )[0]
            
            if len(spectral_centroids) < 2:
                return 1.0
            
            # Calculate spectral continuity
            spectral_diff = np.diff(spectral_centroids)
            spectral_continuity = 1.0 - np.std(spectral_diff) / (np.mean(spectral_centroids) + 1e-8)
            
            return max(0.0, min(1.0, spectral_continuity))
            
        except Exception as e:
            logger.error(f"Error in spectral continuity analysis: {e}")
            return 0.0
    
    def _analyze_transition_smoothness(self, audio: np.ndarray, sr: int) -> float:
        """Analyze transition smoothness at segment boundaries."""
        try:
            # Analyze beginning and end transitions
            transition_length = int(0.1 * sr)  # 100ms transition analysis
            
            # Beginning transition
            if len(audio) > transition_length:
                beginning = audio[:transition_length]
                beginning_energy = np.sqrt(np.mean(beginning ** 2))
            else:
                beginning_energy = np.sqrt(np.mean(audio ** 2))
            
            # End transition
            if len(audio) > transition_length:
                ending = audio[-transition_length:]
                ending_energy = np.sqrt(np.mean(ending ** 2))
            else:
                ending_energy = np.sqrt(np.mean(audio ** 2))
            
            # Overall energy
            overall_energy = np.sqrt(np.mean(audio ** 2))
            
            # Calculate transition smoothness
            # Good transitions should have energy close to overall energy
            beginning_smoothness = 1.0 - abs(beginning_energy - overall_energy) / (overall_energy + 1e-8)
            ending_smoothness = 1.0 - abs(ending_energy - overall_energy) / (overall_energy + 1e-8)
            
            transition_smoothness = (beginning_smoothness + ending_smoothness) / 2
            
            return max(0.0, min(1.0, transition_smoothness))
            
        except Exception as e:
            logger.error(f"Error in transition smoothness analysis: {e}")
            return 0.0
    
    def filter_cut_audio(self, 
                        segment_paths: List[Path], 
                        rejected_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Filter cut/incomplete audio segments.
        
        Args:
            segment_paths: List of audio segment paths to filter
            rejected_dir: Directory to move rejected segments
            
        Returns:
            Dictionary with filtering results
        """
        logger.info(f"🔍 Applying completeness filter to {len(segment_paths)} segments...")
        
        if rejected_dir:
            rejected_dir.mkdir(parents=True, exist_ok=True)
        
        complete_segments = []
        cut_segments = []
        filter_reasons = {}
        
        for i, segment_path in enumerate(segment_paths):
            logger.info(f"Analyzing segment {i+1}/{len(segment_paths)}: {segment_path.name}")
                
                # Analyze completeness
            analysis = self.analyze_audio_completeness(segment_path)
                
            if analysis['is_complete']:
                    complete_segments.append(segment_path)
                logger.debug(f"✅ Complete: {segment_path.name} (score: {analysis['completeness_score']:.2f})")
                else:
                cut_segments.append(segment_path)
                filter_reasons[str(segment_path)] = analysis['filter_reasons']
                
                # Move to rejected directory if specified
                if rejected_dir:
                    rejected_path = rejected_dir / segment_path.name
                    segment_path.rename(rejected_path)
                    logger.debug(f"❌ Moved to rejected: {segment_path.name}")
                
                logger.debug(f"❌ Cut: {segment_path.name} (score: {analysis['completeness_score']:.2f}) - {', '.join(analysis['filter_reasons'])}")
        
        # Calculate statistics
        total_segments = len(segment_paths)
        complete_count = len(complete_segments)
        cut_count = len(cut_segments)
        completeness_rate = complete_count / total_segments if total_segments > 0 else 0.0
        
        logger.info(f"Completeness filter results:")
        logger.info(f"   - Total segments: {total_segments}")
        logger.info(f"   - Complete segments: {complete_count}")
        logger.info(f"   - Cut segments filtered: {cut_count}")
        logger.info(f"   - Completeness rate: {completeness_rate:.1%}")
        
        return {
            'complete_segments': complete_segments,
            'cut_segments': cut_segments,
            'total_segments': total_segments,
            'complete_count': complete_count,
            'cut_count': cut_count,
            'completeness_rate': completeness_rate,
            'filter_reasons': filter_reasons
        }