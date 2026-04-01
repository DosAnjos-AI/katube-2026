"""
Enhanced speaker diarization using pyannote with improved processing.
"""
import os
import torch
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import logging
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, Timeline
import soundfile as sf

from config import Config

logger = logging.getLogger(__name__)

class EnhancedDiarizer:
    def __init__(self, huggingface_token: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.huggingface_token = huggingface_token or Config.HUGGINGFACE_TOKEN
        self.sample_rate = Config.SAMPLE_RATE
        
        # Initialize pipeline
        self.pipeline = None
        self._load_pipeline()
        
        logger.info(f"Diarizer running on {self.device}")
    
    def _load_pipeline(self):
        """Load the pyannote speaker diarization pipeline."""
        try:
            self.pipeline = Pipeline.from_pretrained(
                Config.PYANNOTE_MODEL,
                use_auth_token=self.huggingface_token
            )
            if self.pipeline is not None:
                self.pipeline = self.pipeline.to(self.device)
                logger.info(f"Loaded {Config.PYANNOTE_MODEL} pipeline")
            else:
                raise ValueError("Pipeline returned None - check model access permissions")
        except Exception as e:
            error_msg = f"Failed to load diarization pipeline: {e}"
            if "gated" in str(e).lower() or "unauthorized" in str(e).lower() or "401" in str(e):
                error_msg += "\n\n🚨 SOLUTION: Visit these URLs and accept terms:\n"
                error_msg += "   • https://hf.co/pyannote/speaker-diarization-3.1\n"
                error_msg += "   • https://hf.co/pyannote/segmentation-3.0\n"
                error_msg += "   • https://hf.co/pyannote/embedding\n"
                error_msg += "Then restart the server."
            logger.error(error_msg)
            self.pipeline = None  # Set to None instead of raising
            return
    
    def preprocess_audio(self, audio_path: Path) -> Tuple[torch.Tensor, int]:
        """Preprocess audio for diarization."""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate)
                waveform = resampler(waveform)
                sample_rate = self.sample_rate
            
            # Normalize
            waveform = waveform / torch.max(torch.abs(waveform))
            
            # Move to device
            waveform = waveform.to(self.device)
            
            return waveform, sample_rate
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def diarize_audio(self, audio_path: Path, num_speakers: Optional[int] = None) -> Annotation:
        """
        Perform speaker diarization on an audio file.
        
        Args:
            audio_path: Path to audio file
            num_speakers: Hint for number of speakers (optional)
            
        Returns:
            pyannote Annotation object
        """
        if self.pipeline is None:
            error_msg = "Diarization pipeline not available. Please accept model terms and restart server."
            logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        logger.info(f"Diarizing {audio_path.name}")
        
        # Preprocess audio
        waveform, sample_rate = self.preprocess_audio(audio_path)
        
        # Prepare input for pipeline
        audio_input = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # Add speaker count hint if provided
        if num_speakers is not None:
            # pyannote.audio 3.x way to set number of speakers
            self.pipeline.instantiate({"clustering": {"num_clusters": num_speakers}})
        
            # Run diarization
        # Run diarization
        try:
            diarization = self.pipeline(audio_input)
            
            # pyannote 3.x retorna DiarizeOutput com Annotation dentro
            if hasattr(diarization, 'speaker_diarization'):
                # É DiarizeOutput (pyannote 3.x)
                annotation = diarization.speaker_diarization
                num_speakers = len(annotation.labels())
                logger.info(f"Diarization completed: {num_speakers} speakers detected")
                return annotation  # Retornar Annotation, não DiarizeOutput
            else:
                # É Annotation diretamente (versão antiga)
                num_speakers = len(diarization.labels())
                logger.info(f"Diarization completed: {num_speakers} speakers detected")
                return diarization
            
        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            raise




    def annotation_to_dataframe(self, annotation: Annotation, audio_duration: Optional[float] = None) -> pd.DataFrame:
        """Convert pyannote Annotation to pandas DataFrame."""
        segments_data = []
        
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            segments_data.append({
                'START': segment.start,
                'END': segment.end,
                'DURATION': segment.duration,
                'SPEAKER': speaker,
                'CONFIDENCE': 1.0  # pyannote doesn't provide confidence in this version
            })
        
        df = pd.DataFrame(segments_data)
        
        if not df.empty:
            # Sort by start time
            df = df.sort_values('START').reset_index(drop=True)
            
            # Add additional metrics
            if audio_duration is not None:
                df['RELATIVE_START'] = df['START'] / audio_duration
                df['RELATIVE_END'] = df['END'] / audio_duration
        
        return df
    
    def save_rttm(self, annotation: Annotation, output_path: Path, audio_filename: str):
        """Save diarization results in RTTM format."""
        with open(output_path, 'w') as f:
            annotation.write_rttm(f)
        logger.info(f"RTTM saved to {output_path}")
    
    def post_process_annotation(self, annotation: Annotation, min_duration: float = 0.5) -> Annotation:
        """
        Post-process diarization results.
        
        Args:
            annotation: Original annotation
            min_duration: Minimum segment duration to keep
            
        Returns:
            Processed annotation
        """
        # Remove very short segments
        cleaned = annotation.support(min_duration)
        
        # Skip gap filling for now - simplified approach
        filled = cleaned
        
        # Merge nearby segments from the same speaker
        processed = Annotation()
        
        for speaker in cleaned.labels():
            speaker_timeline = cleaned.label_timeline(speaker)
            # Merge segments that are close together (within 0.5 seconds)
            merged_timeline = speaker_timeline.support(0.5)
            
            for segment in merged_timeline:
                processed[segment] = speaker
        
        return processed
    
    def analyze_speaker_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze speaker statistics from diarization results."""
        if df.empty:
            return {'error': 'No diarization data available'}
        
        stats = {}
        
        # Overall statistics
        total_duration = df['DURATION'].sum()
        stats['total_speech_duration'] = total_duration
        stats['num_segments'] = len(df)
        stats['num_speakers'] = df['SPEAKER'].nunique()
        
        # Per-speaker statistics
        speaker_stats = []
        for speaker in df['SPEAKER'].unique():
            speaker_df = df[df['SPEAKER'] == speaker]
            speaker_info = {
                'speaker': speaker,
                'total_duration': speaker_df['DURATION'].sum(),
                'num_segments': len(speaker_df),
                'avg_segment_duration': speaker_df['DURATION'].mean(),
                'speaking_percentage': (speaker_df['DURATION'].sum() / total_duration) * 100
            }
            speaker_stats.append(speaker_info)
        
        # Sort by speaking time
        speaker_stats = sorted(speaker_stats, key=lambda x: x['total_duration'], reverse=True)
        stats['speakers'] = speaker_stats
        
        # Overlap analysis
        overlaps = self._detect_overlaps_in_annotation(df)
        stats['overlaps'] = overlaps
        
        return stats
    
    def _detect_overlaps_in_annotation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overlaps in the diarization annotation."""
        if df.empty:
            return {'num_overlaps': 0, 'total_overlap_duration': 0.0}
        
        overlaps = []
        
        # Sort by start time
        df_sorted = df.sort_values('START')
        
        for i in range(len(df_sorted) - 1):
            current = df_sorted.iloc[i]
            next_segment = df_sorted.iloc[i + 1]
            
            # Check if segments overlap
            if current['END'] > next_segment['START'] and current['SPEAKER'] != next_segment['SPEAKER']:
                overlap_start = next_segment['START']
                overlap_end = min(current['END'], next_segment['END'])
                overlap_duration = overlap_end - overlap_start
                
                if overlap_duration > 0:
                    overlaps.append({
                        'start': overlap_start,
                        'end': overlap_end,
                        'duration': overlap_duration,
                        'speakers': [current['SPEAKER'], next_segment['SPEAKER']]
                    })
        
        total_overlap_duration = sum(o['duration'] for o in overlaps)
        
        return {
            'num_overlaps': len(overlaps),
            'total_overlap_duration': total_overlap_duration,
            'overlap_details': overlaps
        }
    
    def diarize_batch(self, audio_files: List[Path], output_dir: Path, save_rttm: bool = True) -> Dict[str, Any]:
        """
        Diarize multiple audio files in batch.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save results
            save_rttm: Whether to save RTTM files
            
        Returns:
            Dictionary with batch processing results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        
        for audio_path in audio_files:
            try:
                # Skip if RTTM already exists
                rttm_path = output_dir / f"{audio_path.stem}.rttm"
                if rttm_path.exists() and save_rttm:
                    logger.info(f"RTTM already exists for {audio_path.name}, skipping")
                    continue
                
                # Perform diarization
                annotation = self.diarize_audio(audio_path)
                
                # Convert to DataFrame
                audio_duration = self._get_audio_duration(audio_path)
                df = self.annotation_to_dataframe(annotation, audio_duration)
                
                # Post-process
                processed_annotation = self.post_process_annotation(annotation)
                
                # Save RTTM if requested
                if save_rttm:
                    self.save_rttm(processed_annotation, rttm_path, audio_path.name)
                
                # Analyze statistics
                stats = self.analyze_speaker_statistics(df)
                
                results[str(audio_path)] = {
                    'annotation': processed_annotation,
                    'dataframe': df,
                    'statistics': stats,
                    'rttm_path': str(rttm_path) if save_rttm else None
                }
                
            except Exception as e:
                logger.error(f"Failed to process {audio_path}: {e}")
                results[str(audio_path)] = {'error': str(e)}
        
        return results
    
    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            with sf.SoundFile(audio_path) as f:
                return len(f) / f.samplerate
        except:
            # Fallback using torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform.shape[1] / sample_rate


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    diarizer = EnhancedDiarizer()
    # results = diarizer.diarize_batch([Path("input.flac")], Path("output/"))
    # print(f"Diarization results: {results}")
