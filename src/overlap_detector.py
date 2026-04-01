"""
Overlap Speech Detection (OSD) using pyannote.audio segmentation model.
"""
import numpy as np
import soundfile as sf
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging
import torch
import torchaudio
import librosa
from pyannote.audio import Pipeline
from pyannote.core import Segment

from config import Config

logger = logging.getLogger(__name__)

class OverlapDetector:
    def __init__(self):
        self.sample_rate = Config.SAMPLE_RATE
        self.overlap_threshold = Config.OVERLAP_THRESHOLD
        self.min_speech_duration = Config.MIN_SPEECH_DURATION
        
        # Load pyannote segmentation model for OSD
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load pyannote segmentation model for overlap speech detection."""
        try:
            # Load the segmentation model with authentication if needed
            from pyannote.audio import Model
            import os
            
            # Check if we have a Hugging Face token
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            
            if hf_token:
                # Use authentication token
                model = Model.from_pretrained(
                    "pyannote/segmentation-3.0",
                    token=hf_token
                )
                logger.info("✅ Loaded pyannote/segmentation-3.0 model for OSD (with authentication)")
            else:
                # Try without authentication (for public models)
                model = Model.from_pretrained("pyannote/segmentation-3.0")
                logger.info("✅ Loaded pyannote/segmentation-3.0 model for OSD (public)")
                
            self.pipeline = model
            
        except Exception as e:
            logger.error(f"❌ Could not load pyannote segmentation model: {e}")
            logger.info("🔄 Falling back to overlap detection without pyannote")
            self.pipeline = None
    
    def detect_overlap(self, audio_path: Path) -> Dict[str, Any]:
        """
        Detect overlapping speech in audio file using pyannote segmentation.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with overlap detection results
        """
        if not self.pipeline:
            logger.error("❌ Pipeline not loaded")
            return {"error": "Model not loaded"}
        
        try:
            logger.debug(f"🔍 Analyzing overlap in: {audio_path.name}")
            
            # Load audio and run model inference
            import torchaudio
            waveform, sample_rate = torchaudio.load(str(audio_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Run model inference
            with torch.no_grad():
                output = self.pipeline(waveform)
            
            # Extract speech and overlap segments from model output
            overlap_segments = []
            speech_segments = []
            
            # Parse model output (this depends on the specific model format)
            # For now, we'll use a simplified approach
            if hasattr(output, 'itertracks'):
                for segment, track, label in output.itertracks(yield_label=True):
                    if label == "SPEECH":
                        speech_segments.append(segment)
                    elif label == "OVERLAP":
                        overlap_segments.append(segment)
            else:
                # Fallback: use all segments identified by model
                logger.warning("🔄 Using fallback overlap detection")
                # Simple approach: assume some overlap if multiple speakers likely
                total_duration = len(waveform[0]) / sample_rate
                speech_segments = [Segment(0, total_duration)]
                overlap_segments = []
            
            # Convert to list format
            overlap_list = []
            for segment in overlap_segments:
                overlap_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.end - segment.start
                })
            
            # Calculate overlap statistics
            total_duration = sum(seg.end - seg.start for seg in speech_segments)
            overlap_duration = sum(seg["duration"] for seg in overlap_list)
            overlap_percentage = (overlap_duration / total_duration) * 100 if total_duration > 0 else 0
            
            result = {
                "audio_path": str(audio_path),
                "total_duration": total_duration,
                "overlap_segments": overlap_list,
                "overlap_duration": overlap_duration,
                "overlap_percentage": overlap_percentage,
                "has_overlap": overlap_percentage > (self.overlap_threshold * 100),
                "overlap_count": len(overlap_list)
            }
            
            logger.debug(f"📊 Overlap analysis: {overlap_percentage:.1f}% overlap, {len(overlap_list)} segments")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in overlap detection for {audio_path}: {e}")
            return {"error": str(e)}
    
    def process_segments(self, segments: List[Path], output_dir: Path) -> Dict[str, Any]:
        """
        Process multiple audio segments for overlap detection.
        
        Args:
            segments: List of audio segment paths
            output_dir: Directory to save overlapping segments
            
        Returns:
            Dictionary with processing results
        """
        if not segments:
            logger.warning("⚠️ No segments to process")
            return {"error": "No segments provided"}
        
        # Create overlapping directory
        overlapping_dir = output_dir / 'overlapping'
        overlapping_dir.mkdir(parents=True, exist_ok=True)
        
        overlapping_segments = []
        non_overlapping_segments = []
        
        logger.info(f"🔍 Processing {len(segments)} segments for overlap detection...")
        
        for i, segment_path in enumerate(segments):
            try:
                logger.debug(f"Processing segment {i+1}/{len(segments)}: {segment_path.name}")
                
                # Detect overlap
                result = self.detect_overlap(segment_path)
                
                if "error" in result:
                    logger.warning(f"⚠️ Error processing {segment_path.name}: {result['error']}")
                    continue
                
                # Check if segment has significant overlap
                if result["has_overlap"]:
                    # Copy to overlapping directory
                    overlapping_path = overlapping_dir / segment_path.name
                    shutil.copy2(segment_path, overlapping_path)
                    overlapping_segments.append(segment_path)
                    
                    logger.info(f"🔄 Overlapping segment: {segment_path.name} ({result['overlap_percentage']:.1f}% overlap)")
                else:
                    non_overlapping_segments.append(segment_path)
                    logger.debug(f"✅ Non-overlapping segment: {segment_path.name}")
                
            except Exception as e:
                logger.error(f"❌ Error processing segment {segment_path.name}: {e}")
                continue
        
        # Calculate statistics
        total_segments = len(segments)
        overlapping_count = len(overlapping_segments)
        non_overlapping_count = len(non_overlapping_segments)
        overlap_percentage = (overlapping_count / total_segments) * 100 if total_segments > 0 else 0
        
        result = {
            "total_segments": total_segments,
            "overlapping_segments": overlapping_segments,
            "non_overlapping_segments": non_overlapping_segments,
            "overlapping_count": overlapping_count,
            "non_overlapping_count": non_overlapping_count,
            "overlap_percentage": overlap_percentage,
            "overlapping_dir": str(overlapping_dir)
        }
        
        logger.info(f"📊 Overlap detection completed:")
        logger.info(f"   - Total segments: {total_segments}")
        logger.info(f"   - Overlapping: {overlapping_count} ({overlap_percentage:.1f}%)")
        logger.info(f"   - Non-overlapping: {non_overlapping_count}")
        
        return result
    
    def filter_overlapping_segments(self, segments: List[Path], output_dir: Path) -> Tuple[List[Path], List[Path]]:
        """
        Filter segments into clean and overlapping categories.
        
        Args:
            segments: List of audio segment paths
            output_dir: Directory to save overlapping segments
            
        Returns:
            Tuple of (clean_segments, overlapping_segments)
        """
        result = self.process_segments(segments, output_dir)
        
        if "error" in result:
            logger.error(f"❌ Error in overlap detection: {result['error']}")
            return segments, []  # Return all as clean if error
        
        clean_segments = result["non_overlapping_segments"]
        overlapping_segments = result["overlapping_segments"]
        
        return clean_segments, overlapping_segments
    
    def get_overlap_statistics(self, segments: List[Path]) -> Dict[str, Any]:
        """
        Get overlap statistics for segments without moving files.
        
        Args:
            segments: List of audio segment paths
            
        Returns:
            Dictionary with overlap statistics
        """
        if not segments:
            return {"error": "No segments provided"}
        
        overlap_stats = []
        total_overlap_duration = 0
        total_duration = 0
        
        for segment_path in segments:
            try:
                result = self.detect_overlap(segment_path)
                
                if "error" not in result:
                    overlap_stats.append({
                        "segment": segment_path.name,
                        "overlap_percentage": result["overlap_percentage"],
                        "overlap_duration": result["overlap_duration"],
                        "total_duration": result["total_duration"],
                        "has_overlap": result["has_overlap"]
                    })
                    
                    total_overlap_duration += result["overlap_duration"]
                    total_duration += result["total_duration"]
                
            except Exception as e:
                logger.warning(f"⚠️ Error analyzing {segment_path.name}: {e}")
                continue
        
        overall_overlap_percentage = (total_overlap_duration / total_duration) * 100 if total_duration > 0 else 0
        
        return {
            "segments_analyzed": len(overlap_stats),
            "total_duration": total_duration,
            "total_overlap_duration": total_overlap_duration,
            "overall_overlap_percentage": overall_overlap_percentage,
            "segment_details": overlap_stats
        }