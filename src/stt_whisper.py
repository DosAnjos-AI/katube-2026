"""
Whisper STT Transcriber
Specialized Whisper model for Portuguese Brazilian transcription
"""
import torch
import librosa
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from naming_utils import extract_base_name, generate_standard_name

logger = logging.getLogger(__name__)

class WhisperSTTTranscriber:
    """
    Whisper STT transcriber for audio segments.
    Specialized for Portuguese Brazilian.
    """
    
    def __init__(self, 
                 whisper_model_name: str = "freds0/distil-whisper-large-v3-ptbr",
                 device: str = None,
                 huggingface_token: Optional[str] = None):
        """
        Initialize Whisper STT transcriber.
        
        Args:
            whisper_model_name: HuggingFace Whisper model name
            device: Device to run Whisper on ('cpu' or 'cuda')
            huggingface_token: HuggingFace token for authentication
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.whisper_model_name = whisper_model_name
        self.huggingface_token = huggingface_token
        
        # Initialize models
        self.whisper_processor = None
        self.whisper_model = None
        
        logger.info("🔄 Initializing Whisper STT model...")
        self._load_models()
        
    def _load_models(self):
        """Load Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.whisper_model_name,
                token=self.huggingface_token
            )
            
            # 🔥 FIX: SEMPRE usar float32 para evitar conflitos de mixed precision
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_name,
                torch_dtype=torch.float32,  # ✅ SEMPRE float32
                token=self.huggingface_token
            )
            self.whisper_model.to(self.device)
            logger.info("✅ Whisper model loaded successfully (float32)")
                
        except Exception as e:
            logger.error(f"❌ Error loading Whisper STT model: {e}")
            raise RuntimeError(f"Failed to load Whisper STT model: {e}")
    
    def _preprocess_audio(self, audio_path: Path, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio file for Whisper STT.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def transcribe_audio(self, audio_path: Path) -> str:
        """
        Transcribe audio using Whisper model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Process audio
            input_features = self.whisper_processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_features.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = self.whisper_model.generate(input_features)
                transcription = self.whisper_processor.batch_decode(
                    predicted_ids, 
                    skip_special_tokens=True
                )[0]
            
            logger.debug(f"Whisper transcription for {audio_path.name}: {transcription[:50]}...")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing with Whisper {audio_path}: {e}")
            return ""
    
    def transcribe_segments(self, 
                           segment_paths: List[Path], 
                           output_dir: Path) -> Dict[str, Any]:
        """
        Transcribe multiple audio segments using Whisper STT.
        
        Args:
            segment_paths: List of audio segment paths to transcribe
            output_dir: Directory to save transcription results
            
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"🎤 Starting Whisper STT transcription of {len(segment_paths)} segments...")
        
        # Create output directory
        stt_dir = output_dir / 'stt_results'
        whisper_dir = stt_dir / 'STT-whisper'
        whisper_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe segments
        whisper_results = []
        
        for i, segment_path in enumerate(segment_paths):
            logger.info(f"Processing segment {i+1}/{len(segment_paths)}: {segment_path.name}")
            
            try:
                # Transcribe with Whisper
                transcription = self.transcribe_audio(segment_path)
                
                # Save transcription with standardized naming
                base_name = extract_base_name(segment_path)
                standard_name = generate_standard_name(base_name, "stt_whisper", i+1)
                whisper_file = whisper_dir / f"{standard_name}.txt"
                with open(whisper_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                whisper_results.append({
                    "segment": segment_path.name,
                    "transcription": transcription,
                    "file": str(whisper_file)
                })
                
                logger.info(f"✅ Transcribed {segment_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing segment {segment_path.name}: {e}")
        
        logger.info(f"Whisper STT completed: {len(whisper_results)} transcriptions")
        
        return {
            "whisper_results": whisper_results,
            "whisper_dir": str(whisper_dir),
            "total_segments": len(segment_paths),
            "whisper_count": len(whisper_results)
        }
