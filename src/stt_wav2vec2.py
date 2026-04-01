"""
WAV2VEC2 STT Transcriber
Specialized WAV2VEC2 model for Portuguese Brazilian transcription
"""
import torch
import librosa
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from naming_utils import extract_base_name, generate_standard_name

logger = logging.getLogger(__name__)

class WAV2VEC2STTTranscriber:
    """
    WAV2VEC2 STT transcriber for audio segments.
    Specialized for Portuguese Brazilian using alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization
    """
    
    def __init__(self, 
                 wav2vec2_model_name: str = "alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization",
             device: str = None):
        """
        Initialize WAV2VEC2 STT transcriber.
        
        Args:
            wav2vec2_model_name: HuggingFace WAV2VEC2 model name
            device: Device to run WAV2VEC2 on (None = auto-detect, 'cpu', or 'cuda')
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
     
        self.wav2vec2_model_name = wav2vec2_model_name
        
        # Initialize models
        self.wav2vec2_processor = None
        self.wav2vec2_model = None
        
        logger.info("🔄 Initializing WAV2VEC2 STT model...")
        self._load_models()
        
    def _load_models(self):
        """Load WAV2VEC2 model."""
        try:
            # Load WAV2VEC2 model (especializado em português brasileiro)
            logger.info(f"Loading WAV2VEC2 model: {self.wav2vec2_model_name}")
            logger.info("   Model: alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization")
            logger.info("   Specialized for Brazilian Portuguese")
            
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                self.wav2vec2_model_name
            )
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                self.wav2vec2_model_name
            )
            self.wav2vec2_model.to(self.device)
            logger.info("✅ WAV2VEC2 model loaded successfully")
                
        except Exception as e:
            logger.error(f"❌ Error loading WAV2VEC2 STT model: {e}")
            raise RuntimeError(f"Failed to load WAV2VEC2 STT model: {e}")
    
    def _preprocess_audio(self, audio_path: Path, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio file for WAV2VEC2 STT.
        
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
        Transcribe audio using WAV2VEC2 model.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Process audio
            input_values = self.wav2vec2_processor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values.to(self.device)
            
            # Generate transcription
            with torch.no_grad():
                logits = self.wav2vec2_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
            
            logger.debug(f"WAV2VEC2 transcription for {audio_path.name}: {transcription[:50]}...")
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing with WAV2VEC2 {audio_path}: {e}")
            return ""
    
    def transcribe_segments(self, 
                           segment_paths: List[Path], 
                           output_dir: Path) -> Dict[str, Any]:
        """
        Transcribe multiple audio segments using WAV2VEC2 STT.
        
        Args:
            segment_paths: List of audio segment paths to transcribe
            output_dir: Directory to save transcription results
            
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"🎤 Starting WAV2VEC2 STT transcription of {len(segment_paths)} segments...")
        
        # Create output directory
        stt_dir = output_dir / 'stt_results'
        wav2vec2_dir = stt_dir / 'STT-wav2vec2'
        wav2vec2_dir.mkdir(parents=True, exist_ok=True)
        
        # Transcribe segments
        wav2vec2_results = []
        
        for i, segment_path in enumerate(segment_paths):
            logger.info(f"Processing segment {i+1}/{len(segment_paths)}: {segment_path.name}")
            
            try:
                # Transcribe with WAV2VEC2
                transcription = self.transcribe_audio(segment_path)
                
                # Save transcription with standardized naming
                base_name = extract_base_name(segment_path)
                standard_name = generate_standard_name(base_name, "stt_wav2vec2", i+1)
                wav2vec2_file = wav2vec2_dir / f"{standard_name}.txt"
                with open(wav2vec2_file, 'w', encoding='utf-8') as f:
                    f.write(transcription)
                
                wav2vec2_results.append({
                    "segment": segment_path.name,
                    "transcription": transcription,
                    "file": str(wav2vec2_file)
                })
                
                logger.info(f"✅ Transcribed {segment_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing segment {segment_path.name}: {e}")
        
        logger.info(f"WAV2VEC2 STT completed: {len(wav2vec2_results)} transcriptions")
        
        return {
            "wav2vec2_results": wav2vec2_results,
            "wav2vec2_dir": str(wav2vec2_dir),
            "total_segments": len(segment_paths),
            "wav2vec2_count": len(wav2vec2_results)
        }
