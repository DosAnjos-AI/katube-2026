"""
Speech-to-Text (STT) module using Whisper and WAV2VEC2 for Brazilian Portuguese.
"""
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import librosa
import soundfile as sf

logger = logging.getLogger(__name__)

class STTTranscriber:
    def __init__(self, 
                 whisper_model_name: str = "freds0/distil-whisper-large-v3-ptbr",
                 wav2vec2_model_name: str = "facebook/wav2vec2-base-960h",
                 device: str = "cpu",
                 huggingface_token: Optional[str] = None):
        """
        Initialize STT transcriber with Whisper and WAV2VEC2 models.
        
        Args:
            whisper_model_name: HuggingFace model name for Whisper
            wav2vec2_model_name: HuggingFace model name for WAV2VEC2
            device: Device to run models on ('cpu' or 'cuda')
            huggingface_token: HuggingFace token for authentication
        """
        self.device = device
        self.whisper_model_name = whisper_model_name
        self.wav2vec2_model_name = wav2vec2_model_name
        self.huggingface_token = huggingface_token
        
        # Initialize models
        self.whisper_processor = None
        self.whisper_model = None
        self.wav2vec2_processor = None
        self.wav2vec2_model = None
        
        logger.info("🔄 Initializing STT models...")
        self._load_models()
        
    def _load_models(self):
        """Load Whisper and WAV2VEC2 models."""
        try:
            # Load Whisper model
            logger.info(f"Loading Whisper model: {self.whisper_model_name}")
            self.whisper_processor = WhisperProcessor.from_pretrained(
                self.whisper_model_name,
                token=self.huggingface_token
            )
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
                self.whisper_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=self.huggingface_token
            )
            self.whisper_model.to(self.device)
            logger.info("✅ Whisper model loaded successfully")
            
            # Load WAV2VEC2 model (público - sem token necessário)
            logger.info(f"Loading WAV2VEC2 model: {self.wav2vec2_model_name}")
            self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                self.wav2vec2_model_name
            )
            self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(
                self.wav2vec2_model_name
            )
            self.wav2vec2_model.to(self.device)
            logger.info("✅ WAV2VEC2 model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading STT models: {e}")
            raise RuntimeError(f"Failed to load STT models: {e}")
    
    def _preprocess_audio(self, audio_path: Path, target_sr: int = 16000) -> np.ndarray:
        """
        Preprocess audio file for STT.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Preprocessed audio array
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Ensure mono
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise
    
    def transcribe_with_whisper(self, audio_path: Path) -> str:
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
    
    def transcribe_with_wav2vec2(self, audio_path: Path) -> str:
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
    
    def transcribe_audio(self, audio_path: Path) -> Dict[str, str]:
        """
        Transcribe audio using both Whisper and WAV2VEC2.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcriptions from both models
        """
        logger.info(f"Transcribing {audio_path.name}...")
        
        # Transcribe with both models
        whisper_text = self.transcribe_with_whisper(audio_path)
        wav2vec2_text = self.transcribe_with_wav2vec2(audio_path)
        
        return {
            "whisper": whisper_text,
            "wav2vec2": wav2vec2_text,
            "audio_path": str(audio_path)
        }
    
    
    def transcribe_segments(self, 
                           segment_paths: List[Path], 
                           output_dir: Path) -> Dict[str, Any]:
        """
        Transcribe multiple audio segments and save results separately.
        
        Args:
            segment_paths: List of audio segment paths
            output_dir: Output directory for results
            
        Returns:
            Dictionary with transcription results
        """
        logger.info(f"Transcribing {len(segment_paths)} segments...")
        
        # Create output directories
        whisper_dir = output_dir / 'STT-whisper'
        wav2vec2_dir = output_dir / 'STT-wav2vec2'
        
        whisper_dir.mkdir(parents=True, exist_ok=True)
        wav2vec2_dir.mkdir(parents=True, exist_ok=True)
        
        whisper_results = []
        wav2vec2_results = []
        
        for i, segment_path in enumerate(segment_paths):
            logger.info(f"Processing segment {i+1}/{len(segment_paths)}: {segment_path.name}")
            
            try:
                # Transcribe with Whisper
                whisper_text = self.transcribe_with_whisper(segment_path)
                whisper_file = whisper_dir / f"{segment_path.stem}_whisper.txt"
                with open(whisper_file, 'w', encoding='utf-8') as f:
                    f.write(whisper_text)
                whisper_results.append({
                    "segment": segment_path.name,
                    "transcription": whisper_text,
                    "file": str(whisper_file)
                })
                
                # Transcribe with WAV2VEC2
                wav2vec2_text = self.transcribe_with_wav2vec2(segment_path)
                wav2vec2_file = wav2vec2_dir / f"{segment_path.stem}_wav2vec2.txt"
                with open(wav2vec2_file, 'w', encoding='utf-8') as f:
                    f.write(wav2vec2_text)
                wav2vec2_results.append({
                    "segment": segment_path.name,
                    "transcription": wav2vec2_text,
                    "file": str(wav2vec2_file)
                })
                
                logger.info(f"✅ Transcribed {segment_path.name}")
                
            except Exception as e:
                logger.error(f"Error processing segment {segment_path.name}: {e}")
        
        logger.info(f"STT completed: {len(whisper_results)} Whisper transcriptions, {len(wav2vec2_results)} WAV2VEC2 transcriptions")
        
        return {
            "whisper_results": whisper_results,
            "wav2vec2_results": wav2vec2_results,
            "whisper_dir": str(whisper_dir),
            "wav2vec2_dir": str(wav2vec2_dir),
            "total_segments": len(segment_paths),
            "whisper_count": len(whisper_results),
            "wav2vec2_count": len(wav2vec2_results)
        }
    
