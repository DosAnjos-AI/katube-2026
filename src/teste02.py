# ~/katube_teste/katube-novo/src/teste_whisper_transcribe.py
import torch
import librosa
import numpy as np
from pathlib import Path
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_transcribe():
    """Testa transcrição do primeiro segmento que travou"""
    
    # Path do segmento que travou (ADAPTAR se necessário)
    audio_path = Path("/home/ubuntu/katube_teste/katube-novo/audios_baixados/output/EhzSC3LWez4/stt_ready/speaker_SPEAKER_00/EhzSC3LWez4_segment_000_SPEAKER_00_1.43_24.41.flac")
    
    if not audio_path.exists():
        logger.error(f"❌ Arquivo não encontrado: {audio_path}")
        logger.info("⚠️ Ajuste o path manualmente no código do teste")
        return False
    
    model_name = "freds0/distil-whisper-large-v3-ptbr"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        logger.info(f"🎵 Loading audio: {audio_path.name}")
        logger.info(f"📍 Path completo: {audio_path}")
        logger.info(f"📏 Tamanho arquivo: {audio_path.stat().st_size / 1024:.2f} KB")
        
        # Carregar áudio
        audio, sr = librosa.load(audio_path, sr=16000)
        
        logger.info(f"✅ Audio loaded: {len(audio) / sr:.2f}s at {sr}Hz")
        logger.info(f"🔢 Shape: {audio.shape}, dtype: {audio.dtype}")
        
        # Normalizar
        audio = audio / np.max(np.abs(audio))
        logger.info("✅ Audio normalized")
        
        # Carregar modelo
        logger.info(f"🤖 Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        model = model.to(device)
        logger.info(f"✅ Model loaded on {device}")
        
        # Process audio
        logger.info("🎤 Processing audio...")
        input_features = processor(
            audio, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(device)
        
        logger.info(f"✅ Input features: shape={input_features.shape}")
        
        # Generate transcription
        logger.info("📝 Generating transcription...")
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
            transcription = processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0]
        
        logger.info("🎉 TRANSCRIPTION SUCCESS!")
        logger.info(f"📄 Result: {transcription}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Whisper transcribe test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_whisper_transcribe()