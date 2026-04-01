# ~/katube_teste/katube-novo/src/teste_whisper_loading.py
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_whisper_loading():
    """Testa carregamento do modelo Whisper na GPU"""
    
    model_name = "freds0/distil-whisper-large-v3-ptbr"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"🔄 Carregando Whisper model: {model_name}")
    logger.info(f"📍 Device: {device}")
    logger.info(f"💾 CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"💾 VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    try:
        # Carregar processor
        logger.info("📦 Loading processor...")
        processor = WhisperProcessor.from_pretrained(model_name)
        logger.info("✅ Processor loaded")
        
        # Carregar model
        logger.info("🤖 Loading model...")
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        logger.info("✅ Model loaded (CPU)")
        
        # Mover para GPU
        logger.info(f"🚀 Moving model to {device}...")
        model = model.to(device)
        logger.info(f"✅ Model on {device}")
        
        # Verificar uso de memória
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
            logger.info(f"💾 VRAM Allocated: {memory_allocated:.2f} GB")
            logger.info(f"💾 VRAM Reserved: {memory_reserved:.2f} GB")
        
        logger.info("🎉 Whisper loading test: SUCCESS")
        return True
        
    except Exception as e:
        logger.error(f"❌ Whisper loading test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_whisper_loading()