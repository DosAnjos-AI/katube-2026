"""
Configuration settings for the audio processing pipeline.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Audio settings
    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', '24000'))
    SEGMENT_MIN_DURATION = float(os.getenv('SEGMENT_MIN_DURATION', '4.0'))
    SEGMENT_MAX_DURATION = float(os.getenv('SEGMENT_MAX_DURATION', '18.0'))
    SEGMENT_OVERLAP = float(os.getenv('SEGMENT_OVERLAP', '0.5'))
    
    # Diarization settings
    PYANNOTE_MODEL = os.getenv('PYANNOTE_MODEL', 'pyannote/speaker-diarization-3.1')
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    
    # Voice overlap detection
    OVERLAP_THRESHOLD = float(os.getenv('OVERLAP_THRESHOLD', '0.9'))  # Increased from 0.5 to 0.9 (90% overlap required to classify as overlapping)
    MIN_SPEECH_DURATION = float(os.getenv('MIN_SPEECH_DURATION', '0.5'))
    
    # Audio segmentation limits
    MAX_SEGMENTS = int(os.getenv('MAX_SEGMENTS', '5000'))  # Maximum segments per audio
    
    # MOS Quality Filter settings (OBRIGATÓRIO)
    MOS_THRESHOLD = float(os.getenv('MOS_THRESHOLD', '3.0'))  # Minimum MOS score to accept
    # ENABLE_MOS_FILTER sempre True - filtro é obrigatório
    AUDIO_FORMAT = 'flac'  # ou 'wav'
    # Directories
    BASE_DIR = Path(__file__).parent.parent
    # Detecta diretório automaticamente ou usa variável de ambiente
    if os.getenv('AUDIOS_BAIXADOS_DIR'):
        AUDIOS_BAIXADOS_DIR = Path(os.getenv('AUDIOS_BAIXADOS_DIR'))
    else:
        # Usa pasta relativa ao projeto (funciona em qualquer SO)
        AUDIOS_BAIXADOS_DIR = BASE_DIR / 'audios_baixados'
        OUTPUT_DIR = AUDIOS_BAIXADOS_DIR / "output"
        
    # STT preparation settings
    MAX_SEGMENT_SIZE = 25 * 1024 * 1024  # 25MB max per segment for STT
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories."""
        for dir_path in [cls.OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
