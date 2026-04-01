"""
Warning suppression configuration
"""
import warnings
import logging

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="webrtcvad")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.*moved.*")
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.AudioMetaData.*moved.*")
warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm.*deprecated.*")

# Suppress specific connection warnings
warnings.filterwarnings("ignore", message=".*Could not load Wav2Vec2 model.*")
warnings.filterwarnings("ignore", message=".*401 Client Error.*")

# Suppress specific module warnings
warnings.filterwarnings("ignore", message=".*ESPnet is not installed.*")

# Set logging level for specific modules
logging.getLogger("webrtcvad").setLevel(logging.ERROR)
logging.getLogger("pyannote.audio").setLevel(logging.ERROR)
logging.getLogger("speechbrain").setLevel(logging.ERROR)
logging.getLogger("torchaudio").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("s3prl").setLevel(logging.ERROR)
