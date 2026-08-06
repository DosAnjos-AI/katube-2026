#!/usr/bin/env python3
"""
Module m01_model_manager.py
Centralized AI model manager using the Singleton pattern
Loads models once and reuses them across multiple executions
"""

import inspect
import os
import sys
from importlib.metadata import PackageNotFoundError, version as versao_pacote
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from dotenv import load_dotenv

# Add root folder to the path to import config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env at the project root
load_dotenv(PROJECT_ROOT / '.env')

from config import (
    MASTER,
    MOS_FILTER,
    OVERLAP_DETECTOR,
    STT_WHISPER,
    STT_WAV2VEC2,
    DEEPFILTERNET_DENOISER
)


# ==============================================================================
# CONSTANTS - MODEL IDs (HARDCODED)
# ==============================================================================

WHISPER_MODEL_ID = "freds0/distil-whisper-large-v3-ptbr"
WAV2VEC_MODEL_ID = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese"


# ==============================================================================
# LOUD LOG - ERROR THAT CANNOT GO UNNOTICED
# ==============================================================================

def _log_erro(linhas: List[str]) -> None:
    """
    Logs an error at high visibility, on stderr, with the [ERRO] prefix.

    Used for the GPU-to-CPU fallback: a run that falls back to CPU
    without anyone noticing costs about 9.7x real time and passes as
    normal.
    """
    print("", file=sys.stderr)
    print("[ERRO] " + "=" * 62, file=sys.stderr)
    for linha in linhas:
        print(f"[ERRO] {linha}", file=sys.stderr)
    print("[ERRO] " + "=" * 62, file=sys.stderr)
    sys.stderr.flush()


# ==============================================================================
# COMPATIBILITY BETWEEN VERSIONS - AUTHENTICATION PARAMETER NAME
# ==============================================================================

# Names already used by libraries in the HuggingFace ecosystem for the
# same parameter. The order is the preference order: the new name first.
NOMES_PARAMETRO_TOKEN = ('token', 'use_auth_token')


def _kwarg_autenticacao(funcao: Callable, token: Optional[str],
                        pacote: str) -> Dict[str, Optional[str]]:
    """
    Discovers, in the signature of the installed version, which
    authentication parameter name the function accepts, and returns the
    kwarg ready for the call.

    The HuggingFace ecosystem renamed 'use_auth_token' to 'token'.
    Different versions of the same library coexist between the local
    machine and the server, and the signature - not the version number -
    is the fact that decides the call.

    Args:
        funcao: The function that will be called (e.g., Pipeline.from_pretrained)
        token: Token value to pass
        pacote: Package distribution name, only for the error message

    Returns:
        A one-item dictionary: {accepted_name: token}

    Raises:
        RuntimeError: If the signature does not accept any of the known
                      names. Proceeding without authentication is not an
                      option: the model is gated and the failure would
                      show up further down, far from the cause.
    """
    parametros = inspect.signature(funcao).parameters

    for nome in NOMES_PARAMETRO_TOKEN:
        if nome in parametros:
            return {nome: token}

    try:
        versao = versao_pacote(pacote)
    except PackageNotFoundError:
        versao = "desconhecida"

    raise RuntimeError(
        f"ERRO: {pacote} {versao} nao aceita nenhum parametro de autenticacao "
        f"conhecido em {funcao.__qualname__}.\n"
        f"Nomes procurados: {', '.join(NOMES_PARAMETRO_TOKEN)}\n"
        f"Parametros aceitos pela assinatura instalada: "
        f"{', '.join(parametros)}"
    )


# ==============================================================================
# SINGLETON CLASS - MODEL MANAGER
# ==============================================================================

class ModelManager:
    """
    Singleton manager for AI models
    Ensures single loading and reuse of instances
    """

    _instance = None  # Single singleton instance
    
    def __new__(cls):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initializes the manager (runs only once)"""
        if self._initialized:
            return
        
        # Cache of loaded models
        self._whisper = None
        self._wav2vec = None
        self._pyannote = None
        self._squim = None
        self._deepfilternet = None

        # Mark as initialized
        self._initialized = True
        
        print("\n" + "="*70)
        print("MODEL MANAGER INICIALIZADO")
        print("="*70)
        print("Modelos serao carregados sob demanda (lazy loading)")
        print("="*70 + "\n")
    
    # ==========================================================================
    # HELPER METHODS - DEVICE MANAGEMENT
    # ==========================================================================
    
    def _obter_device(self, config_device: str, nome_bloco: str) -> str:
        """
        Determines the device from the configuration.

        Accepts ONLY "gpu" or "cpu" - the "auto" value was eliminated
        from the project. There is no silent resolution or hidden
        default value.

        Args:
            config_device: value of the block's 'device' field
            nome_bloco: name of the config block, for the error message

        Returns:
            "cuda" or "cpu"

        Raises:
            ValueError: invalid device value
            RuntimeError: "gpu" requested without CUDA and MASTER['fallback_cpu'] False
        """
        device_config = str(config_device).lower()

        if device_config == "cpu":
            return "cpu"

        if device_config != "gpu":
            raise ValueError(
                f"{nome_bloco}['device'] invalido: {config_device!r}. "
                "Valores aceitos: 'gpu' ou 'cpu'"
            )

        if torch.cuda.is_available():
            return "cuda"

        # GPU requested and CUDA unavailable: fallback_cpu is the decider
        if not MASTER.get('fallback_cpu', False):
            raise RuntimeError(
                f"{nome_bloco}['device']='gpu' mas CUDA nao esta disponivel "
                "e MASTER['fallback_cpu'] e False"
            )

        _log_erro([
            f"{nome_bloco}['device']='gpu' mas CUDA nao esta disponivel",
            "MASTER['fallback_cpu']=True - caindo para CPU",
            "DISPOSITIVO EFETIVO: cpu",
        ])
        return "cpu"

    def _fallback_para_cpu(self, erro: Exception, device: str,
                           nome_modelo: str, nome_bloco: str) -> bool:
        """
        Decides whether a failed load should be redone on CPU.

        Args:
            erro: original loading exception
            device: device on which the attempt failed (None if the
                    failure happened before the device was resolved)
            nome_modelo: model name, for the log
            nome_bloco: config block, for the log

        Returns:
            True if the caller should reload on CPU; False if the error
            should bring the module down.
        """
        # Failure on CPU, or before resolving the device, has nowhere to fall back to
        if device != "cuda":
            return False

        if not MASTER.get('fallback_cpu', False):
            _log_erro([
                f"{nome_modelo}: falha ao carregar em GPU "
                f"({nome_bloco}['device']='gpu')",
                f"Excecao original: {type(erro).__name__}: {erro}",
                "MASTER['fallback_cpu']=False - sem fallback, o modulo vai falhar",
            ])
            return False

        _log_erro([
            f"{nome_modelo}: falha ao carregar em GPU "
            f"({nome_bloco}['device']='gpu')",
            f"Excecao original: {type(erro).__name__}: {erro}",
            "MASTER['fallback_cpu']=True - recarregando em CPU",
            "DISPOSITIVO EFETIVO: cpu",
        ])
        return True


    def _obter_device_id(self, device: str) -> int:
        """
        Converts the device string to a device_id (for the transformers
        pipeline)

        Args:
            device: "cuda" or "cpu"

        Returns:
            0 for GPU, -1 for CPU
        """
        return 0 if device == "cuda" else -1
    
    # ==========================================================================
    # WHISPER - STT
    # ==========================================================================
    
    def _carregar_whisper(self, device: str) -> Any:
        """Loads the Whisper pipeline on the given device"""
        from transformers import pipeline

        print(f"Modelo: {WHISPER_MODEL_ID}")
        print(f"Device: {device}")

        return pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL_ID,
            device=self._obter_device_id(device)
        )

    def get_whisper(self) -> Any:
        """
        Gets the Whisper pipeline (loads once, reuses afterward)

        Returns:
            transformers pipeline for Whisper

        Raises:
            RuntimeError: If the module is disabled in MASTER
            Exception: If loading fails and there is no fallback
        """
        # Return if already loaded
        if self._whisper is not None:
            return self._whisper

        # Check whether the module is active in MASTER
        if not MASTER.get('transcricao_whisper', False):
            raise RuntimeError("ERRO: Whisper desabilitado no MASTER config")

        print("\n" + "-"*70)
        print("CARREGANDO MODELO: Whisper")
        print("-"*70)

        device = None
        try:
            device = self._obter_device(STT_WHISPER['device'], 'STT_WHISPER')
            self._whisper = self._carregar_whisper(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'Whisper', 'STT_WHISPER'):
                print(f"ERRO ao carregar Whisper: {e}")
                print("-"*70 + "\n")
                raise
            # Retry on CPU: if this also fails, the error propagates
            self._whisper = self._carregar_whisper("cpu")

        print("Whisper carregado com sucesso")
        print("-"*70 + "\n")

        return self._whisper
    
    # ==========================================================================
    # WAV2VEC - STT
    # ==========================================================================
    
    def _carregar_wav2vec(self, device: str) -> Any:
        """Loads the wav2vec pipeline on the given device"""
        from transformers import pipeline

        print(f"Modelo: {WAV2VEC_MODEL_ID}")
        print(f"Device: {device}")

        return pipeline(
            "automatic-speech-recognition",
            model=WAV2VEC_MODEL_ID,
            device=self._obter_device_id(device)
        )

    def get_wav2vec(self) -> Any:
        """
        Gets the wav2vec pipeline (loads once, reuses afterward)

        Returns:
            transformers pipeline for wav2vec

        Raises:
            RuntimeError: If the module is disabled in MASTER
            Exception: If loading fails and there is no fallback
        """
        # Return if already loaded
        if self._wav2vec is not None:
            return self._wav2vec

        # Check whether the module is active in MASTER
        if not MASTER.get('transcricao_wav2vec', False):
            raise RuntimeError("ERRO: wav2vec desabilitado no MASTER config")

        print("\n" + "-"*70)
        print("CARREGANDO MODELO: wav2vec")
        print("-"*70)

        device = None
        try:
            device = self._obter_device(STT_WAV2VEC2['device'], 'STT_WAV2VEC2')
            self._wav2vec = self._carregar_wav2vec(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'wav2vec', 'STT_WAV2VEC2'):
                print(f"ERRO ao carregar wav2vec: {e}")
                print("-"*70 + "\n")
                raise
            # Retry on CPU: if this also fails, the error propagates
            self._wav2vec = self._carregar_wav2vec("cpu")

        print("wav2vec carregado com sucesso")
        print("-"*70 + "\n")

        return self._wav2vec
    
    # ==========================================================================
    # PYANNOTE - OVERLAP DETECTION
    # ==========================================================================
    
    def _carregar_pyannote(self, device: str) -> Any:
        """Loads the pyannote pipeline on the given device"""
        from pyannote.audio import Pipeline

        # Model comes from config - there is no more hardcoded constant
        modelo_id = OVERLAP_DETECTOR['modelo']

        # HuggingFace token — read from .env
        hf_token = os.getenv('HF_TOKEN')

        print(f"Modelo: {modelo_id}")
        print(f"Device: {device}")

        # The authentication parameter name changes between pyannote
        # versions: it is read from the installed signature, never assumed.
        pipeline = Pipeline.from_pretrained(
            modelo_id,
            **_kwarg_autenticacao(
                Pipeline.from_pretrained, hf_token, 'pyannote.audio'
            )
        )

        # Move to device
        pipeline.to(torch.device(device))

        return pipeline

    def get_pyannote(self) -> Any:
        """
        Gets the pyannote pipeline (loads once, reuses afterward)

        Returns:
            pyannote.audio pipeline

        Raises:
            RuntimeError: If the module is disabled in MASTER
            Exception: If loading fails and there is no fallback
        """
        # Return if already loaded
        if self._pyannote is not None:
            return self._pyannote

        # Check whether the module is active in MASTER
        if not MASTER.get('overlap', False):
            raise RuntimeError("ERRO: Overlap detector desabilitado no MASTER config")

        print("\n" + "-"*70)
        print("CARREGANDO MODELO: pyannote")
        print("-"*70)

        device = None
        try:
            device = self._obter_device(OVERLAP_DETECTOR['device'], 'OVERLAP_DETECTOR')
            self._pyannote = self._carregar_pyannote(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'pyannote', 'OVERLAP_DETECTOR'):
                print(f"ERRO ao carregar pyannote: {e}")
                print("-"*70 + "\n")
                raise
            # Retry on CPU: if this also fails, the error propagates
            self._pyannote = self._carregar_pyannote("cpu")

        print("pyannote carregado com sucesso")
        print("-"*70 + "\n")

        return self._pyannote
    
    # ==========================================================================
    # SQUIM - MOS QUALITY ASSESSMENT
    # ==========================================================================
    
    def _carregar_squim(self, device: str) -> Any:
        """Loads the SQUIM model on the given device"""
        import torchaudio

        print(f"Modelo: SQUIM_OBJECTIVE (torchaudio)")
        print(f"Device: {device}")

        modelo = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        return modelo.to(device)

    def get_squim(self) -> Any:
        """
        Gets the SQUIM model (loads once, reuses afterward)

        Returns:
            torchaudio SQUIM model

        Raises:
            RuntimeError: If the module is disabled in MASTER
            Exception: If loading fails and there is no fallback
        """
        # Return if already loaded
        if self._squim is not None:
            return self._squim

        # Check whether the module is active in MASTER
        if not MASTER.get('mos_filter', False):
            raise RuntimeError("ERRO: MOS filter desabilitado no MASTER config")

        print("\n" + "-"*70)
        print("CARREGANDO MODELO: SQUIM")
        print("-"*70)

        device = None
        try:
            device = self._obter_device(MOS_FILTER['device'], 'MOS_FILTER')
            self._squim = self._carregar_squim(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'SQUIM', 'MOS_FILTER'):
                print(f"ERRO ao carregar SQUIM: {e}")
                print("-"*70 + "\n")
                raise
            # Retry on CPU: if this also fails, the error propagates
            self._squim = self._carregar_squim("cpu")

        print("SQUIM carregado com sucesso")
        print("-"*70 + "\n")

        return self._squim
    
    # ==========================================================================
    # DEEPFILTERNET3 - AUDIO DENOISING
    # ==========================================================================
    
    def _carregar_deepfilternet(self, device: str) -> Tuple[Any, Any, int]:
        """
        Loads DeepFilterNet3 and forces the library onto the device from
        the config.

        Finding A29: init_df() does not accept a device parameter, and
        enhance() resolves the device on its own, via get_device(),
        which picks cuda:0 whenever CUDA is available. Without the
        adjustment below, on a server with a GPU and device='cpu' the
        model stays on CPU while the features go to cuda:0.

        The adjustment has to come AFTER init_df(): init_df() itself
        reloads DeepFilterNet's internal config parser and would erase
        the value.
        """
        from df import init_df
        from df.config import config as df_config
        from df.utils import get_device as df_get_device

        post_filter = DEEPFILTERNET_DENOISER['post_filter']

        print(f"Modelo: DeepFilterNet3")
        print(f"Device: {device}")
        print(f"Post-filter: {post_filter}")

        modelo, df_state, _ = init_df(
            post_filter=post_filter,
            log_level="ERROR"  # Reduce verbosity
        )

        # A29: force the library's get_device() onto the config's device
        df_config.set("DEVICE", device, str, section="train")

        # Move model to device
        modelo = modelo.to(device)

        # Log check: the internal device has to match what was requested.
        # A divergence here is finding A29 manifesting itself - it cannot
        # pass silently on the first run with GPU.
        device_interno = str(df_get_device())
        print(f"Device interno do DeepFilterNet (get_device): {device_interno}")
        if device_interno.split(':')[0] != device:
            _log_erro([
                "DeepFilterNet3: dispositivo interno DIVERGE do config (achado A29)",
                f"Pedido pelo config: {device}",
                f"get_device() da biblioteca: {device_interno}",
                "As features irao para um dispositivo diferente do modelo",
            ])

        return (modelo, df_state, df_state.sr())

    def get_deepfilternet(self) -> Tuple[Any, Any, int]:
        """
        Gets the DeepFilterNet model and state (loads once, reuses
        afterward)

        Returns:
            Tuple (model, df_state, sample_rate)

        Raises:
            RuntimeError: If the module is disabled in MASTER
            Exception: If loading fails and there is no fallback

        KNOWN LIMITATION (decision recorded, instruction 20): the CPU
        retry calls init_df() again, and init_df() goes back to choosing
        the device on its own. On a GPU that fails, this model's
        fallback may not complete - unlike the other four.
        """
        # Return if already loaded
        if self._deepfilternet is not None:
            return self._deepfilternet

        # Check whether the module is active in MASTER
        if not MASTER.get('Denoiser', False):
            raise RuntimeError("ERRO: Denoiser desabilitado no MASTER config")

        print("\n" + "-"*70)
        print("CARREGANDO MODELO: DeepFilterNet3")
        print("-"*70)

        device = None
        try:
            device = self._obter_device(DEEPFILTERNET_DENOISER['device'],
                                        'DEEPFILTERNET_DENOISER')
            self._deepfilternet = self._carregar_deepfilternet(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'DeepFilterNet3',
                                           'DEEPFILTERNET_DENOISER'):
                print(f"ERRO ao carregar DeepFilterNet3: {e}")
                print("-"*70 + "\n")
                raise
            # Retry on CPU: if this also fails, the error propagates
            self._deepfilternet = self._carregar_deepfilternet("cpu")

        sr = self._deepfilternet[2]
        print(f"DeepFilterNet3 carregado com sucesso (SR={sr} Hz)")
        print("-"*70 + "\n")

        return self._deepfilternet
    
    # ==========================================================================
    # UTILITIES - MEMORY MANAGEMENT
    # ==========================================================================
    
    def clear_cache(self):
        """Clears the GPU cache (useful for freeing VRAM)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cache GPU limpo")
    
    def get_vram_usage(self) -> dict:
        """
        Gets current VRAM usage

        Returns:
            Dict with GPU memory information
        """
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3
        }
    
    def print_status(self):
        """Prints the current status of the loaded models"""
        print("\n" + "="*70)
        print("STATUS DO MODEL MANAGER")
        print("="*70)
        print(f"Whisper carregado:       {'SIM' if self._whisper is not None else 'NAO'}")
        print(f"wav2vec carregado:       {'SIM' if self._wav2vec is not None else 'NAO'}")
        print(f"pyannote carregado:      {'SIM' if self._pyannote is not None else 'NAO'}")
        print(f"SQUIM carregado:         {'SIM' if self._squim is not None else 'NAO'}")
        print(f"DeepFilterNet carregado: {'SIM' if self._deepfilternet is not None else 'NAO'}")
        
        # VRAM info
        vram = self.get_vram_usage()
        if vram["available"]:
            print(f"\nVRAM alocada:  {vram['allocated_gb']:.2f} GB")
            print(f"VRAM reservada: {vram['reserved_gb']:.2f} GB")
            print(f"VRAM total:     {vram['total_gb']:.2f} GB")
        else:
            print("\nGPU: Nao disponivel (rodando em CPU)")
        
        print("="*70 + "\n")


# ==============================================================================
# CONVENIENCE FUNCTION - GET SINGLETON INSTANCE
# ==============================================================================

def get_manager() -> ModelManager:
    """
    Gets the singleton instance of ModelManager

    Returns:
        The single instance of ModelManager
    """
    return ModelManager()


# ==============================================================================
# MODULE TEST - LOADS ALL ENABLED MODELS
# ==============================================================================

if __name__ == "__main__":
    print("TESTANDO MODEL MANAGER")
    print("="*70)
    print("Este teste carregara TODOS os modelos habilitados no MASTER")
    print("="*70 + "\n")
    
    # Create instance
    manager = get_manager()

    # Test singleton
    manager2 = ModelManager()
    print(f"Singleton OK: {manager is manager2}\n")

    # Initial status
    manager.print_status()

    # Load models according to MASTER
    modelos_carregados = 0
    modelos_falhados = 0
    
    print("="*70)
    print("INICIANDO CARREGAMENTO DOS MODELOS")
    print("="*70)
    
    # Whisper
    if MASTER.get('transcricao_whisper', False):
        try:
            manager.get_whisper()
            modelos_carregados += 1
        except Exception as e:
            print(f"[FALHA] Whisper nao pode ser carregado: {e}")
            modelos_falhados += 1
    else:
        print("[SKIP] Whisper desabilitado no MASTER")
    
    # wav2vec
    if MASTER.get('transcricao_wav2vec', False):
        try:
            manager.get_wav2vec()
            modelos_carregados += 1
        except Exception as e:
            print(f"[FALHA] wav2vec nao pode ser carregado: {e}")
            modelos_falhados += 1
    else:
        print("[SKIP] wav2vec desabilitado no MASTER")
    
    # pyannote
    if MASTER.get('overlap', False):
        try:
            manager.get_pyannote()
            modelos_carregados += 1
        except Exception as e:
            print(f"[FALHA] pyannote nao pode ser carregado: {e}")
            modelos_falhados += 1
    else:
        print("[SKIP] pyannote desabilitado no MASTER")
    
    # SQUIM
    if MASTER.get('mos_filter', False):
        try:
            manager.get_squim()
            modelos_carregados += 1
        except Exception as e:
            print(f"[FALHA] SQUIM nao pode ser carregado: {e}")
            modelos_falhados += 1
    else:
        print("[SKIP] SQUIM desabilitado no MASTER")
    
    # DeepFilterNet
    if MASTER.get('Denoiser', False):
        try:
            manager.get_deepfilternet()
            modelos_carregados += 1
        except Exception as e:
            print(f"[FALHA] DeepFilterNet nao pode ser carregado: {e}")
            modelos_falhados += 1
    else:
        print("[SKIP] DeepFilterNet desabilitado no MASTER")
    
    # Final report
    print("\n" + "="*70)
    print("RELATORIO FINAL DE CARREGAMENTO")
    print("="*70)
    print(f"Modelos carregados com sucesso: {modelos_carregados}")
    print(f"Modelos com falha: {modelos_falhados}")
    print("="*70 + "\n")

    # Final status
    manager.print_status()
    
    print("\nTESTE CONCLUIDO")