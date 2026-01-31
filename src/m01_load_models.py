#!/usr/bin/env python3
"""
Modulo m01_model_manager.py
Gerenciador centralizado de modelos de IA usando padrao Singleton
Carrega modelos 1x e reutiliza entre multiplas execucoes
"""

import sys
from pathlib import Path
from typing import Optional, Any, Tuple
import torch

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    MASTER,
    MOS_FILTER,
    OVERLAP_DETECTOR,
    STT_WHISPER,
    STT_WAV2VEC2,
    DEEPFILTERNET_DENOISER
)


# ==============================================================================
# CONSTANTES - IDs DOS MODELOS (HARDCODED)
# ==============================================================================

WHISPER_MODEL_ID = "freds0/distil-whisper-large-v3-ptbr"
WAV2VEC_MODEL_ID = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese"
PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"


# ==============================================================================
# CLASSE SINGLETON - GERENCIADOR DE MODELOS
# ==============================================================================

class ModelManager:
    """
    Gerenciador singleton de modelos de IA
    Garante carregamento unico e reutilizacao de instancias
    """
    
    _instance = None  # Instancia unica do singleton
    
    def __new__(cls):
        """Implementacao do padrao Singleton"""
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Inicializa gerenciador (executa apenas 1x)"""
        if self._initialized:
            return
        
        # Cache de modelos carregados
        self._whisper = None
        self._wav2vec = None
        self._pyannote = None
        self._squim = None
        self._deepfilternet = None
        
        # Marca como inicializado
        self._initialized = True
        
        print("\n" + "="*70)
        print("MODEL MANAGER INICIALIZADO")
        print("="*70)
        print("Modelos serao carregados sob demanda (lazy loading)")
        print("="*70 + "\n")
    
    # ==========================================================================
    # METODOS AUXILIARES - DEVICE MANAGEMENT
    # ==========================================================================
    
    def _obter_device(self, config_device: str) -> str:
        """
        Determina device baseado em configuracao
        
        Args:
            config_device: Valor do config ("auto", "gpu", "cpu")
            
        Returns:
            "cuda" ou "cpu"
        """
        device_config = config_device.lower()
        
        if device_config == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device_config == "gpu":
            if not torch.cuda.is_available():
                print("AVISO: GPU solicitada mas CUDA nao disponivel. Usando CPU.")
                return "cpu"
            return "cuda"
        else:  # cpu
            return "cpu"
    
    def _obter_device_id(self, device: str) -> int:
        """
        Converte device string para device_id (para transformers pipeline)
        
        Args:
            device: "cuda" ou "cpu"
            
        Returns:
            0 para GPU, -1 para CPU
        """
        return 0 if device == "cuda" else -1
    
    # ==========================================================================
    # WHISPER - STT
    # ==========================================================================
    
    def get_whisper(self) -> Any:
        """
        Obtem pipeline Whisper (carrega 1x, reutiliza depois)
        
        Returns:
            Pipeline do transformers para Whisper
            
        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar
        """
        # Retorna se ja carregado
        if self._whisper is not None:
            return self._whisper
        
        # Verifica se modulo esta ativo no MASTER
        if not MASTER.get('transcricao_whisper', False):
            raise RuntimeError("ERRO: Whisper desabilitado no MASTER config")
        
        print("\n" + "-"*70)
        print("CARREGANDO MODELO: Whisper")
        print("-"*70)
        
        try:
            from transformers import pipeline
            
            # Obter device do bloco especifico
            device = self._obter_device(STT_WHISPER.get('device', 'auto'))
            device_id = self._obter_device_id(device)
            
            print(f"Modelo: {WHISPER_MODEL_ID}")
            print(f"Device: {device}")
            
            # Carregar modelo
            self._whisper = pipeline(
                "automatic-speech-recognition",
                model=WHISPER_MODEL_ID,
                device=device_id
            )
            
            print("✓ Whisper carregado com sucesso")
            print("-"*70 + "\n")
            
            return self._whisper
            
        except Exception as e:
            print(f"✗ ERRO ao carregar Whisper: {e}")
            print("-"*70 + "\n")
            raise
    
    # ==========================================================================
    # WAV2VEC - STT
    # ==========================================================================
    
    def get_wav2vec(self) -> Any:
        """
        Obtem pipeline wav2vec (carrega 1x, reutiliza depois)
        
        Returns:
            Pipeline do transformers para wav2vec
            
        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar
        """
        # Retorna se ja carregado
        if self._wav2vec is not None:
            return self._wav2vec
        
        # Verifica se modulo esta ativo no MASTER
        if not MASTER.get('transcricao_wav2vec', False):
            raise RuntimeError("ERRO: wav2vec desabilitado no MASTER config")
        
        print("\n" + "-"*70)
        print("CARREGANDO MODELO: wav2vec")
        print("-"*70)
        
        try:
            from transformers import pipeline
            
            # Obter device do bloco especifico
            device = self._obter_device(STT_WAV2VEC2.get('device', 'auto'))
            device_id = self._obter_device_id(device)
            
            print(f"Modelo: {WAV2VEC_MODEL_ID}")
            print(f"Device: {device}")
            
            # Carregar modelo
            self._wav2vec = pipeline(
                "automatic-speech-recognition",
                model=WAV2VEC_MODEL_ID,
                device=device_id
            )
            
            print("✓ wav2vec carregado com sucesso")
            print("-"*70 + "\n")
            
            return self._wav2vec
            
        except Exception as e:
            print(f"✗ ERRO ao carregar wav2vec: {e}")
            print("-"*70 + "\n")
            raise
    
    # ==========================================================================
    # PYANNOTE - OVERLAP DETECTION
    # ==========================================================================
    
    def get_pyannote(self) -> Any:
        """
        Obtem pipeline pyannote (carrega 1x, reutiliza depois)
        
        Returns:
            Pipeline pyannote.audio
            
        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar
        """
        # Retorna se ja carregado
        if self._pyannote is not None:
            return self._pyannote
        
        # Verifica se modulo esta ativo no MASTER
        if not MASTER.get('overlap', False):
            raise RuntimeError("ERRO: Overlap detector desabilitado no MASTER config")
        
        print("\n" + "-"*70)
        print("CARREGANDO MODELO: pyannote")
        print("-"*70)
        
        try:
            from pyannote.audio import Pipeline
            
            # Obter device do bloco especifico
            device = self._obter_device(OVERLAP_DETECTOR.get('device', 'auto'))
            
            # Token HuggingFace (opcional)
            hf_token = OVERLAP_DETECTOR.get('hf_token')
            
            print(f"Modelo: {PYANNOTE_MODEL_ID}")
            print(f"Device: {device}")
            
            # Carregar modelo
            if hf_token:
                self._pyannote = Pipeline.from_pretrained(
                    PYANNOTE_MODEL_ID,
                    token=hf_token
                )
            else:
                self._pyannote = Pipeline.from_pretrained(PYANNOTE_MODEL_ID)
            
            # Mover para device
            self._pyannote.to(torch.device(device))
            
            print("✓ pyannote carregado com sucesso")
            print("-"*70 + "\n")
            
            return self._pyannote
            
        except Exception as e:
            print(f"✗ ERRO ao carregar pyannote: {e}")
            print("-"*70 + "\n")
            raise
    
    # ==========================================================================
    # SQUIM - MOS QUALITY ASSESSMENT
    # ==========================================================================
    
    def get_squim(self) -> Any:
        """
        Obtem modelo SQUIM (carrega 1x, reutiliza depois)
        
        Returns:
            Modelo SQUIM do torchaudio
            
        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar
        """
        # Retorna se ja carregado
        if self._squim is not None:
            return self._squim
        
        # Verifica se modulo esta ativo no MASTER
        if not MASTER.get('mos_filter', False):
            raise RuntimeError("ERRO: MOS filter desabilitado no MASTER config")
        
        print("\n" + "-"*70)
        print("CARREGANDO MODELO: SQUIM")
        print("-"*70)
        
        try:
            import torchaudio
            
            # Obter device do bloco especifico
            device = self._obter_device(MOS_FILTER.get('device', 'auto'))
            
            print(f"Modelo: SQUIM_OBJECTIVE (torchaudio)")
            print(f"Device: {device}")
            
            # Carregar modelo
            self._squim = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
            self._squim = self._squim.to(device)
            
            print("✓ SQUIM carregado com sucesso")
            print("-"*70 + "\n")
            
            return self._squim
            
        except Exception as e:
            print(f"✗ ERRO ao carregar SQUIM: {e}")
            print("-"*70 + "\n")
            raise
    
    # ==========================================================================
    # DEEPFILTERNET3 - AUDIO DENOISING
    # ==========================================================================
    
    def get_deepfilternet(self) -> Tuple[Any, Any, int]:
        """
        Obtem modelo e estado DeepFilterNet (carrega 1x, reutiliza depois)
        
        Returns:
            Tupla (modelo, df_state, sample_rate)
            
        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar
        """
        # Retorna se ja carregado
        if self._deepfilternet is not None:
            return self._deepfilternet
        
        # Verifica se modulo esta ativo no MASTER
        if not MASTER.get('Denoiser', False):
            raise RuntimeError("ERRO: Denoiser desabilitado no MASTER config")
        
        print("\n" + "-"*70)
        print("CARREGANDO MODELO: DeepFilterNet3")
        print("-"*70)
        
        try:
            from df import init_df
            
            # Obter device do bloco especifico
            device = self._obter_device(DEEPFILTERNET_DENOISER.get('device', 'auto'))
            
            # Parametros do DeepFilterNet
            post_filter = DEEPFILTERNET_DENOISER.get('post_filter', 1)
            
            print(f"Modelo: DeepFilterNet3")
            print(f"Device: {device}")
            print(f"Post-filter: {post_filter}")
            
            # Carregar modelo
            modelo, df_state, _ = init_df(
                post_filter=post_filter,
                log_level="ERROR"  # Reduz verbosidade
            )
            
            # Mover modelo para device
            modelo = modelo.to(device)
            
            # Obter sample rate do df_state
            sr = df_state.sr()
            
            # Armazenar tupla completa
            self._deepfilternet = (modelo, df_state, sr)
            
            print(f"✓ DeepFilterNet3 carregado com sucesso (SR={sr} Hz)")
            print("-"*70 + "\n")
            
            return self._deepfilternet
            
        except Exception as e:
            print(f"✗ ERRO ao carregar DeepFilterNet3: {e}")
            print("-"*70 + "\n")
            raise
    
    # ==========================================================================
    # UTILIDADES - GESTAO DE MEMORIA
    # ==========================================================================
    
    def clear_cache(self):
        """Limpa cache de GPU (util para liberar VRAM)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("✓ Cache GPU limpo")
    
    def get_vram_usage(self) -> dict:
        """
        Obtem uso atual de VRAM
        
        Returns:
            Dict com informacoes de memoria GPU
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
        """Imprime status atual dos modelos carregados"""
        print("\n" + "="*70)
        print("STATUS DO MODEL MANAGER")
        print("="*70)
        print(f"Whisper carregado:       {'SIM' if self._whisper is not None else 'NAO'}")
        print(f"wav2vec carregado:       {'SIM' if self._wav2vec is not None else 'NAO'}")
        print(f"pyannote carregado:      {'SIM' if self._pyannote is not None else 'NAO'}")
        print(f"SQUIM carregado:         {'SIM' if self._squim is not None else 'NAO'}")
        print(f"DeepFilterNet carregado: {'SIM' if self._deepfilternet is not None else 'NAO'}")
        
        # Info de VRAM
        vram = self.get_vram_usage()
        if vram["available"]:
            print(f"\nVRAM alocada:  {vram['allocated_gb']:.2f} GB")
            print(f"VRAM reservada: {vram['reserved_gb']:.2f} GB")
            print(f"VRAM total:     {vram['total_gb']:.2f} GB")
        else:
            print("\nGPU: Nao disponivel (rodando em CPU)")
        
        print("="*70 + "\n")


# ==============================================================================
# FUNCAO DE CONVENIENCIA - OBTER INSTANCIA SINGLETON
# ==============================================================================

def get_manager() -> ModelManager:
    """
    Obtem instancia singleton do ModelManager
    
    Returns:
        Instancia unica do ModelManager
    """
    return ModelManager()


# ==============================================================================
# TESTE DO MODULO - CARREGA TODOS OS MODELOS HABILITADOS
# ==============================================================================

if __name__ == "__main__":
    print("TESTANDO MODEL MANAGER")
    print("="*70)
    print("Este teste carregara TODOS os modelos habilitados no MASTER")
    print("="*70 + "\n")
    
    # Criar instancia
    manager = get_manager()
    
    # Testar singleton
    manager2 = ModelManager()
    print(f"Singleton OK: {manager is manager2}\n")
    
    # Status inicial
    manager.print_status()
    
    # Carregar modelos conforme MASTER
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
    
    # Relatorio final
    print("\n" + "="*70)
    print("RELATORIO FINAL DE CARREGAMENTO")
    print("="*70)
    print(f"Modelos carregados com sucesso: {modelos_carregados}")
    print(f"Modelos com falha: {modelos_falhados}")
    print("="*70 + "\n")
    
    # Status final
    manager.print_status()
    
    print("\nTESTE CONCLUIDO")