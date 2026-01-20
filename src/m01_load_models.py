"""
Módulo de gerenciamento e carregamento de modelos de ML/AI
Responsável por detectar dispositivo (CPU/GPU), carregar modelos,
testar inferência e manter modelos em memória durante toda execução.

Padrão: Singleton - uma única instância durante toda execução
Filosofia: KISS - implementação incremental de modelos
"""

import torch
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import time
import gc

# Importa configurações do projeto
from config import MODELS


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Timeouts do config (conservadores para diferentes máquinas)
DOWNLOAD_TIMEOUT = MODELS['timeouts']['download_segundos']
LOAD_TIMEOUT = MODELS['timeouts']['load_segundos']
INFERENCE_TIMEOUT = MODELS['timeouts']['inference_segundos']


class ModelManager:
    """
    Gerenciador central de modelos ML/AI.
    
    Responsabilidades:
    - Detectar dispositivo disponível (CPU/GPU)
    - Carregar modelos de forma incremental
    - Testar inferência em cada modelo
    - Manter modelos em memória durante execução
    - Reportar status e uso de recursos
    
    Padrão Singleton: apenas uma instância por execução
    """
    
    _instance = None
    
    def __new__(cls, device: Optional[str] = None):
        """
        Implementação do padrão Singleton.
        Garante apenas uma instância do ModelManager.
        """
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, device: Optional[str] = None):
        """
        Inicializa o gerenciador de modelos.
        
        Args:
            device: Dispositivo para carregar modelos (opcional)
                   None (padrão) - usa configuração do config.py
                   "auto" - detecta automaticamente
                   "cpu" - força CPU
                   "cuda" - força GPU
        """
        # Evita reinicialização em instâncias subsequentes
        if self._initialized:
            return
        
        # Se device não especificado, usa config
        if device is None:
            device = MODELS['device']
            logger.info(f"Device obtido do config.py: '{device}'")
        else:
            logger.info(f"Device manual (override): '{device}'")
            
        self.device_config = device
        self.device = None
        self.models = {}  # Dicionário para armazenar modelos carregados
        self.model_status = {}  # Status de cada modelo
        
        # Detecta e configura dispositivo
        self._detect_device()
        
        # Log inicial
        logger.info(f"ModelManager inicializado - Device: {self.device}")
        if self.device.type == "cuda":
            self._log_gpu_info()
        
        self._initialized = True
    
    def _detect_device(self) -> None:
        """
        Detecta o dispositivo disponível baseado na configuração.
        
        Lógica:
        - "auto": detecta CUDA, fallback para CPU
        - "cuda": força GPU (falha se não disponível)
        - "cpu": força CPU
        """
        if self.device_config == "cpu":
            self.device = torch.device("cpu")
            logger.info("Dispositivo configurado: CPU (forçado)")
            
        elif self.device_config == "cuda":
            if not torch.cuda.is_available():
                error_msg = "GPU (CUDA) não disponível, mas foi configurado device='cuda'"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            self.device = torch.device("cuda")
            logger.info("Dispositivo configurado: GPU/CUDA (forçado)")
            
        elif self.device_config == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Dispositivo detectado: GPU/CUDA (automático)")
            else:
                self.device = torch.device("cpu")
                logger.info("Dispositivo detectado: CPU (automático - GPU não disponível)")
        else:
            error_msg = f"Configuração de device inválida: '{self.device_config}'. Use 'auto', 'cpu' ou 'cuda'"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _log_gpu_info(self) -> None:
        """
        Loga informações detalhadas sobre GPU disponível.
        Útil para debug e monitoramento de recursos.
        """
        if self.device.type != "cuda":
            return
        
        try:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            logger.info(f"GPU detectada: {gpu_name}")
            logger.info(f"VRAM total: {total_memory:.2f} GB")
        except Exception as e:
            logger.warning(f"Não foi possível obter informações da GPU: {e}")
    
    def _check_vram_available(self) -> Dict[str, float]:
        """
        Verifica memória VRAM disponível (apenas para GPU).
        
        Returns:
            Dict com informações de memória:
            - total_gb: VRAM total
            - allocated_gb: VRAM alocada
            - reserved_gb: VRAM reservada
            - free_gb: VRAM livre estimada
        """
        if self.device.type != "cuda":
            return {
                "total_gb": 0.0,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": 0.0
            }
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - reserved
            
            return {
                "total_gb": round(total, 2),
                "allocated_gb": round(allocated, 2),
                "reserved_gb": round(reserved, 2),
                "free_gb": round(free, 2)
            }
        except Exception as e:
            logger.error(f"Erro ao verificar VRAM: {e}")
            return {
                "total_gb": 0.0,
                "allocated_gb": 0.0,
                "reserved_gb": 0.0,
                "free_gb": 0.0
            }
    
    def _cleanup_memory(self) -> None:
        """
        Limpa memória não utilizada (CPU e GPU).
        Útil após descarregar modelos ou entre operações pesadas.
        """
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Memória GPU limpa")
    
    def _test_model_inference(
        self, 
        model_name: str, 
        model: Any, 
        test_function: callable
    ) -> bool:
        """
        Testa inferência de um modelo com timeout.
        
        Args:
            model_name: Nome do modelo para logging
            model: Instância do modelo carregado
            test_function: Função que executa teste de inferência
                          Deve retornar True se sucesso, False caso contrário
        
        Returns:
            bool: True se teste passou, False caso contrário
        """
        logger.info(f"Testando inferência do modelo: {model_name}")
        start_time = time.time()
        
        try:
            # TODO: Implementar timeout real com threading/multiprocessing
            # Por enquanto, assume que test_function é rápida
            success = test_function(model)
            
            elapsed = time.time() - start_time
            
            if success:
                logger.info(f"Teste de inferência OK - {model_name} ({elapsed:.2f}s)")
                return True
            else:
                logger.error(f"Teste de inferência FALHOU - {model_name}")
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Erro no teste de inferência - {model_name}: {e} ({elapsed:.2f}s)")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status completo do ModelManager e modelos carregados.
        
        Returns:
            Dict com informações:
            - device: dispositivo em uso
            - vram: informações de memória (se GPU)
            - models: status de cada modelo carregado
        """
        status = {
            "device": str(self.device),
            "device_type": self.device.type,
            "models_loaded": len(self.models),
            "models": {}
        }
        
        # Adiciona informações de VRAM se GPU
        if self.device.type == "cuda":
            status["vram"] = self._check_vram_available()
        
        # Adiciona status de cada modelo
        for model_name, model_info in self.model_status.items():
            status["models"][model_name] = model_info
        
        return status
    
    def unload_model(self, model_name: str) -> bool:
        """
        Descarrega um modelo específico da memória.
        
        Args:
            model_name: Nome do modelo a descarregar
        
        Returns:
            bool: True se descarregado com sucesso
        """
        if model_name not in self.models:
            logger.warning(f"Modelo '{model_name}' não está carregado")
            return False
        
        try:
            # Remove referências
            del self.models[model_name]
            if model_name in self.model_status:
                self.model_status[model_name]["loaded"] = False
            
            # Limpa memória
            self._cleanup_memory()
            
            logger.info(f"Modelo '{model_name}' descarregado com sucesso")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao descarregar modelo '{model_name}': {e}")
            return False
    
    # =========================================================================
    # PLACEHOLDER: Métodos de carregamento de modelos específicos
    # Serão implementados incrementalmente
    # =========================================================================
    
    def load_whisper(self) -> bool:
        """
        Carrega modelo Whisper para transcrição.
        
        TODO: Implementar carregamento
        - Modelo: freds0/distil-whisper-large-v3-ptbr
        - Biblioteca: HuggingFace Transformers
        - VRAM estimada: 6-8 GB (float32)
        
        Returns:
            bool: True se carregado com sucesso
        """
        logger.warning("load_whisper() ainda não implementado")
        return False
    
    def load_wav2vec2(self) -> bool:
        """
        Carrega modelo WAV2VEC2 para transcrição.
        
        TODO: Implementar carregamento
        - Modelo: alefiury/wav2vec2-large-xlsr-53-coraa-brazilian-portuguese-gain-normalization
        - Biblioteca: HuggingFace Transformers
        - VRAM estimada: 3-4 GB (float32)
        
        Returns:
            bool: True se carregado com sucesso
        """
        logger.warning("load_wav2vec2() ainda não implementado")
        return False
    
    def load_pyannote(self) -> bool:
        """
        Carrega modelo Pyannote para segmentação de speakers.
        
        TODO: Implementar carregamento
        - Modelo: pyannote/segmentation-3.0
        - Biblioteca: pyannote.audio
        - VRAM estimada: 1.5-2 GB (float32)
        
        Returns:
            bool: True se carregado com sucesso
        """
        logger.warning("load_pyannote() ainda não implementado")
        return False


# =========================================================================
# Função auxiliar para facilitar uso
# =========================================================================

def get_model_manager(device: Optional[str] = None) -> ModelManager:
    """
    Retorna instância singleton do ModelManager.
    
    Args:
        device: Dispositivo opcional (None usa config.py)
                "auto" - detecta automaticamente
                "cpu" - força CPU
                "cuda" - força GPU
    
    Returns:
        ModelManager: Instância única do gerenciador
    
    Exemplos:
        # Usa configuração do config.py
        manager = get_model_manager()
        
        # Override manual
        manager = get_model_manager(device="cpu")
    """
    return ModelManager(device=device)


# =========================================================================
# Teste básico do módulo
# =========================================================================

if __name__ == "__main__":
    print("=== Teste do ModelManager ===\n")
    
    # Cria instância usando config.py
    print("Testando com device do config.py...")
    manager = get_model_manager()
    
    # Mostra status inicial
    status = manager.get_status()
    print(f"Device: {status['device']}")
    print(f"Device Type: {status['device_type']}")
    
    if status['device_type'] == "cuda":
        vram = status['vram']
        print(f"\nVRAM Info:")
        print(f"  Total: {vram['total_gb']:.2f} GB")
        print(f"  Alocada: {vram['allocated_gb']:.2f} GB")
        print(f"  Reservada: {vram['reserved_gb']:.2f} GB")
        print(f"  Livre: {vram['free_gb']:.2f} GB")
    
    print(f"\nModelos carregados: {status['models_loaded']}")
    
    print("\n--- Timeouts configurados ---")
    print(f"Download: {DOWNLOAD_TIMEOUT}s")
    print(f"Load: {LOAD_TIMEOUT}s")
    print(f"Inference: {INFERENCE_TIMEOUT}s")
    
    print("\n=== Teste concluído ===")