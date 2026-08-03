#!/usr/bin/env python3
"""
Modulo m01_model_manager.py
Gerenciador centralizado de modelos de IA usando padrao Singleton
Carrega modelos 1x e reutiliza entre multiplas execucoes
"""

import os
import sys
from pathlib import Path
from typing import Any, List, Tuple
import torch
from dotenv import load_dotenv

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Carregar variaveis de ambiente do .env na raiz do projeto
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
# CONSTANTES - IDs DOS MODELOS (HARDCODED)
# ==============================================================================

WHISPER_MODEL_ID = "freds0/distil-whisper-large-v3-ptbr"
WAV2VEC_MODEL_ID = "lgris/wav2vec2-large-xlsr-open-brazilian-portuguese"


# ==============================================================================
# LOG ALTO - ERRO QUE NAO PODE PASSAR DESPERCEBIDO
# ==============================================================================

def _log_erro(linhas: List[str]) -> None:
    """
    Registra erro em nivel alto, no stderr, com prefixo [ERRO].

    Usado para a queda de GPU para CPU: uma rodada que cai para CPU sem
    ninguem perceber custa cerca de 9,7x tempo real e passa por normal.
    """
    print("", file=sys.stderr)
    print("[ERRO] " + "=" * 62, file=sys.stderr)
    for linha in linhas:
        print(f"[ERRO] {linha}", file=sys.stderr)
    print("[ERRO] " + "=" * 62, file=sys.stderr)
    sys.stderr.flush()


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
    
    def _obter_device(self, config_device: str, nome_bloco: str) -> str:
        """
        Determina device a partir da configuracao.

        Aceita SOMENTE "gpu" ou "cpu" - o valor "auto" foi eliminado do
        projeto. Nao ha resolucao silenciosa nem valor padrao escondido.

        Args:
            config_device: valor do campo 'device' do bloco
            nome_bloco: nome do bloco do config, para a mensagem de erro

        Returns:
            "cuda" ou "cpu"

        Raises:
            ValueError: valor de device invalido
            RuntimeError: "gpu" pedida sem CUDA e MASTER['fallback_cpu'] False
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

        # GPU pedida e CUDA indisponivel: quem decide e o fallback_cpu
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
        Decide se um carregamento que falhou deve ser refeito em CPU.

        Args:
            erro: excecao original do carregamento
            device: device em que a tentativa falhou (None se a falha foi
                    anterior a resolucao do device)
            nome_modelo: nome do modelo, para o log
            nome_bloco: bloco do config, para o log

        Returns:
            True se o chamador deve recarregar em CPU; False se o erro
            deve derrubar o modulo.
        """
        # Falha em CPU, ou antes de resolver o device, nao tem para onde cair
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
    
    def _carregar_whisper(self, device: str) -> Any:
        """Carrega o pipeline Whisper no device indicado"""
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
        Obtem pipeline Whisper (carrega 1x, reutiliza depois)

        Returns:
            Pipeline do transformers para Whisper

        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar e nao houver fallback
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

        device = None
        try:
            device = self._obter_device(STT_WHISPER['device'], 'STT_WHISPER')
            self._whisper = self._carregar_whisper(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'Whisper', 'STT_WHISPER'):
                print(f"ERRO ao carregar Whisper: {e}")
                print("-"*70 + "\n")
                raise
            # Retry em CPU: se este tambem falhar, o erro propaga
            self._whisper = self._carregar_whisper("cpu")

        print("Whisper carregado com sucesso")
        print("-"*70 + "\n")

        return self._whisper
    
    # ==========================================================================
    # WAV2VEC - STT
    # ==========================================================================
    
    def _carregar_wav2vec(self, device: str) -> Any:
        """Carrega o pipeline wav2vec no device indicado"""
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
        Obtem pipeline wav2vec (carrega 1x, reutiliza depois)

        Returns:
            Pipeline do transformers para wav2vec

        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar e nao houver fallback
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

        device = None
        try:
            device = self._obter_device(STT_WAV2VEC2['device'], 'STT_WAV2VEC2')
            self._wav2vec = self._carregar_wav2vec(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'wav2vec', 'STT_WAV2VEC2'):
                print(f"ERRO ao carregar wav2vec: {e}")
                print("-"*70 + "\n")
                raise
            # Retry em CPU: se este tambem falhar, o erro propaga
            self._wav2vec = self._carregar_wav2vec("cpu")

        print("wav2vec carregado com sucesso")
        print("-"*70 + "\n")

        return self._wav2vec
    
    # ==========================================================================
    # PYANNOTE - OVERLAP DETECTION
    # ==========================================================================
    
    def _carregar_pyannote(self, device: str) -> Any:
        """Carrega o pipeline pyannote no device indicado"""
        from pyannote.audio import Pipeline

        # Modelo vem do config - nao ha mais constante hardcoded
        modelo_id = OVERLAP_DETECTOR['modelo']

        # Token HuggingFace — lido do .env
        hf_token = os.getenv('HF_TOKEN')

        print(f"Modelo: {modelo_id}")
        print(f"Device: {device}")

        pipeline = Pipeline.from_pretrained(
            modelo_id,
            token=hf_token
        )

        # Mover para device
        pipeline.to(torch.device(device))

        return pipeline

    def get_pyannote(self) -> Any:
        """
        Obtem pipeline pyannote (carrega 1x, reutiliza depois)

        Returns:
            Pipeline pyannote.audio

        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar e nao houver fallback
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

        device = None
        try:
            device = self._obter_device(OVERLAP_DETECTOR['device'], 'OVERLAP_DETECTOR')
            self._pyannote = self._carregar_pyannote(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'pyannote', 'OVERLAP_DETECTOR'):
                print(f"ERRO ao carregar pyannote: {e}")
                print("-"*70 + "\n")
                raise
            # Retry em CPU: se este tambem falhar, o erro propaga
            self._pyannote = self._carregar_pyannote("cpu")

        print("pyannote carregado com sucesso")
        print("-"*70 + "\n")

        return self._pyannote
    
    # ==========================================================================
    # SQUIM - MOS QUALITY ASSESSMENT
    # ==========================================================================
    
    def _carregar_squim(self, device: str) -> Any:
        """Carrega o modelo SQUIM no device indicado"""
        import torchaudio

        print(f"Modelo: SQUIM_OBJECTIVE (torchaudio)")
        print(f"Device: {device}")

        modelo = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        return modelo.to(device)

    def get_squim(self) -> Any:
        """
        Obtem modelo SQUIM (carrega 1x, reutiliza depois)

        Returns:
            Modelo SQUIM do torchaudio

        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar e nao houver fallback
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

        device = None
        try:
            device = self._obter_device(MOS_FILTER['device'], 'MOS_FILTER')
            self._squim = self._carregar_squim(device)
        except Exception as e:
            if not self._fallback_para_cpu(e, device, 'SQUIM', 'MOS_FILTER'):
                print(f"ERRO ao carregar SQUIM: {e}")
                print("-"*70 + "\n")
                raise
            # Retry em CPU: se este tambem falhar, o erro propaga
            self._squim = self._carregar_squim("cpu")

        print("SQUIM carregado com sucesso")
        print("-"*70 + "\n")

        return self._squim
    
    # ==========================================================================
    # DEEPFILTERNET3 - AUDIO DENOISING
    # ==========================================================================
    
    def _carregar_deepfilternet(self, device: str) -> Tuple[Any, Any, int]:
        """
        Carrega o DeepFilterNet3 e submete a biblioteca ao device do config.

        Achado A29: o init_df() nao aceita parametro device, e o enhance()
        resolve o dispositivo sozinho, por get_device(), que escolhe cuda:0
        sempre que houver CUDA. Sem o ajuste abaixo, num servidor com GPU e
        device='cpu' o modelo fica em CPU e as features vao para cuda:0.

        O ajuste tem de vir DEPOIS do init_df(): o proprio init_df() recarrega
        o parser da config interna do DeepFilterNet e apagaria o valor.
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
            log_level="ERROR"  # Reduz verbosidade
        )

        # A29: submeter o get_device() da biblioteca ao device do config
        df_config.set("DEVICE", device, str, section="train")

        # Mover modelo para device
        modelo = modelo.to(device)

        # Conferencia por log: o dispositivo interno tem de bater com o pedido.
        # Divergencia aqui e o achado A29 se manifestando - nao pode passar
        # em silencio na primeira rodada com GPU.
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
        Obtem modelo e estado DeepFilterNet (carrega 1x, reutiliza depois)

        Returns:
            Tupla (modelo, df_state, sample_rate)

        Raises:
            RuntimeError: Se modulo desabilitado no MASTER
            Exception: Se carregamento falhar e nao houver fallback

        LIMITACAO CONHECIDA (decisao registrada, instrucao 20): o retry em
        CPU chama init_df() de novo, e o init_df() volta a escolher o
        dispositivo por conta propria. Numa GPU que falhe, o fallback deste
        modelo pode nao se completar - ao contrario dos outros quatro.
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
            # Retry em CPU: se este tambem falhar, o erro propaga
            self._deepfilternet = self._carregar_deepfilternet("cpu")

        sr = self._deepfilternet[2]
        print(f"DeepFilterNet3 carregado com sucesso (SR={sr} Hz)")
        print("-"*70 + "\n")

        return self._deepfilternet
    
    # ==========================================================================
    # UTILIDADES - GESTAO DE MEMORIA
    # ==========================================================================
    
    def clear_cache(self):
        """Limpa cache de GPU (util para liberar VRAM)"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cache GPU limpo")
    
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