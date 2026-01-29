#!/usr/bin/env python3
"""
Modulo m08_whisper.py
Transcreve segmentos de audio usando Whisper (distil-whisper-large-v3-ptbr)
Adiciona campo 'stt_whisper' aos metadados JSON
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import librosa

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import STT_WHISPER, PROJECT_ROOT


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# ID do video a processar
id_video = '0aICqierMVA'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"
PASTA_AUDIOS = PROJECT_ROOT / "arquivos" / "temp" / id_video / "03-segments_16khz"

# Caminhos de saida
PASTA_OUTPUT_STT = PROJECT_ROOT / "arquivos" / "temp" / id_video / "06-stt_whisper"
PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO  # Sobrescreve na mesma pasta

# Extensoes de audio suportadas
EXTENSOES_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Modelo Whisper
MODELO_WHISPER = "freds0/distil-whisper-large-v3-ptbr"


# ==============================================================================
# FUNCOES DE MODELO E DEVICE
# ==============================================================================

def detectar_device_e_dtype() -> Tuple[str, torch.dtype]:
    """
    Detecta o device disponivel e define o dtype apropriado
    
    Returns:
        Tupla (device: str, torch_dtype: torch.dtype)
    """
    device_config = STT_WHISPER['device']
    
    if device_config == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            torch_dtype = torch.float16
            print("Device detectado: CUDA (GPU)")
        else:
            device = 'cpu'
            torch_dtype = torch.float32
            print("Device detectado: CPU")
    elif device_config == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU solicitada mas CUDA nao disponivel")
        device = 'cuda'
        torch_dtype = torch.float16
        print("Device configurado: CUDA (GPU)")
    elif device_config == 'cpu':
        device = 'cpu'
        torch_dtype = torch.float32
        print("Device configurado: CPU")
    else:
        raise ValueError(f"Device invalido no config: {device_config}")
    
    return device, torch_dtype


def calcular_batch_size_auto(device: str) -> int:
    """
    Calcula o batch_size automatico baseado em VRAM disponivel
    
    Args:
        device: 'cuda' ou 'cpu'
        
    Returns:
        Batch size otimizado
    """
    if device == 'cpu':
        return 1
    
    try:
        # Obter VRAM total disponivel
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Calculo conservador baseado em testes empiricos
        # distil-whisper-large-v3: ~2.5GB base + ~0.4GB por audio adicional
        if vram_total_gb >= 20:
            return 16
        elif vram_total_gb >= 12:
            return 8
        elif vram_total_gb >= 8:
            return 4
        elif vram_total_gb >= 4:
            return 2
        else:
            return 1
    except:
        # Fallback seguro
        return 4


def obter_batch_size(device: str) -> int:
    """
    Obtem o batch_size final considerando config e device
    
    Args:
        device: 'cuda' ou 'cpu'
        
    Returns:
        Batch size a ser usado
    """
    # CPU sempre usa batch_size=1
    if device == 'cpu':
        batch_config = STT_WHISPER['batch']['batch_size']
        if batch_config != 1:
            print(f"CPU detectada: batch_size ajustado de {batch_config} para 1 automaticamente")
        return 1
    
    # GPU: usar config ou calcular automatico
    batch_config = STT_WHISPER['batch']['batch_size']
    
    if batch_config == 'auto':
        batch_size = calcular_batch_size_auto(device)
        print(f"Batch size automatico calculado: {batch_size}")
        return batch_size
    else:
        print(f"Batch size configurado: {batch_config}")
        return int(batch_config)


def carregar_modelo_whisper(device: str, torch_dtype: torch.dtype) -> pipeline:
    """
    Carrega o modelo Whisper e cria pipeline de transcricao
    
    Args:
        device: 'cuda' ou 'cpu'
        torch_dtype: torch.float16 ou torch.float32
        
    Returns:
        Pipeline de transcricao configurado
    """
    print(f"\nCarregando modelo Whisper: {MODELO_WHISPER}")
    print("Aguarde, isso pode levar alguns minutos na primeira execucao...")
    
    # Carregar modelo
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODELO_WHISPER,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    model.to(device)
    
    # Carregar processor
    processor = AutoProcessor.from_pretrained(MODELO_WHISPER)
    
    # Criar pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    print("Modelo carregado com sucesso!")
    return pipe


# ==============================================================================
# FUNCOES DE JSON
# ==============================================================================

def carregar_json(caminho: Path) -> Optional[Dict]:
    """
    Carrega arquivo JSON
    
    Args:
        caminho: Path do arquivo JSON
        
    Returns:
        Dicionario com dados ou None se erro
    """
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"ERRO ao carregar {caminho.name}: {e}")
        return None


def salvar_json(dados: Dict, caminho: Path) -> bool:
    """
    Salva dados em arquivo JSON
    
    Args:
        dados: Dicionario a salvar
        caminho: Path do arquivo destino
        
    Returns:
        True se sucesso, False se erro
    """
    try:
        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"ERRO ao salvar {caminho.name}: {e}")
        return False


def carregar_metadados() -> Tuple[Optional[Dict], Optional[Dict]]:
    """
    Carrega os arquivos JSON de metadados
    
    Returns:
        Tupla (json_filtrado, json_acompanhamento)
        json_filtrado pode ser None se nao existir
        json_acompanhamento deve existir (obrigatorio)
    """
    # Arquivo de acompanhamento (obrigatorio)
    arquivo_acompanhamento = PASTA_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    json_acompanhamento = carregar_json(arquivo_acompanhamento)
    
    if json_acompanhamento is None:
        print(f"ERRO CRITICO: Arquivo obrigatorio nao encontrado: {arquivo_acompanhamento.name}")
        return None, None
    
    # Arquivo filtrado (opcional)
    arquivo_filtrado = PASTA_JSON_DINAMICO / f"{id_video}.json"
    json_filtrado = carregar_json(arquivo_filtrado)
    
    if json_filtrado is None:
        print(f"Arquivo de filtro nao encontrado: {arquivo_filtrado.name}")
        print("Processando TODOS os segmentos do arquivo de acompanhamento")
    else:
        print(f"Arquivo de filtro encontrado: {arquivo_filtrado.name}")
        print(f"Processando APENAS segmentos filtrados ({len(json_filtrado)} segmentos)")
    
    return json_filtrado, json_acompanhamento


def determinar_segmentos_elegiveis(json_filtrado: Optional[Dict], 
                                   json_acompanhamento: Dict) -> List[str]:
    """
    Determina quais segmentos devem ser processados
    
    Args:
        json_filtrado: JSON com segmentos filtrados (ou None)
        json_acompanhamento: JSON com todos os segmentos
        
    Returns:
        Lista de nomes de arquivos elegiveis para processamento
    """
    if json_filtrado is not None:
        # Usar apenas segmentos do filtrado
        return list(json_filtrado.keys())
    else:
        # Usar todos os segmentos do acompanhamento
        return list(json_acompanhamento.keys())


# ==============================================================================
# FUNCOES DE AUDIO
# ==============================================================================

def listar_arquivos_audio_elegiveis(pasta: Path, 
                                    segmentos_elegiveis: List[str]) -> List[Path]:
    """
    Lista arquivos de audio que estao na lista de elegiveis
    
    Args:
        pasta: Path da pasta com arquivos de audio
        segmentos_elegiveis: Lista de nomes de arquivos elegiveis
        
    Returns:
        Lista de Path dos arquivos encontrados
    """
    arquivos = []
    segmentos_set = set(segmentos_elegiveis)
    
    for arquivo in pasta.iterdir():
        if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES_AUDIO:
            if arquivo.name in segmentos_set:
                arquivos.append(arquivo)
    
    return sorted(arquivos)


def transcrever_batch(pipe: pipeline, 
                     arquivos_audio: List[Path],
                     batch_size: int) -> Dict[str, str]:
    """
    Transcreve um batch de arquivos de audio
    
    Args:
        pipe: Pipeline do Whisper configurado
        arquivos_audio: Lista de paths dos arquivos
        batch_size: Tamanho do batch
        
    Returns:
        Dicionario {nome_arquivo: transcricao}
    """
    resultados = {}
    total = len(arquivos_audio)
    
    # Processar em batches
    for i in range(0, total, batch_size):
        batch = arquivos_audio[i:i + batch_size]
        batch_atual = min(i + batch_size, total)
        
        print(f"Processando batch [{i+1}-{batch_atual}/{total}]...")
        
        # Carregar audios do batch
        audios = []
        nomes = []
        for arquivo in batch:
            try:
                # Carregar audio em 16kHz (sample rate do Whisper)
                audio, _ = librosa.load(str(arquivo), sr=16000, mono=True)
                audios.append(audio)
                nomes.append(arquivo.name)
            except Exception as e:
                print(f"  ERRO ao carregar {arquivo.name}: {e}")
                resultados[arquivo.name] = None
        
        # Transcrever batch
        if audios:
            try:
                # Pipeline aceita lista de arrays
                outputs = pipe(audios, generate_kwargs={"language": "pt", "task": "transcribe"})
                
                # Extrair transcricoes
                for nome, output in zip(nomes, outputs):
                    transcricao = output['text'].strip()
                    resultados[nome] = transcricao
                    print(f"  {nome}: OK")
                    
            except Exception as e:
                print(f"  ERRO no batch: {e}")
                for nome in nomes:
                    if nome not in resultados:
                        resultados[nome] = None
    
    return resultados


def processar_transcricoes(pipe: pipeline,
                          arquivos_audio: List[Path],
                          batch_size: int) -> Dict[str, str]:
    """
    Processa todas as transcricoes com controle de progresso
    
    Args:
        pipe: Pipeline do Whisper
        arquivos_audio: Lista de arquivos para transcrever
        batch_size: Tamanho do batch
        
    Returns:
        Dicionario {nome_arquivo: transcricao}
    """
    print(f"\nIniciando transcricao de {len(arquivos_audio)} arquivos...")
    print(f"Batch size: {batch_size}")
    print("-" * 70)
    
    resultados = transcrever_batch(pipe, arquivos_audio, batch_size)
    
    # Estatisticas
    total = len(resultados)
    sucesso = sum(1 for v in resultados.values() if v is not None)
    falhas = total - sucesso
    
    print("-" * 70)
    print(f"Transcricao concluida: {sucesso}/{total} sucesso, {falhas} falhas")
    
    return resultados


# ==============================================================================
# FUNCOES DE ATUALIZACAO E SALVAMENTO
# ==============================================================================

def atualizar_json_com_transcricoes(json_dados: Dict,
                                    transcricoes: Dict[str, str]) -> Dict:
    """
    Adiciona campo stt_whisper aos metadados
    
    Args:
        json_dados: Dicionario original de metadados
        transcricoes: Dicionario {nome_arquivo: transcricao}
        
    Returns:
        Dicionario atualizado
    """
    json_atualizado = json_dados.copy()
    
    for nome_arquivo, transcricao in transcricoes.items():
        if nome_arquivo in json_atualizado:
            json_atualizado[nome_arquivo]['stt_whisper'] = transcricao
    
    return json_atualizado


def adicionar_transcricoes_null(json_dados: Dict,
                                segmentos_processados: List[str]) -> Dict:
    """
    Adiciona stt_whisper=null para segmentos nao processados
    
    Args:
        json_dados: Dicionario de metadados
        segmentos_processados: Lista de segmentos que foram processados
        
    Returns:
        Dicionario atualizado
    """
    json_atualizado = json_dados.copy()
    processados_set = set(segmentos_processados)
    
    for nome_arquivo in json_atualizado.keys():
        if nome_arquivo not in processados_set:
            if 'stt_whisper' not in json_atualizado[nome_arquivo]:
                json_atualizado[nome_arquivo]['stt_whisper'] = None
    
    return json_atualizado


def salvar_outputs(json_filtrado: Optional[Dict],
                  json_acompanhamento: Dict,
                  segmentos_elegiveis: List[str],
                  transcricoes: Dict[str, str]) -> bool:
    """
    Salva os JSONs atualizados nas pastas de output
    
    Args:
        json_filtrado: JSON filtrado original (ou None)
        json_acompanhamento: JSON acompanhamento original
        segmentos_elegiveis: Lista de segmentos que eram elegiveis
        transcricoes: Dicionario com transcricoes
        
    Returns:
        True se sucesso, False se erro
    """
    # Criar pasta output se nao existir
    PASTA_OUTPUT_STT.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("SALVANDO OUTPUTS")
    print("=" * 70)
    
    # 1. Atualizar JSON acompanhamento (todos os segmentos)
    json_acomp_atualizado = atualizar_json_com_transcricoes(
        json_acompanhamento, 
        transcricoes
    )
    # Adicionar null para nao processados
    json_acomp_atualizado = adicionar_transcricoes_null(
        json_acomp_atualizado,
        segmentos_elegiveis
    )
    
    # Salvar em 06-stt_whisper
    arquivo_acomp_output = PASTA_OUTPUT_STT / f"{id_video}_segments_acompanhamento.json"
    if not salvar_json(json_acomp_atualizado, arquivo_acomp_output):
        return False
    print(f"Salvo: {arquivo_acomp_output}")
    
    # Copiar para 00-json_dinamico (sobrescrever)
    arquivo_acomp_dinamico = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    if not salvar_json(json_acomp_atualizado, arquivo_acomp_dinamico):
        return False
    print(f"Sobrescrito: {arquivo_acomp_dinamico}")
    
    # 2. Se existir JSON filtrado, atualizar e salvar
    if json_filtrado is not None:
        json_filtrado_atualizado = atualizar_json_com_transcricoes(
            json_filtrado,
            transcricoes
        )
        
        # Salvar em 06-stt_whisper
        arquivo_filtrado_output = PASTA_OUTPUT_STT / f"{id_video}_whisper.json"
        if not salvar_json(json_filtrado_atualizado, arquivo_filtrado_output):
            return False
        print(f"Salvo: {arquivo_filtrado_output}")
        
        # Copiar para 00-json_dinamico como {id}.json (sobrescrever)
        arquivo_filtrado_dinamico = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}.json"
        if not salvar_json(json_filtrado_atualizado, arquivo_filtrado_dinamico):
            return False
        print(f"Sobrescrito: {arquivo_filtrado_dinamico}")
    
    print("=" * 70)
    return True


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def main():
    """
    Funcao principal: orquestra todo o fluxo de transcricao
    """
    print("=" * 70)
    print("MODULO 08: TRANSCRICAO WHISPER")
    print("=" * 70)
    print(f"Video ID: {id_video}")
    print(f"Modelo: {MODELO_WHISPER}")
    
    # 1. Detectar device e configurar
    device, torch_dtype = detectar_device_e_dtype()
    batch_size = obter_batch_size(device)
    
    # 2. Carregar modelo
    pipe = carregar_modelo_whisper(device, torch_dtype)
    
    # 3. Carregar metadados
    print("\n" + "=" * 70)
    print("CARREGANDO METADADOS")
    print("=" * 70)
    json_filtrado, json_acompanhamento = carregar_metadados()
    
    if json_acompanhamento is None:
        print("ERRO: Nao foi possivel carregar metadados. Abortando.")
        return
    
    # 4. Determinar segmentos elegiveis
    segmentos_elegiveis = determinar_segmentos_elegiveis(
        json_filtrado, 
        json_acompanhamento
    )
    print(f"\nSegmentos elegiveis para processamento: {len(segmentos_elegiveis)}")
    
    # 5. Listar arquivos de audio elegiveis
    print("\n" + "=" * 70)
    print("LISTANDO ARQUIVOS DE AUDIO")
    print("=" * 70)
    arquivos_audio = listar_arquivos_audio_elegiveis(
        PASTA_AUDIOS,
        segmentos_elegiveis
    )
    
    if not arquivos_audio:
        print("AVISO: Nenhum arquivo de audio elegivel encontrado")
        print("Verifique se os arquivos existem em:", PASTA_AUDIOS)
        return
    
    print(f"Arquivos encontrados: {len(arquivos_audio)}/{len(segmentos_elegiveis)}")
    
    # 6. Processar transcricoes
    print("\n" + "=" * 70)
    print("PROCESSANDO TRANSCRICOES")
    print("=" * 70)
    transcricoes = processar_transcricoes(pipe, arquivos_audio, batch_size)
    
    # 7. Salvar outputs
    sucesso = salvar_outputs(
        json_filtrado,
        json_acompanhamento,
        segmentos_elegiveis,
        transcricoes
    )
    
    # 8. Relatorio final
    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 70)
    if sucesso:
        print("Status: SUCESSO")
        print(f"Transcritos: {len([t for t in transcricoes.values() if t is not None])}")
        print(f"Outputs salvos em: {PASTA_OUTPUT_STT}")
        print(f"JSONs atualizados em: {PASTA_OUTPUT_JSON_DINAMICO}")
    else:
        print("Status: ERRO ao salvar outputs")
    print("=" * 70)


# ==============================================================================
# EXECUCAO
# ==============================================================================

if __name__ == "__main__":
    main()