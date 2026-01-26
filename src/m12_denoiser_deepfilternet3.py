#!/usr/bin/env python3
"""
Modulo m12_denoiser_deepfilternet3.py
Aplica denoising em segmentos de audio usando DeepFilterNet3
Filtra por qualidade MOS e adiciona campo 'utilizou_denoiser' aos metadados JSON
"""

import sys
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import torch
import librosa
import soundfile as sf
import numpy as np
from df import enhance, init_df
from df.enhance import enhance, init_df, save_audio

warnings.filterwarnings("ignore")

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEEPFILTERNET_DENOISER, PROJECT_ROOT


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# ID do video a processar
id_video = 'QN7gUP7nYhQ'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"
PASTA_AUDIOS_ORIGINAIS = PROJECT_ROOT / "arquivos" / "temp" / id_video / "02-segmentos_originais"

# Caminhos de saida
PASTA_OUTPUT_DENOISER = PROJECT_ROOT / "arquivos" / "temp" / id_video / "10-denoiser"
PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO  # Sobrescreve na mesma pasta

# Extensoes de audio suportadas
EXTENSOES_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}

# Configuracoes do DeepFilterNet3
MOS_QUALITY_FILTER = DEEPFILTERNET_DENOISER["mos_quality_filter"]
DEVICE = DEEPFILTERNET_DENOISER["device"]
POST_FILTER = DEEPFILTERNET_DENOISER["post_filter"]
ATTENUATION_LIMIT = DEEPFILTERNET_DENOISER["attenuation_limit"]
SKIP_IF_ALREADY_PROCESSED = DEEPFILTERNET_DENOISER["skip_if_already_processed"]


# ==============================================================================
# FUNCOES DE DEVICE E MODELO
# ==============================================================================

def detectar_device(device_config: str) -> torch.device:
    """
    Detecta o dispositivo de processamento baseado na configuracao
    
    Args:
        device_config: String com configuracao ('auto', 'gpu', 'cpu')
    
    Returns:
        torch.device: Dispositivo selecionado
    """
    if device_config == "cpu":
        print("[INFO] Dispositivo forçado: CPU")
        return torch.device("cpu")
    
    elif device_config == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] Dispositivo forçado: GPU ({torch.cuda.get_device_name(0)})")
            return device
        else:
            raise RuntimeError("GPU forçada mas CUDA não disponível")
    
    elif device_config == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[INFO] Dispositivo detectado: GPU ({torch.cuda.get_device_name(0)})")
            return device
        else:
            device = torch.device("cpu")
            print("[INFO] Dispositivo detectado: CPU (GPU não disponível)")
            return device
    
    else:
        raise ValueError(f"device_config inválido: {device_config}. Use 'auto', 'gpu' ou 'cpu'")


def inicializar_deepfilternet(device: torch.device, post_filter: int, attenuation_limit: float):
    """
    Inicializa o modelo DeepFilterNet3
    
    Args:
        device: Dispositivo torch
        post_filter: Nivel de agressividade do filtro (0, 1, 2)
        attenuation_limit: Limite de atenuacao (0.0-1.0)
    
    Returns:
        Tupla (model, df_state, sample_rate)
    """
    print("[INFO] Inicializando DeepFilterNet3...")
    
    # Inicializa modelo e estado
    model, df_state, sr = init_df(
        post_filter=post_filter,
        log_level="ERROR"  # Reduz verbosidade
    )
    
    # Move modelo para dispositivo
    model = model.to(device)
    
    print(f"[INFO] Modelo carregado: SR={sr}Hz, post_filter={post_filter}, attenuation_limit={attenuation_limit}")
    
    return model, df_state, sr


# ==============================================================================
# FUNCOES DE PROCESSAMENTO DE AUDIO
# ==============================================================================

def processar_audio_denoiser(
    audio_path: Path,
    model,
    df_state,
    device: torch.device,
    attenuation_limit: float
) -> Tuple[np.ndarray, int]:
    """
    Processa um arquivo de audio com DeepFilterNet3
    
    Args:
        audio_path: Caminho do arquivo de audio
        model: Modelo DeepFilterNet
        df_state: Estado do DeepFilterNet
        device: Dispositivo torch
        attenuation_limit: Limite de atenuacao
    
    Returns:
        Tupla (audio_denoised, sample_rate)
    """
    # Carrega audio (DeepFilterNet espera mono em 48kHz)
    audio, sr_original = librosa.load(str(audio_path), sr=48000, mono=True)
    
    # Converte para tensor torch (mantém em CPU - DeepFilterNet requer isso internamente)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # Shape: (1, samples)
    
    # Aplica denoising
    with torch.no_grad():
        audio_denoised = enhance(
            model,
            df_state,
            audio_tensor,
            atten_lim_db=attenuation_limit
        )
    
    # Converte de volta para numpy (resultado pode estar em CPU ou GPU)
    if audio_denoised.is_cuda:
        audio_denoised_np = audio_denoised.cpu().numpy()
    else:
        audio_denoised_np = audio_denoised.numpy()
    
    # Remove dimensão batch
    audio_denoised_np = audio_denoised_np.squeeze(0)
    
    return audio_denoised_np, 48000  # DeepFilterNet sempre retorna 48kHz


def salvar_audio_formato_original(
    audio_denoised: np.ndarray,
    sr: int,
    output_path: Path,
    formato_original: str
) -> None:
    """
    Salva audio processado no mesmo formato do arquivo original
    
    Args:
        audio_denoised: Array numpy com audio processado
        sr: Sample rate
        output_path: Caminho de saida (com extensao original)
        formato_original: Extensao do arquivo original (ex: '.flac', '.mp3')
    """
    # Normaliza audio para evitar clipping
    audio_normalized = np.clip(audio_denoised, -1.0, 1.0)
    
    if formato_original in ['.wav', '.flac']:
        # Formatos lossless: usa soundfile diretamente
        sf.write(str(output_path), audio_normalized, sr, subtype='PCM_16')
    
    elif formato_original in ['.mp3', '.ogg', '.m4a', '.aac']:
        # Formatos comprimidos: usa pydub via arquivo temporario WAV
        temp_wav = output_path.with_suffix('.wav')
        
        # Salva temporariamente como WAV
        sf.write(str(temp_wav), audio_normalized, sr, subtype='PCM_16')
        
        # Converte para formato desejado usando pydub
        from pydub import AudioSegment
        audio_segment = AudioSegment.from_wav(str(temp_wav))
        
        # Define parametros de exportacao por formato
        if formato_original == '.mp3':
            audio_segment.export(str(output_path), format='mp3', bitrate='192k')
        elif formato_original == '.ogg':
            audio_segment.export(str(output_path), format='ogg', codec='libvorbis')
        elif formato_original == '.m4a':
            audio_segment.export(str(output_path), format='mp4', codec='aac')
        elif formato_original == '.aac':
            audio_segment.export(str(output_path), format='adts', codec='aac')
        
        # Remove arquivo temporario
        temp_wav.unlink()
    
    else:
        raise ValueError(f"Formato não suportado: {formato_original}")


# ==============================================================================
# FUNCOES DE MANIPULACAO DE JSON
# ==============================================================================

def carregar_json_dinamico(pasta_json: Path, id_video: str) -> Tuple[Optional[Dict], Dict]:
    """
    Carrega os arquivos JSON dinamicos
    
    Args:
        pasta_json: Pasta contendo os JSON
        id_video: ID do video
    
    Returns:
        Tupla (json_filtrado, json_acompanhamento)
        json_filtrado pode ser None se nao existir
    """
    # Arquivo de filtro (opcional)
    path_filtrado = pasta_json / f"{id_video}.json"
    json_filtrado = None
    
    if path_filtrado.exists():
        with open(path_filtrado, 'r', encoding='utf-8') as f:
            json_filtrado = json.load(f)
        print(f"[INFO] Carregado JSON filtrado: {len(json_filtrado)} segmentos")
    else:
        print("[INFO] JSON filtrado não encontrado - processará todos os segmentos")
    
    # Arquivo de acompanhamento (obrigatorio)
    path_acompanhamento = pasta_json / f"{id_video}_segments_acompanhamento.json"
    
    if not path_acompanhamento.exists():
        raise FileNotFoundError(f"JSON obrigatório não encontrado: {path_acompanhamento}")
    
    with open(path_acompanhamento, 'r', encoding='utf-8') as f:
        json_acompanhamento = json.load(f)
    
    print(f"[INFO] Carregado JSON acompanhamento: {len(json_acompanhamento)} segmentos totais")
    
    return json_filtrado, json_acompanhamento


def determinar_segmentos_processar(
    json_filtrado: Optional[Dict],
    json_acompanhamento: Dict,
    mos_quality_filter: List[str],
    skip_if_processed: bool
) -> Tuple[List[str], Dict[str, bool]]:
    """
    Determina quais segmentos devem ser processados baseado nos filtros
    
    Args:
        json_filtrado: JSON com segmentos filtrados (pode ser None)
        json_acompanhamento: JSON com todos os segmentos
        mos_quality_filter: Lista de qualidades MOS para processar
        skip_if_processed: Se True, pula segmentos ja processados
    
    Returns:
        Tupla (lista_segmentos_elegíveis, dict_status_todos_segmentos)
    """
    segmentos_elegiveis = []
    status_segmentos = {}
    
    # Define base de segmentos a considerar
    base_segmentos = json_filtrado if json_filtrado is not None else json_acompanhamento
    
    # Itera sobre todos os segmentos do acompanhamento
    for nome_arquivo, metadata in json_acompanhamento.items():
        
        # Verifica se segmento esta na base de processamento
        if nome_arquivo not in base_segmentos:
            status_segmentos[nome_arquivo] = None  # Não estava no filtro
            continue
        
        # Verifica se ja foi processado (skip)
        if skip_if_processed and metadata.get("utilizou_denoiser") is True:
            print(f"[SKIP] {nome_arquivo} - Já processado anteriormente")
            status_segmentos[nome_arquivo] = True  # Mantém status anterior
            continue
        
        # Verifica filtro MOS
        mos_qualidade = metadata.get("mos_qualidade")
        
        if mos_qualidade in mos_quality_filter:
            segmentos_elegiveis.append(nome_arquivo)
            status_segmentos[nome_arquivo] = True  # Será processado
        else:
            status_segmentos[nome_arquivo] = False  # Não passa no filtro MOS
    
    return segmentos_elegiveis, status_segmentos


def salvar_json_atualizado(
    json_data: Dict,
    status_segmentos: Dict[str, bool],
    output_path: Path
) -> None:
    """
    Salva JSON com campo 'utilizou_denoiser' atualizado
    
    Args:
        json_data: Dicionario com metadados
        status_segmentos: Dict com status de processamento por segmento
        output_path: Caminho de saida do JSON
    """
    # Cria copia do JSON original
    json_atualizado = json_data.copy()
    
    # Atualiza campo para cada segmento
    for nome_arquivo in json_atualizado.keys():
        if nome_arquivo in status_segmentos:
            json_atualizado[nome_arquivo]["utilizou_denoiser"] = status_segmentos[nome_arquivo]
        else:
            # Segmento não estava nos processados
            json_atualizado[nome_arquivo]["utilizou_denoiser"] = None
    
    # Salva JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_atualizado, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] JSON salvo: {output_path}")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def main():
    """
    Funcao principal de execucao
    """
    print("=" * 70)
    print("MÓDULO 12: DENOISER DEEPFILTERNET3")
    print("=" * 70)
    print(f"ID do vídeo: {id_video}")
    print(f"Filtro MOS: {MOS_QUALITY_FILTER}")
    print(f"Device: {DEVICE}")
    print(f"Post-filter: {POST_FILTER}")
    print(f"Attenuation limit: {ATTENUATION_LIMIT}")
    print(f"Skip já processados: {SKIP_IF_ALREADY_PROCESSED}")
    print("=" * 70)
    print()
    
    # Valida existencia de pastas de input
    if not PASTA_JSON_DINAMICO.exists():
        raise FileNotFoundError(f"Pasta JSON não encontrada: {PASTA_JSON_DINAMICO}")
    
    if not PASTA_AUDIOS_ORIGINAIS.exists():
        raise FileNotFoundError(f"Pasta de áudios não encontrada: {PASTA_AUDIOS_ORIGINAIS}")
    
    # Cria pastas de output
    PASTA_OUTPUT_DENOISER.mkdir(parents=True, exist_ok=True)
    
    # PASSO 1: Carregar JSON dinamico
    print("[PASSO 1/6] Carregando JSON dinâmico...")
    json_filtrado, json_acompanhamento = carregar_json_dinamico(PASTA_JSON_DINAMICO, id_video)
    print()
    
    # PASSO 2: Determinar segmentos a processar
    print("[PASSO 2/6] Determinando segmentos elegíveis...")
    
    if not MOS_QUALITY_FILTER:
        print("[WARNING] mos_quality_filter está vazio - nenhum segmento será processado")
        segmentos_elegiveis = []
        status_segmentos = {nome: False for nome in json_acompanhamento.keys()}
    else:
        segmentos_elegiveis, status_segmentos = determinar_segmentos_processar(
            json_filtrado,
            json_acompanhamento,
            MOS_QUALITY_FILTER,
            SKIP_IF_ALREADY_PROCESSED
        )
    
    print(f"[INFO] Segmentos elegíveis para processamento: {len(segmentos_elegiveis)}")
    
    # Estatisticas por qualidade MOS
    mos_stats = {}
    for nome in segmentos_elegiveis:
        mos = json_acompanhamento[nome].get("mos_qualidade", "desconhecido")
        mos_stats[mos] = mos_stats.get(mos, 0) + 1
    
    for mos, count in sorted(mos_stats.items()):
        print(f"  - {mos}: {count} segmentos")
    print()
    
    # PASSO 3: Inicializar modelo DeepFilterNet
    if segmentos_elegiveis:
        print("[PASSO 3/6] Inicializando DeepFilterNet3...")
        device = detectar_device(DEVICE)
        model, df_state, sr_modelo = inicializar_deepfilternet(device, POST_FILTER, ATTENUATION_LIMIT)
        print()
    else:
        print("[PASSO 3/6] Pulando inicialização - nenhum segmento para processar")
        print()
    
    # PASSO 4: Processar audios
    print("[PASSO 4/6] Processando áudios...")
    
    tempo_inicio = time.time()
    processados = 0
    erros = 0
    
    for idx, nome_arquivo in enumerate(segmentos_elegiveis, 1):
        try:
            # Encontra arquivo de audio na pasta original
            audio_path = PASTA_AUDIOS_ORIGINAIS / nome_arquivo
            
            if not audio_path.exists():
                print(f"[ERRO] Arquivo não encontrado: {nome_arquivo}")
                erros += 1
                status_segmentos[nome_arquivo] = False
                continue
            
            # Determina formato original
            formato_original = audio_path.suffix.lower()
            
            # Processa audio
            print(f"[{idx}/{len(segmentos_elegiveis)}] Processando: {nome_arquivo} ({json_acompanhamento[nome_arquivo].get('mos_qualidade', '?')})")
            
            audio_denoised, sr = processar_audio_denoiser(
                audio_path,
                model,
                df_state,
                device,
                ATTENUATION_LIMIT
            )
            
            # Salva audio processado
            output_audio_path = PASTA_OUTPUT_DENOISER / nome_arquivo
            salvar_audio_formato_original(audio_denoised, sr, output_audio_path, formato_original)
            
            processados += 1
            
        except Exception as e:
            print(f"[ERRO] Falha ao processar {nome_arquivo}: {str(e)}")
            erros += 1
            status_segmentos[nome_arquivo] = False
    
    tempo_total = time.time() - tempo_inicio
    
    print()
    print(f"[INFO] Processamento concluído: {processados} sucessos, {erros} erros")
    
    if processados > 0:
        print(f"[INFO] Tempo total: {tempo_total/60:.1f} minutos ({tempo_total/processados:.2f}s por áudio)")
    else:
        print(f"[INFO] Tempo total: {tempo_total/60:.1f} minutos (nenhum áudio processado com sucesso)")
    
    print()
    
    # PASSO 5: Salvar JSON atualizados na pasta 10-denoiser
    print("[PASSO 5/6] Salvando JSON atualizados (pasta 10-denoiser)...")
    
    # Salva JSON de acompanhamento atualizado
    path_acompanhamento_output = PASTA_OUTPUT_DENOISER / f"{id_video}_segments_acompanhamento.json"
    salvar_json_atualizado(json_acompanhamento, status_segmentos, path_acompanhamento_output)
    
    # Salva JSON filtrado atualizado (se existir)
    if json_filtrado is not None:
        path_filtrado_output = PASTA_OUTPUT_DENOISER / f"{id_video}_denoiser.json"
        
        # Filtra apenas segmentos que estavam no filtro original
        status_filtrados = {k: v for k, v in status_segmentos.items() if k in json_filtrado}
        salvar_json_atualizado(json_filtrado, status_filtrados, path_filtrado_output)
    
    print()
    
    # PASSO 6: Sobrescrever JSON na pasta 00-json_dinamico
    print("[PASSO 6/6] Sobrescrevendo JSON na pasta 00-json_dinamico...")
    
    # Copia JSON de acompanhamento atualizado
    shutil.copy2(
        path_acompanhamento_output,
        PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    )
    print(f"[INFO] Copiado: {id_video}_segments_acompanhamento.json")
    
    # Copia JSON filtrado atualizado (se existir)
    if json_filtrado is not None:
        shutil.copy2(
            path_filtrado_output,
            PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}.json"
        )
        print(f"[INFO] Copiado: {id_video}.json")
    
    print()
    print("=" * 70)
    print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()