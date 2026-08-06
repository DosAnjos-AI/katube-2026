#!/usr/bin/env python3
"""
Module m12_denoiser_deepfilternet3.py
Applies denoising to audio segments using DeepFilterNet3
Filters by MOS quality and adds the 'utilizou_denoiser' field to the JSON metadata
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
from df.enhance import enhance

warnings.filterwarnings("ignore")

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEEPFILTERNET_DENOISER, PROJECT_ROOT
from m01_load_models import ModelManager


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# Configuracoes do DeepFilterNet3 consumidas POR ESTE MODULO.
# Os campos 'device' e 'post_filter' NAO entram aqui: quem os aplica e
# anuncia e o m01, ao carregar o modelo. Le-los aqui so para imprimir
# duplicava a leitura e dava a impressao falsa de que este modulo decide
# o dispositivo.
MOS_QUALITY_FILTER = DEEPFILTERNET_DENOISER["mos_quality_filter"]
ATTENUATION_LIMIT = DEEPFILTERNET_DENOISER["attenuation_limit"]
SKIP_IF_ALREADY_PROCESSED = DEEPFILTERNET_DENOISER["skip_if_already_processed"]


# ==============================================================================
# FUNCOES DE PROCESSAMENTO DE AUDIO
# ==============================================================================

def processar_audio_denoiser(
    audio_path: Path,
    model,
    df_state,
    attenuation_limit: float,
    sr_modelo: int
) -> Tuple[np.ndarray, int]:
    """
    Processes an audio file with DeepFilterNet3, returning it at the
    sample rate it came in at.

    DeepFilterNet3 operates internally at 48 kHz: a 24 kHz segment is
    resampled on input, and that is unavoidable. The output, however,
    is returned at the source rate. Without this the dataset would come
    out with a MIXED sample rate - 48 kHz in segments that went through
    the denoiser, the source rate in those that did not - and the
    48 kHz ones would be stretched 24 kHz material, with no information
    at all above 12 kHz. The resample cost is negligible.

    CHANNELS ARE NOT RESTORED, on purpose: the model is mono, and
    recreating a stereo pair by duplicating the channel would be
    inventing information the denoiser did not produce. m13 takes every
    path to mono anyway (SOX_NORMALIZER['channels']), so this does not
    mix anything into the dataset.

    The device is not a parameter of this function: it is set by m01,
    when loading the model, including writing the DEVICE key into
    DeepFilterNet's internal config (finding A29).

    Args:
        audio_path: Path of the audio file
        model: DeepFilterNet model
        df_state: DeepFilterNet state
        attenuation_limit: Attenuation limit, in decibels
        sr_modelo: Model's internal rate, as announced by m01 - there is
            no fixed number here, the loaded library is what decides

    Returns:
        Tuple (audio_denoised, sample_rate), the sample rate being that
        of the INPUT segment - not the model's internal rate.
    """
    # Taxa nativa do segmento, medida ANTES da reamostragem para o modelo
    sr_fonte = sf.info(str(audio_path)).samplerate

    # Carrega audio (DeepFilterNet espera mono na taxa interna do modelo)
    audio, _ = librosa.load(str(audio_path), sr=sr_modelo, mono=True)

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

    # Volta para a taxa da fonte (o modelo devolve sempre a taxa interna dele)
    if sr_fonte != sr_modelo:
        audio_denoised_np = librosa.resample(
            audio_denoised_np, orig_sr=sr_modelo, target_sr=sr_fonte
        )

    return audio_denoised_np, sr_fonte


def salvar_audio_formato_original(
    audio_denoised: np.ndarray,
    sr: int,
    output_path: Path,
    formato_original: str,
    subtype: str
) -> None:
    """
    Saves the processed audio in the same format as the original file

    Args:
        audio_denoised: Numpy array with the processed audio
        sr: Sample rate
        output_path: Output path (with the original extension)
        formato_original: Original file extension (e.g., '.flac', '.mp3')
        subtype: soundfile subtype read from the INPUT segment. Writing
            with a fixed constant would downgrade a 24-bit segment
            without warning - m04 already delivered it at the source's
            depth, and the denoiser cannot be the point that throws
            that away.
    """
    # Normaliza audio para evitar clipping
    audio_normalized = np.clip(audio_denoised, -1.0, 1.0)

    if formato_original in ['.wav', '.flac']:
        # Formatos lossless: usa soundfile diretamente
        sf.write(str(output_path), audio_normalized, sr, subtype=subtype)

    elif formato_original in ['.mp3', '.ogg', '.m4a', '.aac']:
        # Formatos comprimidos: usa pydub via arquivo temporario WAV
        temp_wav = output_path.with_suffix('.wav')

        # Salva temporariamente como WAV
        sf.write(str(temp_wav), audio_normalized, sr, subtype=subtype)

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

def carregar_json_dinamico(pasta_json: Path, audio_id: str) -> Tuple[Optional[Dict], Dict]:
    """
    Loads the dynamic JSON files.

    Args:
        pasta_json: Folder containing the JSONs
        audio_id: Audio ID

    Returns:
        Tuple (json_filtrado, json_acompanhamento)
        json_filtrado can be None if it does not exist
    """
    # Arquivo de filtro (opcional)
    path_filtrado = pasta_json / f"{audio_id}.json"
    json_filtrado = None

    if path_filtrado.exists():
        with open(path_filtrado, 'r', encoding='utf-8') as f:
            json_filtrado = json.load(f)
        print(f"[INFO] Carregado JSON filtrado: {len(json_filtrado)} segmentos")
    else:
        print("[INFO] JSON filtrado nao encontrado - processara todos os segmentos")

    # Arquivo de acompanhamento (obrigatorio)
    path_acompanhamento = pasta_json / f"{audio_id}_segments_acompanhamento.json"
    
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
    Determines which segments should be processed based on the filters

    Args:
        json_filtrado: JSON with filtered segments (can be None)
        json_acompanhamento: JSON with all segments
        mos_quality_filter: List of MOS qualities to process
        skip_if_processed: If True, skips segments already processed

    Returns:
        Tuple (eligible_segments_list, all_segments_status_dict)
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
    Saves the JSON with the 'utilizou_denoiser' field updated

    Args:
        json_data: Dictionary with metadata
        status_segmentos: Dict with the processing status per segment
        output_path: Output path of the JSON
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

def main(audio_id: str) -> bool:
    """
    Main execution function.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        True if the updated JSONs were written. A missing precondition
        propagates an exception (FileNotFoundError).
    """
    # Definir caminhos baseados no audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    PASTA_AUDIOS_ORIGINAIS = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "02-segmentos_originais"
    PASTA_OUTPUT_DENOISER = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "10-denoiser"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO

    print("=" * 70)
    print("MODULO 12: DENOISER DEEPFILTERNET3")
    print("=" * 70)
    print(f"ID do audio: {audio_id}")
    print(f"Filtro MOS: {MOS_QUALITY_FILTER}")
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
    json_filtrado, json_acompanhamento = carregar_json_dinamico(PASTA_JSON_DINAMICO, audio_id)
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
        
        # Usar ModelManager (singleton)
        manager = ModelManager()
        model, df_state, sr_modelo = manager.get_deepfilternet()
        
        # Device ja gerenciado pelo manager
        # Dispositivo real do modelo carregado pelo m01 - nao ha segunda
        # decisao de device aqui
        print(f"[INFO] Modelo carregado em {next(model.parameters()).device}")
        print(f"[INFO] SR={sr_modelo}Hz, attenuation_limit={ATTENUATION_LIMIT} dB")
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
            
            # Subtipo do segmento de ENTRADA: e ele que define a profundidade
            # de bits da saida, para o denoiser nao rebaixar o que o m04
            # entregou na qualidade da fonte
            subtype_entrada = sf.info(str(audio_path)).subtype

            audio_denoised, sr = processar_audio_denoiser(
                audio_path,
                model,
                df_state,
                ATTENUATION_LIMIT,
                sr_modelo
            )

            # Salva audio processado
            output_audio_path = PASTA_OUTPUT_DENOISER / nome_arquivo
            salvar_audio_formato_original(
                audio_denoised, sr, output_audio_path, formato_original, subtype_entrada
            )
            
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
    path_acompanhamento_output = PASTA_OUTPUT_DENOISER / f"{audio_id}_segments_acompanhamento.json"
    salvar_json_atualizado(json_acompanhamento, status_segmentos, path_acompanhamento_output)

    # Salva JSON filtrado atualizado (se existir)
    if json_filtrado is not None:
        path_filtrado_output = PASTA_OUTPUT_DENOISER / f"{audio_id}_denoiser.json"
        
        # Filtra apenas segmentos que estavam no filtro original
        status_filtrados = {k: v for k, v in status_segmentos.items() if k in json_filtrado}
        salvar_json_atualizado(json_filtrado, status_filtrados, path_filtrado_output)
    
    print()
    
    # PASSO 6: Sobrescrever JSON na pasta 00-json_dinamico
    print("[PASSO 6/6] Sobrescrevendo JSON na pasta 00-json_dinamico...")
    
    # Copia JSON de acompanhamento atualizado
    shutil.copy2(
        path_acompanhamento_output,
        PASTA_OUTPUT_JSON_DINAMICO / f"{audio_id}_segments_acompanhamento.json"
    )
    print(f"[INFO] Copiado: {audio_id}_segments_acompanhamento.json")

    # Copia JSON filtrado atualizado (se existir)
    if json_filtrado is not None:
        shutil.copy2(
            path_filtrado_output,
            PASTA_OUTPUT_JSON_DINAMICO / f"{audio_id}.json"
        )
        print(f"[INFO] Copiado: {audio_id}.json")
    
    print()
    print("=" * 70)
    print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    # Execucao direta exige o audio_id como argumento - nao ha id fixo
    # no codigo. Mesmo padrao do m15_cleanup.py.
    if len(sys.argv) != 2:
        print("Uso: python src/m12_denoiser_deepfilternet3.py <audio_id>")
        sys.exit(1)

    sys.exit(0 if main(sys.argv[1]) else 1)
