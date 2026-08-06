#!/usr/bin/env python3
"""
Module m13_normalizador_audio.py
Normalizes audio segments using SoX
Adds 'sox_*' and 'utilizou_sox' fields to the JSON metadata
"""

import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, Optional, Tuple

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import SOX_NORMALIZER, PROJECT_ROOT


# ==============================================================================
# FUNCOES AUXILIARES - JSON
# ==============================================================================

def carregar_json(caminho: Path) -> Optional[Dict]:
    """
    Loads a JSON file
    Returns None if the file does not exist
    """
    if not caminho.exists():
        return None
    
    try:
        with open(caminho, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON {caminho}: {e}")
        return None
    except Exception as e:
        print(f"Erro ao carregar JSON {caminho}: {e}")
        return None


def salvar_json(dados: Dict, caminho: Path) -> bool:
    """
    Saves data to a JSON file with an atomic write
    Returns True on success, False otherwise
    """
    try:
        # Criar pasta pai se nao existir
        caminho.parent.mkdir(parents=True, exist_ok=True)
        
        # Atomic write: escrever em arquivo temporario + rename
        caminho_temp = caminho.with_suffix('.tmp')
        with open(caminho_temp, 'w', encoding='utf-8') as f:
            json.dump(dados, f, ensure_ascii=False, indent=2)
        
        # Rename atomico
        caminho_temp.replace(caminho)
        return True
        
    except Exception as e:
        print(f"Erro ao salvar JSON {caminho}: {e}")
        return False


# ==============================================================================
# FUNCOES AUXILIARES - SOX
# ==============================================================================

def verificar_sox_instalado() -> bool:
    """
    Checks whether SoX is installed and accessible
    Returns True if available, False otherwise
    """
    try:
        resultado = subprocess.run(
            ['sox', '--version'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=5
        )
        return resultado.returncode == 0
    except FileNotFoundError:
        print("ERRO: SoX nao encontrado. Instale com: sudo apt-get install sox")
        return False
    except Exception as e:
        print(f"Erro ao verificar instalacao do SoX: {e}")
        return False


def construir_comando_sox(
    caminho_input: Path,
    caminho_output: Path,
    config: Dict
) -> list:
    """
    Builds the SoX command based on config parameters
    Correct syntax: sox [input_options] input [output_options] output [effects]
    Returns a list of arguments for subprocess
    """
    comando = ['sox']
    
    # Arquivo de entrada (sem opcoes especiais)
    comando.append(str(caminho_input))
    
    # OPCOES DE OUTPUT (antes do arquivo de saida)
    # Bit depth
    comando.extend(['-b', str(config['bit_depth'])])
    
    # Channels (mono/stereo)
    comando.extend(['-c', str(config['channels'])])
    
    # Arquivo de saida
    comando.append(str(caminho_output))
    
    # EFEITOS (depois do arquivo de saida)
    
    # Sample rate (efeito 'rate')
    comando.extend(['rate', '-h', str(config['sample_rate'])])
    
    # Normalizacao de volume
    normalize_method = config['normalize_method']
    target_db = config['target_level_db']
    
    if normalize_method == 'peak':
        # Normaliza baseado no pico maximo
        comando.extend(['norm', str(target_db)])
    elif normalize_method == 'rms':
        # Normaliza baseado na energia media (RMS)
        comando.extend(['gain', '-n', str(target_db)])
    elif normalize_method == 'loudness':
        # Normaliza baseado em percepcao humana (LUFS)
        comando.extend(['loudness', str(target_db)])

    return comando


def normalizar_audio(
    caminho_input: Path,
    caminho_output: Path,
    config: Dict
) -> bool:
    """
    Normalizes audio using SoX
    Returns True on success, False otherwise
    """
    try:
        # Criar pasta de saida se nao existir
        caminho_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Construir comando
        comando = construir_comando_sox(caminho_input, caminho_output, config)
        
        # Executar SoX
        resultado = subprocess.run(
            comando,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=300  # Timeout de 5 minutos
        )
        
        if resultado.returncode != 0:
            print(f"Erro SoX no arquivo {caminho_input.name}:")
            print(f"  STDERR: {resultado.stderr}")
            return False
        
        # Validar arquivo criado
        if not caminho_output.exists():
            print(f"Erro: Arquivo de saida nao foi criado: {caminho_output}")
            return False
        
        if caminho_output.stat().st_size == 0:
            print(f"Erro: Arquivo de saida vazio: {caminho_output}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        print(f"Timeout ao processar {caminho_input.name}")
        return False
    except Exception as e:
        print(f"Erro ao normalizar {caminho_input.name}: {e}")
        return False


# ==============================================================================
# FUNCOES DE PROCESSAMENTO
# ==============================================================================

def obter_caminho_input_audio(
    nome_audio: str,
    metadados: Dict,
    pasta_audios_denoiser: Path,
    pasta_audios_originais: Path
) -> Optional[Path]:
    """
    Determines the audio input path based on 'utilizou_denoiser'
    Prioritizes the denoiser folder if flag=True, otherwise uses the
    originals
    Returns a Path if the file exists, None otherwise

    Args:
        nome_audio: Audio file name
        metadados: Segment metadata
        pasta_audios_denoiser: Path to the 10-denoiser folder
        pasta_audios_originais: Path to the 02-segmentos_originais folder
    """
    usou_denoiser = metadados.get('utilizou_denoiser', False)
    
    if usou_denoiser:
        caminho = pasta_audios_denoiser / nome_audio
    else:
        caminho = pasta_audios_originais / nome_audio
    
    if not caminho.exists():
        print(f"AVISO: Audio nao encontrado: {caminho}")
        return None
    
    return caminho


def renomear_chave_json(nome_original: str, novo_formato: str) -> str:
    """
    Renames the JSON key, changing the extension according to the
    output format
    Example: "video_001.flac" + "wav" -> "video_001.wav"
    """
    caminho = Path(nome_original)
    novo_nome = caminho.stem + '.' + novo_formato
    return novo_nome


def adicionar_campos_sox(metadados: Dict, config: Dict, processado: bool) -> Dict:
    """
    Adds sox_* fields to the metadata dictionary
    If processado=True, uses config values
    If processado=False, sets all of them to null
    """
    metadados_atualizados = metadados.copy()
    
    if processado:
        metadados_atualizados['sox_sample_rate'] = config['sample_rate']
        metadados_atualizados['sox_bit_depth'] = config['bit_depth']
        metadados_atualizados['sox_channels'] = config['channels']
        metadados_atualizados['sox_output_format'] = config['output_format']
        metadados_atualizados['sox_normalize_method'] = config['normalize_method']
        metadados_atualizados['sox_target_level_db'] = config['target_level_db']
        metadados_atualizados['utilizou_sox'] = True
    else:
        metadados_atualizados['sox_sample_rate'] = None
        metadados_atualizados['sox_bit_depth'] = None
        metadados_atualizados['sox_channels'] = None
        metadados_atualizados['sox_output_format'] = None
        metadados_atualizados['sox_normalize_method'] = None
        metadados_atualizados['sox_target_level_db'] = None
        metadados_atualizados['utilizou_sox'] = False
    
    return metadados_atualizados


def limpar_pasta_vazia(pasta: Path):
    """
    Removes the folder if it is empty (no files)
    Used to avoid empty folders in the dataset
    """
    if not pasta.exists():
        return
    
    try:
        # Verificar se pasta esta vazia (sem arquivos, pode ter subpastas vazias)
        arquivos = list(pasta.rglob('*'))
        arquivos_reais = [f for f in arquivos if f.is_file()]
        
        if len(arquivos_reais) == 0:
            # Pasta vazia: remover
            shutil.rmtree(pasta)
            print(f"\nPasta vazia removida: {pasta}")
    except Exception as e:
        print(f"Aviso: Nao foi possivel remover pasta vazia {pasta}: {e}")


def processar_normalizacao(audio_id: str) -> Optional[Tuple[int, int, int]]:
    """
    Main processing function.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        Tuple (processados, pulados, falhados), or None on a hard
        failure (SoX missing or the tracking JSON missing). None
        distinguishes the failure from a legitimately empty batch,
        which returns (0, 0, 0).
    """
    # Definir caminhos baseados no audio_id
    PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "00-json_dinamico"
    PASTA_AUDIOS_DENOISER = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "10-denoiser"
    PASTA_AUDIOS_ORIGINAIS = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "02-segmentos_originais"
    PASTA_OUTPUT_NORMALIZADOR = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "11-normalizador_audio"
    PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO
    PASTA_OUTPUT_DATASET = PROJECT_ROOT / "dataset" / "audio_dataset" / audio_id
    
    print("\n" + "="*70)
    print("INICIANDO NORMALIZACAO DE AUDIO COM SOX")
    print("="*70)
    
    # Verificar instalacao do SoX
    if not verificar_sox_instalado():
        return None
    
    # Carregar JSONs de entrada
    print(f"\nCarregando JSONs de: {PASTA_JSON_DINAMICO}")
    
    json_acompanhamento = carregar_json(
        PASTA_JSON_DINAMICO / f"{audio_id}_segments_acompanhamento.json"
    )

    if json_acompanhamento is None:
        print(f"ERRO: Arquivo de acompanhamento nao encontrado")
        return None

    json_filtro = carregar_json(PASTA_JSON_DINAMICO / f"{audio_id}.json")
    
    # Determinar segmentos a processar
    if json_filtro is not None:
        print(f"Filtro encontrado: processando apenas segmentos elegiveis")
        segmentos_processar = json_filtro
    else:
        print(f"Filtro nao encontrado: processando TODOS os segmentos")
        segmentos_processar = json_acompanhamento
    
    print(f"Total de segmentos a processar: {len(segmentos_processar)}")
    print(f"Total de segmentos no acompanhamento: {len(json_acompanhamento)}")
    
    # Contadores
    processados = 0
    pulados = 0
    falhados = 0
    
    # Dicionarios para armazenar metadados atualizados
    acompanhamento_atualizado = {}
    normalizado_atualizado = {}
    
    # Processar cada segmento
    print(f"\nProcessando segmentos...")
    
    for nome_audio, metadados in json_acompanhamento.items():
        # Verificar se segmento e elegivel
        if nome_audio not in segmentos_processar:
            # Nao elegivel: adicionar campos null
            metadados_updated = adicionar_campos_sox(metadados, SOX_NORMALIZER, False)
            acompanhamento_atualizado[nome_audio] = metadados_updated
            pulados += 1
            continue
        
        # Segmento elegivel: processar
        print(f"\nProcessando: {nome_audio}")
        
        # Obter caminho de input
        caminho_input = obter_caminho_input_audio(
            nome_audio, 
            metadados,
            PASTA_AUDIOS_DENOISER,
            PASTA_AUDIOS_ORIGINAIS
        )
        if caminho_input is None:
            falhados += 1
            metadados_updated = adicionar_campos_sox(metadados, SOX_NORMALIZER, False)
            acompanhamento_atualizado[nome_audio] = metadados_updated
            continue
        
        # Determinar nome de saida com nova extensao
        nome_output = renomear_chave_json(nome_audio, SOX_NORMALIZER['output_format'])
        caminho_output = PASTA_OUTPUT_DATASET / nome_output
        
        print(f"  Input: {caminho_input}")
        print(f"  Output: {caminho_output}")
        
        # Normalizar audio
        sucesso = normalizar_audio(caminho_input, caminho_output, SOX_NORMALIZER)
        
        if sucesso:
            print(f"  Status: SUCESSO")
            processados += 1
            
            # Adicionar campos sox com valores reais
            metadados_updated = adicionar_campos_sox(metadados, SOX_NORMALIZER, True)
            
            # Atualizar dicionarios com chave renomeada
            acompanhamento_atualizado[nome_output] = metadados_updated
            normalizado_atualizado[nome_output] = metadados_updated
        else:
            print(f"  Status: FALHA")
            falhados += 1
            
            # Adicionar campos null
            metadados_updated = adicionar_campos_sox(metadados, SOX_NORMALIZER, False)
            acompanhamento_atualizado[nome_audio] = metadados_updated
    
    # Salvar JSONs atualizados na pasta 11-normalizador_audio
    print(f"\nSalvando JSONs atualizados em: {PASTA_OUTPUT_NORMALIZADOR}")
    
    caminho_acompanhamento_output = PASTA_OUTPUT_NORMALIZADOR / f"{audio_id}_segments_acompanhamento.json"
    caminho_normalizado_output = PASTA_OUTPUT_NORMALIZADOR / f"{audio_id}_normalizado.json"
    
    salvar_json(acompanhamento_atualizado, caminho_acompanhamento_output)
    
    if normalizado_atualizado:
        salvar_json(normalizado_atualizado, caminho_normalizado_output)
    
    # Copiar JSONs para pasta 00-json_dinamico (sobrescrever)
    print(f"\nCopiando JSONs para: {PASTA_OUTPUT_JSON_DINAMICO}")
    
    caminho_acompanhamento_dinamico = PASTA_OUTPUT_JSON_DINAMICO / f"{audio_id}_segments_acompanhamento.json"
    caminho_filtro_dinamico = PASTA_OUTPUT_JSON_DINAMICO / f"{audio_id}.json"
    
    salvar_json(acompanhamento_atualizado, caminho_acompanhamento_dinamico)
    
    if normalizado_atualizado:
        salvar_json(normalizado_atualizado, caminho_filtro_dinamico)
    
    # Limpar pasta de dataset se ficou vazia
    limpar_pasta_vazia(PASTA_OUTPUT_DATASET)
    
    return processados, pulados, falhados


# ==============================================================================
# MAIN
# ==============================================================================

def main(audio_id: str) -> bool:
    """
    Main function.

    Args:
        audio_id: ID of the audio file to process

    Returns:
        True if the normalization ran, False on a hard failure (SoX
        missing or the tracking JSON missing).
    """
    resultado = processar_normalizacao(audio_id)

    if resultado is None:
        print("\n" + "="*70)
        print("NORMALIZACAO DE AUDIO NAO EXECUTADA - falha dura")
        print("="*70 + "\n")
        return False

    processados, pulados, falhados = resultado

    print("\n" + "="*70)
    print("RESUMO DO PROCESSAMENTO")
    print("="*70)
    print(f"Segmentos processados com sucesso: {processados}")
    print(f"Segmentos pulados (nao elegiveis): {pulados}")
    print(f"Segmentos com falha: {falhados}")
    print(f"Total: {processados + pulados + falhados}")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    # Execucao direta exige o audio_id como argumento - nao ha id fixo
    # no codigo. Mesmo padrao do m15_cleanup.py.
    if len(sys.argv) != 2:
        print("Uso: python src/m13_normalizador_audio.py <audio_id>")
        sys.exit(1)

    sys.exit(0 if main(sys.argv[1]) else 1)
