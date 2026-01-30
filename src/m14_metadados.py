#!/usr/bin/env python3
"""
Modulo m14_metadados.py
Gera metadados finais do dataset em formato CSV
Copia arquivo de acompanhamento JSON para historico
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil


# ==============================================================================
# CONFIGURACAO DE PATHS
# ==============================================================================

# Adicionar pasta raiz ao path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# CONFIGURACAO DE INPUTS/OUTPUTS
# ==============================================================================

# ID do video a processar
id_video = 'QN7gUP7nYhQ'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"
ARQUIVO_JSON_FILTRADO = PASTA_JSON_DINAMICO / f"{id_video}.json"
ARQUIVO_JSON_ACOMPANHAMENTO = PASTA_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"

# Caminhos de saida
PASTA_DATASET = PROJECT_ROOT / "dataset"
ARQUIVO_CSV_DATASET = PASTA_DATASET / "dataset.csv"
PASTA_HISTORICO = PASTA_DATASET / "historico_dataset"
ARQUIVO_JSON_HISTORICO = PASTA_HISTORICO / f"{id_video}.json"

# Configuracoes CSV
CSV_SEPARATOR = '|'
CSV_ENCODING = 'utf-8'

# Colunas fixas iniciais
COLUNAS_FIXAS = ['arquivo_nome', 'caminho']

# Campos a excluir do CSV (mesmo se presentes no JSON)
CAMPOS_EXCLUIDOS = [
    'stt_leg_normalizado',
    'stt_whisper_normalizado',
    'stt_wav2vec_normalizado'
]


# ==============================================================================
# FUNCOES AUXILIARES
# ==============================================================================

def extrair_id_video(nome_arquivo: str) -> str:
    """
    Extrai ID do video (11 primeiros caracteres) do nome do arquivo
    
    Args:
        nome_arquivo: Nome do arquivo de audio
        
    Returns:
        ID do video (11 caracteres)
    """
    return nome_arquivo[:11]


def construir_caminho_audio(nome_arquivo: str) -> str:
    """
    Constroi caminho relativo do audio baseado no ID
    
    Args:
        nome_arquivo: Nome do arquivo de audio
        
    Returns:
        Caminho relativo completo (ex: ./audio_dataset/QN7gUP7nYhQ/QN7gUP7nYhQ_001.flac)
    """
    id_video = extrair_id_video(nome_arquivo)
    return f"./audio_dataset/{id_video}/{nome_arquivo}"


def converter_bool_para_str(valor: Any) -> Any:
    """
    Converte booleanos para formato True/False capitalizado
    
    Args:
        valor: Valor a converter
        
    Returns:
        Valor convertido (True/False se booleano, original caso contrario)
    """
    if isinstance(valor, bool):
        return str(valor)
    return valor


def carregar_json(caminho: Path) -> Optional[Dict[str, Any]]:
    """
    Carrega arquivo JSON
    
    Args:
        caminho: Caminho do arquivo JSON
        
    Returns:
        Dicionario com dados ou None se arquivo nao existe
    """
    if not caminho.exists():
        return None
    
    with open(caminho, 'r', encoding='utf-8') as f:
        return json.load(f)


def obter_todas_colunas(dados_json: Dict[str, Any]) -> List[str]:
    """
    Obtem lista de todas as colunas unicas do JSON (ordem de aparicao)
    Exclui campos definidos em CAMPOS_EXCLUIDOS
    
    Args:
        dados_json: Dicionario com metadados dos segmentos
        
    Returns:
        Lista de nomes de colunas (colunas fixas + campos do JSON filtrados)
    """
    colunas = COLUNAS_FIXAS.copy()
    colunas_json = set()
    
    # Coletar todas as chaves unicas mantendo ordem de primeira aparicao
    for segmento_data in dados_json.values():
        for chave in segmento_data.keys():
            if chave not in colunas_json and chave not in CAMPOS_EXCLUIDOS:
                colunas.append(chave)
                colunas_json.add(chave)
    
    return colunas


def ler_csv_existente(caminho: Path) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    Le CSV existente e retorna colunas e dados
    
    Args:
        caminho: Caminho do arquivo CSV
        
    Returns:
        Tupla (colunas, linhas) onde linhas é lista de dicionarios
    """
    if not caminho.exists():
        return [], []
    
    with open(caminho, 'r', encoding=CSV_ENCODING, newline='') as f:
        reader = csv.DictReader(f, delimiter=CSV_SEPARATOR)
        colunas = reader.fieldnames or []
        linhas = list(reader)
    
    return colunas, linhas


def mesclar_colunas(colunas_existentes: List[str], colunas_novas: List[str]) -> List[str]:
    """
    Mescla colunas existentes com novas (adiciona novas ao final)
    
    Args:
        colunas_existentes: Colunas ja presentes no CSV
        colunas_novas: Colunas do JSON atual
        
    Returns:
        Lista de colunas mescladas
    """
    colunas_final = colunas_existentes.copy()
    
    for col in colunas_novas:
        if col not in colunas_final:
            colunas_final.append(col)
    
    return colunas_final


def preparar_linha_csv(nome_arquivo: str, dados_segmento: Dict[str, Any], 
                       colunas: List[str]) -> Dict[str, Any]:
    """
    Prepara linha de dados para CSV
    
    Args:
        nome_arquivo: Nome do arquivo de audio
        dados_segmento: Dados do segmento do JSON
        colunas: Lista de todas as colunas
        
    Returns:
        Dicionario com dados da linha (todas as colunas preenchidas)
    """
    linha = {}
    
    for col in colunas:
        if col == 'arquivo_nome':
            linha[col] = nome_arquivo
        elif col == 'caminho':
            linha[col] = construir_caminho_audio(nome_arquivo)
        elif col in dados_segmento:
            linha[col] = converter_bool_para_str(dados_segmento[col])
        else:
            linha[col] = 'null'
    
    return linha


def escrever_csv(caminho: Path, colunas: List[str], linhas: List[Dict[str, Any]]):
    """
    Escreve CSV com colunas e linhas
    
    Args:
        caminho: Caminho do arquivo CSV
        colunas: Lista de nomes das colunas
        linhas: Lista de dicionarios com dados
    """
    # Garantir que pasta existe
    caminho.parent.mkdir(parents=True, exist_ok=True)
    
    with open(caminho, 'w', encoding=CSV_ENCODING, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=colunas, delimiter=CSV_SEPARATOR)
        writer.writeheader()
        writer.writerows(linhas)


def copiar_json_historico(origem: Path, destino: Path):
    """
    Copia arquivo JSON de acompanhamento para historico
    
    Args:
        origem: Caminho do arquivo JSON de origem
        destino: Caminho do arquivo JSON de destino
    """
    # Garantir que pasta existe
    destino.parent.mkdir(parents=True, exist_ok=True)
    
    # Copiar arquivo (sobrescreve se ja existe)
    shutil.copy2(origem, destino)
    print(f"JSON de historico copiado: {destino}")


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def processar_metadados():
    """
    Processa metadados e gera outputs:
    1. Adiciona/atualiza dataset.csv
    2. Copia JSON de acompanhamento para historico
    """
    print(f"Processando metadados para video: {id_video}")
    print("-" * 80)
    
    # Carregar JSON (prioridade: filtrado, fallback: acompanhamento)
    dados_json = carregar_json(ARQUIVO_JSON_FILTRADO)
    fonte_json = "filtrado"
    
    if dados_json is None:
        print(f"Arquivo filtrado nao encontrado: {ARQUIVO_JSON_FILTRADO}")
        print(f"Usando arquivo de acompanhamento: {ARQUIVO_JSON_ACOMPANHAMENTO}")
        dados_json = carregar_json(ARQUIVO_JSON_ACOMPANHAMENTO)
        fonte_json = "acompanhamento"
        
        if dados_json is None:
            print(f"ERRO: Arquivo de acompanhamento nao encontrado: {ARQUIVO_JSON_ACOMPANHAMENTO}")
            return
    else:
        print(f"Usando arquivo filtrado: {ARQUIVO_JSON_FILTRADO}")
    
    print(f"Total de segmentos a processar: {len(dados_json)}")
    print("-" * 80)
    
    # Obter colunas do JSON atual
    colunas_json = obter_todas_colunas(dados_json)
    
    # Ler CSV existente (se houver)
    colunas_csv_existentes, linhas_csv_existentes = ler_csv_existente(ARQUIVO_CSV_DATASET)
    
    # Mesclar colunas
    colunas_finais = mesclar_colunas(colunas_csv_existentes, colunas_json)
    
    # Atualizar linhas existentes com novas colunas (preencher com null)
    if colunas_csv_existentes and colunas_finais != colunas_csv_existentes:
        print(f"Novas colunas detectadas: {set(colunas_finais) - set(colunas_csv_existentes)}")
        for linha in linhas_csv_existentes:
            for col in colunas_finais:
                if col not in linha:
                    linha[col] = 'null'
    
    # Preparar novas linhas
    linhas_novas = []
    for nome_arquivo, dados_segmento in dados_json.items():
        linha = preparar_linha_csv(nome_arquivo, dados_segmento, colunas_finais)
        linhas_novas.append(linha)
    
    # Combinar linhas existentes + novas
    linhas_totais = linhas_csv_existentes + linhas_novas
    
    # Escrever CSV
    escrever_csv(ARQUIVO_CSV_DATASET, colunas_finais, linhas_totais)
    print(f"CSV atualizado: {ARQUIVO_CSV_DATASET}")
    print(f"Total de linhas no CSV: {len(linhas_totais)} ({len(linhas_csv_existentes)} existentes + {len(linhas_novas)} novas)")
    print("-" * 80)
    
    # Copiar JSON de acompanhamento para historico
    copiar_json_historico(ARQUIVO_JSON_ACOMPANHAMENTO, ARQUIVO_JSON_HISTORICO)
    print("-" * 80)
    print("Processamento concluido com sucesso!")


# ==============================================================================
# EXECUCAO
# ==============================================================================

if __name__ == "__main__":
    processar_metadados()