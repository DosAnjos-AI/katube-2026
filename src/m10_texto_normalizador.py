#!/usr/bin/env python3
"""
Modulo m10_texto_normalizador.py
Normaliza textos de legendas e transcrições STT (Whisper e WAV2VEC2)
Adiciona campos normalizados aos metadados JSON para análise WER
"""

import sys
import json
import re
import unicodedata
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from shutil import copy2

# Adicionar pasta raiz ao path para importar config
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import TEXT_NORMALIZER, PROJECT_ROOT


# ==============================================================================
# CONFIGURACAO
# ==============================================================================

# ID do video a processar
id_video = 'QN7gUP7nYhQ'

# Caminhos de entrada
PASTA_JSON_DINAMICO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "00-json_dinamico"

# Caminhos de saida
PASTA_OUTPUT_NORMALIZADO = PROJECT_ROOT / "arquivos" / "temp" / id_video / "08-normalizador_texto"
PASTA_OUTPUT_JSON_DINAMICO = PASTA_JSON_DINAMICO  # Sobrescreve na mesma pasta

# Arquivos de entrada/saida
ARQUIVO_FILTRO = PASTA_JSON_DINAMICO / f"{id_video}.json"
ARQUIVO_ACOMPANHAMENTO = PASTA_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# MAPEAMENTO DE CARACTERES E SIGLAS
# ==============================================================================

# Mapeamento de caracteres especiais para portugues
CHARS_MAP = str.maketrans({
    'ï': 'i', 'ù': 'u', 'ö': 'o', 'î': 'i', 'ñ': 'n',
    'ë': 'e', 'ì': 'i', 'ò': 'o', 'ů': 'u', 'ẽ': 'e',
    'ü': 'u', 'è': 'e', 'æ': 'a', 'å': 'a', 'ø': 'o',
    'þ': 't', 'ð': 'd', 'ß': 's', 'ł': 'l', 'đ': 'd',
    'ć': 'c', 'č': 'c', 'š': 's', 'ž': 'z', 'ý': 'y'
})

# Mapeamento de siglas e abreviacoes comuns em portugues brasileiro
SIGLAS_ABREVIACOES = {
    # Tratamentos e titulos
    r'\bsr\.?\b': 'senhor',
    r'\bsra\.?\b': 'senhora',
    r'\bdr\.?\b': 'doutor',
    r'\bdra\.?\b': 'doutora',
    r'\bprof\.?\b': 'professor',
    r'\bprofa\.?\b': 'professora',
    
    # Unidades de medida
    r'\bkm\/h\b': 'quilometros por hora',
    r'\bkm\b': 'quilometros',
    r'\bkg\b': 'quilogramas',
    r'\bg\b': 'gramas',
    r'\bm\b': 'metros',
    r'\bcm\b': 'centimetros',
    r'\bmm\b': 'milimetros',
    
    # Expressoes comuns
    r'\betc\.?\b': 'etcetera',
    r'\bobs\.?\b': 'observacao',
    r'\bpag\.?\b': 'pagina',
    r'\btel\.?\b': 'telefone',
    r'\bcep\.?\b': 'codigo de enderecamento postal',
    r'\bcpf\.?\b': 'cadastro de pessoas fisicas',
    r'\brg\.?\b': 'registro geral',
}


# ==============================================================================
# FUNCOES DE CONVERSAO NUMERICA
# ==============================================================================

def number_to_words_pt(num: int, with_accents: bool = False) -> str:
    """
    Converte numero para extenso em portugues
    Suporta numeros de 0 ate 999.999.999
    
    Args:
        num: Numero inteiro para converter
        with_accents: Se True, inclui acentuacao (tres, decimo)
                     Se False, sem acentos (tres, decimo)
    
    Returns:
        Numero por extenso
    """
    if num == 0:
        return 'zero'
    
    # Unidades
    if with_accents:
        unidades = ['', 'um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
    else:
        unidades = ['', 'um', 'dois', 'tres', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove']
    
    # 10 a 19
    especiais = ['dez', 'onze', 'doze', 'treze', 'quatorze', 'quinze',
                 'dezesseis', 'dezessete', 'dezoito', 'dezenove']
    
    # Dezenas
    dezenas = ['', '', 'vinte', 'trinta', 'quarenta', 'cinquenta',
               'sessenta', 'setenta', 'oitenta', 'noventa']
    
    # Centenas
    centenas = ['', 'cento', 'duzentos', 'trezentos', 'quatrocentos',
                'quinhentos', 'seiscentos', 'setecentos', 'oitocentos', 'novecentos']
    
    def converter_ate_999(n):
        if n == 0:
            return ''
        elif n < 10:
            return unidades[n]
        elif n < 20:
            return especiais[n - 10]
        elif n < 100:
            dez = n // 10
            uni = n % 10
            if uni == 0:
                return dezenas[dez]
            return f"{dezenas[dez]} e {unidades[uni]}"
        else:  # n < 1000
            cen = n // 100
            resto = n % 100
            if n == 100:
                return 'cem'
            elif resto == 0:
                return centenas[cen]
            return f"{centenas[cen]} e {converter_ate_999(resto)}"
    
    if num < 1000:
        return converter_ate_999(num)
    elif num < 1000000:
        milhares = num // 1000
        resto = num % 1000
        if milhares == 1:
            mil_text = 'mil'
        else:
            mil_text = f"{converter_ate_999(milhares)} mil"
        
        if resto == 0:
            return mil_text
        return f"{mil_text} e {converter_ate_999(resto)}"
    else:
        milhoes = num // 1000000
        resto = num % 1000000
        if milhoes == 1:
            milhao_text = 'um milhao'
        else:
            milhao_text = f"{converter_ate_999(milhoes)} milhoes"
        
        if resto == 0:
            return milhao_text
        elif resto < 1000:
            return f"{milhao_text} e {converter_ate_999(resto)}"
        else:
            return f"{milhao_text} {number_to_words_pt(resto, with_accents)}"


def ordinal_to_words_pt(num: int, gender: str = 'm', with_accents: bool = False) -> str:
    """
    Converte numero ordinal para extenso
    
    Args:
        num: Numero ordinal
        gender: 'm' para masculino, 'f' para feminino
        with_accents: Se True, inclui acentuacao
    
    Returns:
        Ordinal por extenso
    """
    if with_accents:
        ordinais_m = {
            1: 'primeiro', 2: 'segundo', 3: 'terceiro', 4: 'quarto', 5: 'quinto',
            6: 'sexto', 7: 'sétimo', 8: 'oitavo', 9: 'nono', 10: 'décimo',
            11: 'décimo primeiro', 12: 'décimo segundo', 13: 'décimo terceiro',
            14: 'décimo quarto', 15: 'décimo quinto', 16: 'décimo sexto',
            17: 'décimo sétimo', 18: 'décimo oitavo', 19: 'décimo nono',
            20: 'vigésimo', 30: 'trigésimo', 40: 'quadragésimo',
            50: 'quinquagésimo', 60: 'sexagésimo', 70: 'septuagésimo',
            80: 'octogésimo', 90: 'nonagésimo', 100: 'centésimo'
        }
    else:
        ordinais_m = {
            1: 'primeiro', 2: 'segundo', 3: 'terceiro', 4: 'quarto', 5: 'quinto',
            6: 'sexto', 7: 'setimo', 8: 'oitavo', 9: 'nono', 10: 'decimo',
            11: 'decimo primeiro', 12: 'decimo segundo', 13: 'decimo terceiro',
            14: 'decimo quarto', 15: 'decimo quinto', 16: 'decimo sexto',
            17: 'decimo setimo', 18: 'decimo oitavo', 19: 'decimo nono',
            20: 'vigesimo', 30: 'trigesimo', 40: 'quadragesimo',
            50: 'quinquagesimo', 60: 'sexagesimo', 70: 'septuagesimo',
            80: 'octogesimo', 90: 'nonagesimo', 100: 'centesimo'
        }
    
    ordinais_f = {k: v.replace('o', 'a') for k, v in ordinais_m.items()}
    ordinais = ordinais_f if gender == 'f' else ordinais_m
    
    return ordinais.get(num, f"{num}o")


# ==============================================================================
# FUNCOES DE NORMALIZACAO
# ==============================================================================

def apply_char_mapping(text: str) -> str:
    """
    Aplica mapeamento de caracteres especiais
    
    Args:
        text: Texto para aplicar mapeamento
    
    Returns:
        Texto com caracteres mapeados
    """
    return text.translate(CHARS_MAP)


def expand_abbreviations(text: str) -> str:
    """
    Expande siglas e abreviacoes comuns
    
    Args:
        text: Texto com possíveis siglas
    
    Returns:
        Texto com siglas expandidas
    """
    for pattern, replacement in SIGLAS_ABREVIACOES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def advanced_number_to_text(text: str, with_accents: bool = False) -> str:
    """
    Conversao avancada de numeros para texto
    Suporta: decimais, percentuais, ordinais
    
    Args:
        text: Texto com numeros
        with_accents: Se True, numeros por extenso com acento
    
    Returns:
        Texto com numeros convertidos
    """
    # Ordinais (1º, 2ª, 3º, etc.)
    def replace_ordinal(match):
        num = int(match.group(1))
        gender = 'f' if match.group(2) == 'ª' else 'm'
        return ordinal_to_words_pt(num, gender, with_accents)
    
    text = re.sub(r'(\d+)[ºª]', replace_ordinal, text)
    
    # Decimais (ex: 3.14, 2,5)
    def replace_decimal(match):
        inteiro = int(match.group(1))
        decimal = match.group(2)
        int_text = number_to_words_pt(inteiro, with_accents)
        dec_text = ' '.join([number_to_words_pt(int(d), with_accents) for d in decimal])
        return f"{int_text} virgula {dec_text}"
    
    text = re.sub(r'(\d+)[,\.](\d+)', replace_decimal, text)
    
    # Percentuais
    def replace_percent(match):
        num = int(match.group(1))
        return f"{number_to_words_pt(num, with_accents)} por cento"
    
    text = re.sub(r'(\d+)%', replace_percent, text)
    
    # Numeros inteiros restantes
    def replace_integer(match):
        num = int(match.group(0))
        return number_to_words_pt(num, with_accents)
    
    text = re.sub(r'\b\d+\b', replace_integer, text)
    
    return text


def remove_html_tags(text: str) -> str:
    """
    Remove tags HTML do texto
    
    Args:
        text: Texto com possiveis tags HTML
    
    Returns:
        Texto limpo
    """
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def normalize_text(
    text: str,
    remove_punctuation: bool = True,
    remove_accents: bool = True
) -> Optional[str]:
    """
    Normalizacao completa do texto para analise STT
    
    Args:
        text: Texto para normalizar
        remove_punctuation: Se True, remove pontuacao de diccao (.,;!?_)
        remove_accents: Se True, remove acentos graficos ('`^~)
    
    Returns:
        Texto normalizado ou None se vazio
    """
    if not text or text.strip() == "":
        return None
    
    # Remove HTML
    text = remove_html_tags(text)
    
    # Lowercase
    text = text.lower()
    
    # Expande siglas e abreviacoes
    text = expand_abbreviations(text)
    
    # Converte numeros para texto (com ou sem acentos conforme config)
    text = advanced_number_to_text(text, with_accents=not remove_accents)
    
    # Remove ou mantem acentos graficos
    if remove_accents:
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
    
    # Aplica mapeamento de caracteres especiais
    text = apply_char_mapping(text)
    
    # Remove pontuacao de diccao se configurado
    if remove_punctuation:
        # Remove pontuacao que afeta diccao: . , ; ! ? _
        text = re.sub(r'[.,;!?_]', ' ', text)
        # Remove outros caracteres especiais restantes
        text = re.sub(r'[^\w\s]', ' ', text)
    else:
        # Mantem pontuacao de diccao, remove apenas outros caracteres especiais
        # Preserva: . , ; ! ? _ (e letras, numeros, espacos)
        text = re.sub(r'[^\w\s.,;!?_]', ' ', text)
    
    # Normaliza espacos multiplos para espaco unico
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


# ==============================================================================
# FUNCOES DE PROCESSAMENTO DE JSON
# ==============================================================================

def load_json(filepath: Path) -> Optional[Dict]:
    """
    Carrega arquivo JSON
    
    Args:
        filepath: Caminho do arquivo JSON
    
    Returns:
        Dicionario com dados ou None se erro
    """
    if not filepath.exists():
        logger.warning(f"Arquivo nao encontrado: {filepath}")
        return None
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Erro ao carregar JSON {filepath}: {e}")
        return None


def save_json(data: Dict, filepath: Path) -> bool:
    """
    Salva dados em arquivo JSON
    
    Args:
        data: Dicionario para salvar
        filepath: Caminho do arquivo de saida
    
    Returns:
        True se sucesso, False se erro
    """
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Erro ao salvar JSON {filepath}: {e}")
        return False


def normalizar_segmentos(
    dados_acompanhamento: Dict,
    segmentos_elegiveis: Optional[List[str]] = None
) -> Tuple[Dict, Dict]:
    """
    Normaliza textos dos segmentos
    
    Args:
        dados_acompanhamento: Dados completos de todos segmentos
        segmentos_elegiveis: Lista de IDs de segmentos para processar
                            Se None, processa todos
    
    Returns:
        Tupla (dados_acompanhamento_atualizado, dados_elegiveis_normalizados)
    """
    # Configuracoes de normalizacao
    remove_punct = TEXT_NORMALIZER.get('remove_punctuation_diction', True)
    remove_acc = TEXT_NORMALIZER.get('remove_accents_graphic', True)
    
    logger.info(f"Configuracao de normalizacao:")
    logger.info(f"  - remove_punctuation_diction: {remove_punct}")
    logger.info(f"  - remove_accents_graphic: {remove_acc}")
    
    # Dados de saida
    dados_normalizados_elegiveis = {}
    
    total_segmentos = len(dados_acompanhamento)
    elegiveis_count = len(segmentos_elegiveis) if segmentos_elegiveis else total_segmentos
    
    logger.info(f"Total de segmentos no acompanhamento: {total_segmentos}")
    logger.info(f"Segmentos elegiveis para normalizacao: {elegiveis_count}")
    
    processados = 0
    
    for segment_id, segment_data in dados_acompanhamento.items():
        # Verifica se segmento eh elegivel
        eh_elegivel = (segmentos_elegiveis is None) or (segment_id in segmentos_elegiveis)
        
        if eh_elegivel:
            # Normaliza campo "texto" -> "stt_leg_normalizado"
            if 'texto' in segment_data and segment_data['texto']:
                texto_norm = normalize_text(
                    segment_data['texto'],
                    remove_punctuation=remove_punct,
                    remove_accents=remove_acc
                )
                segment_data['stt_leg_normalizado'] = texto_norm
            else:
                segment_data['stt_leg_normalizado'] = None
            
            # Normaliza campo "stt_whisper" -> "stt_whisper_normalizado"
            if 'stt_whisper' in segment_data and segment_data['stt_whisper']:
                whisper_norm = normalize_text(
                    segment_data['stt_whisper'],
                    remove_punctuation=remove_punct,
                    remove_accents=remove_acc
                )
                segment_data['stt_whisper_normalizado'] = whisper_norm
            else:
                segment_data['stt_whisper_normalizado'] = None
            
            # Normaliza campo "stt_wav2vec" -> "stt_wav2vec_normalizado"
            if 'stt_wav2vec' in segment_data and segment_data['stt_wav2vec']:
                wav2vec_norm = normalize_text(
                    segment_data['stt_wav2vec'],
                    remove_punctuation=remove_punct,
                    remove_accents=remove_acc
                )
                segment_data['stt_wav2vec_normalizado'] = wav2vec_norm
            else:
                segment_data['stt_wav2vec_normalizado'] = None
            
            # Adiciona aos dados normalizados elegiveis
            dados_normalizados_elegiveis[segment_id] = segment_data.copy()
            processados += 1
            
        else:
            # Segmento nao elegivel: adiciona campos como null
            segment_data['stt_leg_normalizado'] = None
            segment_data['stt_whisper_normalizado'] = None
            segment_data['stt_wav2vec_normalizado'] = None
    
    logger.info(f"Segmentos processados com normalizacao: {processados}")
    
    return dados_acompanhamento, dados_normalizados_elegiveis


# ==============================================================================
# FUNCAO PRINCIPAL
# ==============================================================================

def processar_normalizacao():
    """
    Processa normalizacao de textos STT
    
    Fluxo:
    1. Carrega arquivo de acompanhamento (obrigatorio)
    2. Carrega arquivo de filtro (opcional)
    3. Normaliza segmentos elegiveis
    4. Salva em 08-normalizador_texto/
    5. Copia para 00-json_dinamico/ (sobrescreve)
    """
    logger.info("="*70)
    logger.info("MODULO 10: NORMALIZADOR DE TEXTO")
    logger.info("="*70)
    logger.info(f"Video ID: {id_video}")
    logger.info("")
    
    # Verifica arquivo de acompanhamento (obrigatorio)
    if not ARQUIVO_ACOMPANHAMENTO.exists():
        logger.error(f"ERRO: Arquivo de acompanhamento nao encontrado:")
        logger.error(f"  {ARQUIVO_ACOMPANHAMENTO}")
        return False
    
    # Carrega dados de acompanhamento
    logger.info(f"Carregando arquivo de acompanhamento...")
    dados_acompanhamento = load_json(ARQUIVO_ACOMPANHAMENTO)
    
    if dados_acompanhamento is None:
        logger.error("ERRO: Falha ao carregar arquivo de acompanhamento")
        return False
    
    logger.info(f"  OK - {len(dados_acompanhamento)} segmentos encontrados")
    
    # Verifica arquivo de filtro (opcional)
    segmentos_elegiveis = None
    tem_arquivo_filtro = ARQUIVO_FILTRO.exists()
    
    if tem_arquivo_filtro:
        logger.info(f"Carregando arquivo de filtro...")
        dados_filtro = load_json(ARQUIVO_FILTRO)
        
        if dados_filtro:
            segmentos_elegiveis = list(dados_filtro.keys())
            logger.info(f"  OK - {len(segmentos_elegiveis)} segmentos elegiveis")
        else:
            logger.warning("  AVISO: Arquivo de filtro existe mas nao foi carregado")
            logger.info("  Processando todos os segmentos")
    else:
        logger.info("Arquivo de filtro nao encontrado")
        logger.info("  Processando todos os segmentos")
    
    logger.info("")
    logger.info("Normalizando textos...")
    logger.info("-"*70)
    
    # Normaliza segmentos
    dados_acompanhamento_atualizado, dados_elegiveis_normalizados = normalizar_segmentos(
        dados_acompanhamento,
        segmentos_elegiveis
    )
    
    logger.info("-"*70)
    logger.info("")
    
    # Cria pasta de output 01
    PASTA_OUTPUT_NORMALIZADO.mkdir(parents=True, exist_ok=True)
    
    # Salva arquivo de acompanhamento atualizado em 08-normalizador_texto/
    arquivo_acomp_output = PASTA_OUTPUT_NORMALIZADO / f"{id_video}_segments_acompanhamento.json"
    logger.info(f"Salvando arquivo de acompanhamento atualizado...")
    
    if save_json(dados_acompanhamento_atualizado, arquivo_acomp_output):
        logger.info(f"  OK - {arquivo_acomp_output}")
    else:
        logger.error(f"  ERRO ao salvar: {arquivo_acomp_output}")
        return False
    
    # Salva arquivo normalizado (apenas se havia arquivo de filtro)
    if tem_arquivo_filtro and dados_elegiveis_normalizados:
        arquivo_norm_output = PASTA_OUTPUT_NORMALIZADO / f"{id_video}_normalizado.json"
        logger.info(f"Salvando arquivo normalizado (elegiveis)...")
        
        if save_json(dados_elegiveis_normalizados, arquivo_norm_output):
            logger.info(f"  OK - {arquivo_norm_output}")
        else:
            logger.error(f"  ERRO ao salvar: {arquivo_norm_output}")
            return False
    
    logger.info("")
    logger.info("Copiando arquivos para 00-json_dinamico/...")
    logger.info("-"*70)
    
    # Copia arquivo de acompanhamento para 00-json_dinamico/ (sobrescreve)
    arquivo_acomp_destino = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}_segments_acompanhamento.json"
    
    try:
        copy2(arquivo_acomp_output, arquivo_acomp_destino)
        logger.info(f"  OK - Copiado: {arquivo_acomp_destino.name}")
    except Exception as e:
        logger.error(f"  ERRO ao copiar arquivo de acompanhamento: {e}")
        return False
    
    # Copia arquivo normalizado para 00-json_dinamico/ como {id}.json (se existir)
    if tem_arquivo_filtro and dados_elegiveis_normalizados:
        arquivo_norm_source = PASTA_OUTPUT_NORMALIZADO / f"{id_video}_normalizado.json"
        arquivo_norm_destino = PASTA_OUTPUT_JSON_DINAMICO / f"{id_video}.json"
        
        try:
            copy2(arquivo_norm_source, arquivo_norm_destino)
            logger.info(f"  OK - Copiado: {arquivo_norm_destino.name}")
        except Exception as e:
            logger.error(f"  ERRO ao copiar arquivo normalizado: {e}")
            return False
    
    logger.info("-"*70)
    logger.info("")
    logger.info("="*70)
    logger.info("PROCESSAMENTO CONCLUIDO COM SUCESSO")
    logger.info("="*70)
    logger.info(f"Segmentos processados: {len(dados_elegiveis_normalizados) if dados_elegiveis_normalizados else len(dados_acompanhamento_atualizado)}")
    logger.info(f"Pasta output: {PASTA_OUTPUT_NORMALIZADO}")
    logger.info("")
    
    return True


# ==============================================================================
# EXECUCAO
# ==============================================================================

if __name__ == "__main__":
    sucesso = processar_normalizacao()
    
    if not sucesso:
        logger.error("ERRO: Processamento falhou")
        sys.exit(1)
    
    sys.exit(0)