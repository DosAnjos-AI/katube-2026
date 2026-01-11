"""
Módulo 04: Segmentador Inteligente de Áudio
Segmenta áudios baseado em timestamps CSV respeitando locutores e limites de duração
"""

import json
import subprocess
from pathlib import Path
import csv
import sys

# Importar configurações do projeto
sys.path.append(str(Path(__file__).parent.parent))
from config import SEGMENTADOR_AUDIO

id_video= 'kzwB1kLsLes'

# =============================================================================
# VARIÁVEIS DE TESTE (hardcoded para desenvolvimento)
# =============================================================================

PASTA_ORIGEM = "arquivos/temp/"+id_video+"/01-arquivos_originais"  # Substituir pelo caminho real (input: CSV + áudio)
PASTA_DESTINO = "arquivos/temp/"+id_video+"/02-segmentos_originais"  # Substituir pelo caminho real (output: segmentos + JSON)


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def converter_timestamp_para_segundos(timestamp: str) -> float:
    """
    Converte timestamp HH:MM:SS.mmm para segundos
    
    Args:
        timestamp: String no formato HH:MM:SS.mmm
        
    Returns:
        Float com total de segundos
    """
    partes = timestamp.split(':')
    horas = int(partes[0])
    minutos = int(partes[1])
    segundos = float(partes[2])
    
    return horas * 3600 + minutos * 60 + segundos


def segundos_para_timestamp(segundos: float) -> str:
    """
    Converte segundos para timestamp HH:MM:SS.mmm
    
    Args:
        segundos: Float com total de segundos
        
    Returns:
        String no formato HH:MM:SS.mmm
    """
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = segundos % 60
    
    return f"{horas:02d}:{minutos:02d}:{segs:06.3f}"


def detectar_specs_audio(caminho_audio: Path) -> dict:
    """
    Detecta formato, bitrate e sample rate do áudio usando ffprobe
    
    Args:
        caminho_audio: Path do arquivo de áudio
        
    Returns:
        Dict com: formato, bitrate, sample_rate
    """
    try:
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(caminho_audio)
        ]
        
        resultado = subprocess.run(cmd, capture_output=True, text=True, check=True)
        dados = json.loads(resultado.stdout)
        
        # Extrair informações do primeiro stream de áudio
        stream_audio = next(s for s in dados['streams'] if s['codec_type'] == 'audio')
        formato = dados['format']['format_name'].split(',')[0]
        
        specs = {
            'formato': caminho_audio.suffix[1:],  # Remove o ponto da extensão
            'bitrate': stream_audio.get('bit_rate', 'N/A'),
            'sample_rate': stream_audio.get('sample_rate', 'N/A'),
            'codec': stream_audio.get('codec_name', 'N/A')
        }
        
        return specs
        
    except Exception as e:
        print(f"Erro ao detectar specs do áudio: {e}")
        return None


def calcular_ponto_corte(tempo_fim_anterior: str, tempo_inicio_proximo: str) -> float:
    """
    Calcula ponto médio entre fim de um segmento e início do próximo
    
    Args:
        tempo_fim_anterior: Timestamp fim do segmento anterior
        tempo_inicio_proximo: Timestamp início do próximo segmento
        
    Returns:
        Float com o ponto de corte em segundos
    """
    fim_seg = converter_timestamp_para_segundos(tempo_fim_anterior)
    inicio_seg = converter_timestamp_para_segundos(tempo_inicio_proximo)
    
    return (fim_seg + inicio_seg) / 2


# =============================================================================
# FUNÇÃO PRINCIPAL: CARREGAR E PROCESSAR CSV
# =============================================================================

def carregar_csv(caminho_csv: Path) -> list:
    """
    Carrega arquivo CSV com timestamps
    
    Args:
        caminho_csv: Path do arquivo CSV
        
    Returns:
        Lista de dicionários com os dados das linhas
    """
    linhas = []
    
    with open(caminho_csv, 'r', encoding='utf-8') as f:
        leitor = csv.DictReader(f, delimiter='|')
        for linha in leitor:
            linhas.append({
                'trecho': linha['Trecho'],
                'tempo_inicio': linha['tempo_inicio'],
                'tempo_fim': linha['tempo_fim'],
                'comeca_locutor': linha['comeca_locutor'].strip().lower() == 'true'
            })
    
    return linhas


def agrupar_segmentos(linhas: list, min_seg: float, max_seg: float, tolerancia: float = 0.8) -> list:
    """
    Agrupa linhas do CSV em segmentos respeitando limites e quebras de locutor
    
    Args:
        linhas: Lista de dicionários com dados do CSV
        min_seg: Duração mínima em segundos
        max_seg: Duração máxima em segundos
        tolerancia: Tolerância em segundos para min/max
        
    Returns:
        Lista de segmentos, cada um com: inicio, fim, texto, linhas_usadas
    """
    segmentos = []
    i = 0
    
    while i < len(linhas):
        # Inicia novo segmento (pode ser quebra de locutor ou continuação normal)
        inicio_seg = linhas[i]['tempo_inicio']
        texto_acumulado = []
        linhas_usadas = []
        j = i
        
        while j < len(linhas):
            # Verifica se esta linha é uma quebra de locutor E não é a primeira do segmento atual
            if j > i and linhas[j]['comeca_locutor']:
                # Encontrou nova quebra de locutor, para aqui
                break
            
            # Adiciona linha atual
            texto_acumulado.append(linhas[j]['trecho'])
            linhas_usadas.append(j)
            fim_seg = linhas[j]['tempo_fim']
            
            # Calcular duração acumulada
            duracao = converter_timestamp_para_segundos(fim_seg) - converter_timestamp_para_segundos(inicio_seg)
            
            # Verifica se próxima linha quebra locutor
            proxima_quebra = (j + 1 < len(linhas) and linhas[j + 1]['comeca_locutor'])
            
            # Decidir se finaliza segmento
            if duracao >= (min_seg - tolerancia):
                # Atingiu mínimo, para imediatamente
                break
            
            # Se não atingiu mínimo mas próxima é quebra, verifica se deve descartar ou aceitar
            if proxima_quebra:
                # Próxima linha é quebra de locutor
                # Se já temos algum conteúdo mas não atingiu MIN, aceita mesmo assim
                # (evita descartar segmentos isolados entre quebras)
                if duracao < (min_seg - tolerancia) and len(linhas_usadas) > 0:
                    # Aceita segmento menor que MIN se está isolado entre quebras
                    break
            
            # Verifica se adicionar próxima linha ultrapassaria MAX (se próxima não for quebra)
            if j + 1 < len(linhas) and not proxima_quebra:
                duracao_com_proxima = converter_timestamp_para_segundos(linhas[j + 1]['tempo_fim']) - converter_timestamp_para_segundos(inicio_seg)
                if duracao_com_proxima > (max_seg + tolerancia):
                    # Ultrapassaria max, finaliza aqui
                    break
            
            # Continua acumulando
            j += 1
            
            # Se chegou no final
            if j >= len(linhas):
                break
        
        # Calcular duração final
        duracao_final = converter_timestamp_para_segundos(fim_seg) - converter_timestamp_para_segundos(inicio_seg)
        
        # Adiciona segmento (agora aceita todos, mesmo < MIN se isolados)
        segmentos.append({
            'tempo_inicio': inicio_seg,
            'tempo_fim': fim_seg,
            'duracao': duracao_final,
            'texto': ' '.join(texto_acumulado),
            'linhas_usadas': linhas_usadas
        })
        
        # Avança para próxima linha não processada
        i = j + 1
    
    return segmentos


# =============================================================================
# FUNÇÃO: SEGMENTAR ÁUDIO COM FFMPEG
# =============================================================================

def segmentar_audio(caminho_audio: Path, segmentos: list, pasta_destino: Path, id_audio: str, formato: str, specs: dict):
    """
    Segmenta áudio usando ffmpeg baseado nos timestamps calculados
    
    Args:
        caminho_audio: Path do áudio original
        segmentos: Lista de segmentos com tempo_inicio e tempo_fim
        pasta_destino: Path da pasta onde salvar segmentos
        id_audio: ID do áudio (usado no nome dos arquivos)
        formato: Extensão do arquivo (mp3, flac, wav, etc)
        specs: Dicionário com especificações do áudio (codec, bitrate, sample_rate)
    """
    pasta_destino.mkdir(parents=True, exist_ok=True)
    
    for idx, seg in enumerate(segmentos, start=1):
        nome_segmento = f"{id_audio}_{idx:03d}.{formato}"
        caminho_segmento = pasta_destino / nome_segmento
        
        # Converter timestamps para segundos
        inicio_seg = converter_timestamp_para_segundos(seg['tempo_inicio'])
        fim_seg = converter_timestamp_para_segundos(seg['tempo_fim'])
        duracao_seg = fim_seg - inicio_seg
        
        # Construir comando ffmpeg com recodificação para corte físico
        cmd = [
            'ffmpeg',
            '-i', str(caminho_audio),
            '-ss', str(inicio_seg),
            '-t', str(duracao_seg),
        ]
        
        # Adicionar parâmetros de qualidade baseados no original
        if formato == 'flac':
            cmd.extend(['-c:a', 'flac'])
        elif formato == 'mp3':
            if specs['bitrate'] != 'N/A':
                cmd.extend(['-c:a', 'libmp3lame', '-b:a', specs['bitrate']])
            else:
                cmd.extend(['-c:a', 'libmp3lame', '-q:a', '0'])
        elif formato == 'wav':
            cmd.extend(['-c:a', 'pcm_s16le'])
        else:
            # Para outros formatos, tenta manter codec original
            if specs['codec'] != 'N/A':
                cmd.extend(['-c:a', specs['codec']])
        
        # Sample rate
        if specs['sample_rate'] != 'N/A':
            cmd.extend(['-ar', specs['sample_rate']])
        
        # Sobrescrever se existir
        cmd.extend(['-y', str(caminho_segmento)])
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"✓ Segmento criado: {nome_segmento}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Erro ao criar segmento {nome_segmento}: {e.stderr.decode()}")


# =============================================================================
# FUNÇÃO: GERAR JSON DE TRACKING
# =============================================================================

def gerar_json_tracking(segmentos: list, pasta_destino: Path, id_audio: str, formato: str, tipo_legenda: str):
    """
    Gera arquivo JSON com metadados dos segmentos
    
    Args:
        segmentos: Lista de segmentos processados
        pasta_destino: Path da pasta de destino
        id_audio: ID do áudio
        formato: Extensão do arquivo
        tipo_legenda: "auto" ou "manual"
    """
    tracking = {}
    
    for idx, seg in enumerate(segmentos, start=1):
        nome_segmento = f"{id_audio}_{idx:03d}.{formato}"
        
        tracking[nome_segmento] = {
            'tempo_inicio': seg['tempo_inicio'],
            'tempo_fim': seg['tempo_fim'],
            'duracao': round(seg['duracao'], 2),
            'texto': seg['texto'],
            'legenda': tipo_legenda
        }
    
    # Salvar JSON
    nome_json = f"{id_audio}_segments_originais.json"
    caminho_json = pasta_destino / nome_json
    
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON de tracking criado: {nome_json}")


# =============================================================================
# FUNÇÃO PRINCIPAL: EXECUTAR SEGMENTAÇÃO COMPLETA
# =============================================================================

def executar_segmentacao():
    """
    Executa fluxo completo de segmentação de áudio
    """
    # Configurações
    pasta_origem = Path(PASTA_ORIGEM)
    pasta_destino = Path(PASTA_DESTINO)
    min_seg = SEGMENTADOR_AUDIO['min_seg']
    max_seg = SEGMENTADOR_AUDIO['max_seg']
    
    # Localizar arquivos CSV e áudio
    arquivos_csv = list(pasta_origem.glob("*.csv"))
    
    if not arquivos_csv:
        print("✗ Nenhum arquivo CSV encontrado na pasta de origem")
        return
    
    # Processar cada CSV (assumindo 1 CSV por pasta)
    for csv_path in arquivos_csv:
        print(f"\n{'='*70}")
        print(f"Processando: {csv_path.name}")
        print(f"{'='*70}")
        
        # Extrair ID e tipo de legenda do nome do CSV
        nome_csv = csv_path.stem
        tipo_legenda = nome_csv.split('_')[0]  # "auto" ou "manual"
        id_audio = nome_csv[-11:]  # Últimos 11 caracteres
        
        # Localizar áudio correspondente (único áudio na pasta)
        arquivos_audio = [f for f in pasta_origem.iterdir() if f.suffix in ['.mp3', '.flac', '.wav', '.m4a', '.ogg']]
        
        if not arquivos_audio:
            print(f"✗ Nenhum arquivo de áudio encontrado para {csv_path.name}")
            continue
        
        audio_path = arquivos_audio[0]
        formato = audio_path.suffix[1:]
        
        print(f"Áudio: {audio_path.name}")
        print(f"ID: {id_audio} | Formato: {formato} | Legenda: {tipo_legenda}")
        
        # Detectar especificações do áudio
        specs = detectar_specs_audio(audio_path)
        if specs:
            print(f"Specs: {specs['codec']} | {specs['bitrate']} bps | {specs['sample_rate']} Hz")
        
        # Carregar CSV
        linhas = carregar_csv(csv_path)
        print(f"Linhas no CSV: {len(linhas)}")
        
        # Agrupar em segmentos
        segmentos = agrupar_segmentos(linhas, min_seg, max_seg)
        print(f"Segmentos criados: {len(segmentos)}")
        
        # Filtrar segmentos que não atendem MIN
        segmentos_validos = [
            seg for seg in segmentos 
            if seg['duracao'] >= (min_seg - 0.8)  # Tolerância de 0.8s
        ]
        
        segmentos_descartados = len(segmentos) - len(segmentos_validos)
        if segmentos_descartados > 0:
            print(f"Segmentos descartados (< MIN): {segmentos_descartados}")
        
        print(f"Segmentos válidos: {len(segmentos_validos)}")
        
        # Segmentar áudio
        print("\nSegmentando áudio...")
        segmentar_audio(audio_path, segmentos_validos, pasta_destino, id_audio, formato, specs)
        
        # Gerar JSON de tracking
        print("\nGerando JSON de tracking...")
        gerar_json_tracking(segmentos_validos, pasta_destino, id_audio, formato, tipo_legenda)
        
        print(f"\n✓ Processamento concluído para {csv_path.name}")


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    executar_segmentacao()