#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo 00 - Downloader de Áudios do YouTube
Realiza download de áudios e legendas em português do YouTube
"""

import sys
import csv
import time
import random
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Adiciona o diretório raiz ao path para importar config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DOWNLOADER, PROJECT_ROOT

try:
    import yt_dlp
except ImportError:
    print("ERRO: yt-dlp não está instalado!")
    print("Instale com: pip install yt-dlp")
    sys.exit(1)


# ============================================================================
# CONFIGURAÇÃO DE LOGGING
# ============================================================================

def configurar_logging() -> None:
    """
    Configura sistema de logging que sobrescreve a cada execução
    """
    pasta_logs = PROJECT_ROOT / 'arquivos' / 'links_download'
    pasta_logs.mkdir(parents=True, exist_ok=True)
    
    arquivo_log = pasta_logs / 'output.log'
    
    # Remove handlers existentes
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configura novo logging (modo 'w' sobrescreve)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(arquivo_log, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info("="*70)
    logging.info("INICIANDO MÓDULO DOWNLOADER")
    logging.info("="*70)


# ============================================================================
# VALIDAÇÕES INICIAIS
# ============================================================================

def validar_ambiente() -> bool:
    """
    Valida se ambiente está preparado para execução
    Retorna True se tudo OK, False caso contrário
    """
    logging.info("Validando ambiente de execução...")
    
    # Verifica estrutura de pastas
    pasta_links = PROJECT_ROOT / 'arquivos' / 'links_download'
    pasta_audios = PROJECT_ROOT / 'arquivos' / 'audios'
    
    pasta_links.mkdir(parents=True, exist_ok=True)
    pasta_audios.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Pasta de links: {pasta_links}")
    logging.info(f"Pasta de áudios: {pasta_audios}")
    
    # Verifica se CSV existe
    csv_path = pasta_links / 'links.csv'
    if not csv_path.exists():
        logging.error(f"Arquivo CSV não encontrado: {csv_path}")
        logging.error("Crie o arquivo com header 'url_link' e adicione os links")
        return False
    
    # Valida estrutura do CSV
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if 'url_link' not in reader.fieldnames:
                logging.error("CSV deve ter coluna 'url_link' no header")
                return False
    except Exception as e:
        logging.error(f"Erro ao ler CSV: {e}")
        return False
    
    # Cria arquivos de log se não existirem
    log_sucesso = pasta_links / 'download_sucesso.txt'
    log_rejeitado = pasta_links / 'download_rejeitado.txt'
    
    if not log_sucesso.exists():
        log_sucesso.write_text('url_link_sucesso\n', encoding='utf-8')
        logging.info("Criado arquivo de log: download_sucesso.txt")
    
    if not log_rejeitado.exists():
        log_rejeitado.write_text('url_link_rejeitado\n', encoding='utf-8')
        logging.info("Criado arquivo de log: download_rejeitado.txt")
    
    logging.info("Ambiente validado com sucesso!")
    return True


# ============================================================================
# MANIPULAÇÃO DE CSV
# ============================================================================

def ler_links_csv() -> List[str]:
    """
    Lê todos os links do arquivo CSV
    Retorna lista de URLs
    """
    csv_path = PROJECT_ROOT / 'arquivos' / 'links_download' / 'links.csv'
    links = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                url = row.get('url_link', '').strip()
                if url:
                    links.append(url)
    except Exception as e:
        logging.error(f"Erro ao ler CSV: {e}")
    
    return links


def remover_linha_csv(url_removida: str) -> None:
    """
    Remove uma linha específica do CSV e sobrescreve o arquivo
    """
    csv_path = PROJECT_ROOT / 'arquivos' / 'links_download' / 'links.csv'
    
    try:
        # Lê todas as linhas
        linhas = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['url_link'].strip() != url_removida:
                    linhas.append(row)
        
        # Sobrescreve arquivo
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(linhas)
        
        logging.debug(f"Linha removida do CSV: {url_removida}")
    
    except Exception as e:
        logging.error(f"Erro ao remover linha do CSV: {e}")


# ============================================================================
# LOGS DE SUCESSO E REJEIÇÃO
# ============================================================================

def registrar_sucesso(url: str) -> None:
    """
    Registra URL no arquivo de downloads bem-sucedidos
    """
    log_path = PROJECT_ROOT / 'arquivos' / 'links_download' / 'download_sucesso.txt'
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"{url}\n")
        logging.info(f"Registrado em sucesso: {url}")
    except Exception as e:
        logging.error(f"Erro ao registrar sucesso: {e}")


def registrar_rejeitado(url: str, motivo: str = "") -> None:
    """
    Registra URL no arquivo de downloads rejeitados
    """
    log_path = PROJECT_ROOT / 'arquivos' / 'links_download' / 'download_rejeitado.txt'
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            linha = f"{url}" + (f" | {motivo}" if motivo else "")
            f.write(f"{linha}\n")
        logging.warning(f"Registrado em rejeitado: {url} | {motivo}")
    except Exception as e:
        logging.error(f"Erro ao registrar rejeitado: {e}")


# ============================================================================
# EXTRAÇÃO DE INFORMAÇÕES DO VÍDEO
# ============================================================================

def extrair_info_video(url: str) -> Optional[Dict]:
    """
    Extrai informações do vídeo sem baixar
    Retorna dicionário com metadados ou None em caso de erro
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info
    except Exception as e:
        logging.error(f"Erro ao extrair informações de {url}: {e}")
        return None


def identificar_tipo_url(info: Dict) -> str:
    """
    Identifica se é vídeo individual, playlist ou canal
    Retorna: 'video', 'playlist' ou 'channel'
    """
    if info.get('_type') == 'playlist':
        return 'playlist'
    elif 'entries' in info:
        return 'channel'
    else:
        return 'video'


def filtrar_por_duracao(duracao_segundos: int) -> bool:
    """
    Verifica se vídeo está dentro dos limites de duração configurados
    Retorna True se está dentro dos limites, False caso contrário
    """
    min_dur = DOWNLOADER['duracao']['minima_segundos']
    max_dur = DOWNLOADER['duracao']['maxima_segundos']
    
    if duracao_segundos < min_dur:
        logging.warning(f"Vídeo muito curto: {duracao_segundos}s (mínimo: {min_dur}s)")
        return False
    
    if duracao_segundos > max_dur:
        logging.warning(f"Vídeo muito longo: {duracao_segundos}s (máximo: {max_dur}s)")
        return False
    
    return True


# ============================================================================
# DOWNLOAD DE LEGENDAS
# ============================================================================

def verificar_e_baixar_legendas(video_id: str, info: Dict, pasta_output: Path) -> Tuple[bool, str]:
    """
    Verifica disponibilidade e baixa legendas em português
    Retorna (sucesso: bool, tipo_legenda: str)
    tipo_legenda pode ser: 'manual', 'auto' ou ''
    """
    idiomas = DOWNLOADER['legendas']['idiomas']
    aceitar_auto = DOWNLOADER['legendas']['aceitar_automaticas']
    
    subtitles = info.get('subtitles', {})
    auto_captions = info.get('automatic_captions', {})
    
    logging.info(f"Verificando legendas para vídeo {video_id}...")
    
    # Tenta legendas manuais
    for idioma in idiomas:
        if idioma in subtitles:
            logging.info(f"Legenda manual encontrada: {idioma}")
            if baixar_legenda(video_id, info['webpage_url'], idioma, False, pasta_output):
                return True, 'manual'
    
    # Tenta legendas automáticas se configurado
    if aceitar_auto:
        for idioma in idiomas:
            if idioma in auto_captions:
                logging.info(f"Legenda automática encontrada: {idioma}")
                if baixar_legenda(video_id, info['webpage_url'], idioma, True, pasta_output):
                    return True, 'auto'
    
    logging.warning(f"Nenhuma legenda em português encontrada para {video_id}")
    return False, ''


def baixar_legenda(video_id: str, url: str, idioma: str, is_auto: bool, pasta_output: Path) -> bool:
    """
    Baixa legenda específica
    Retorna True se sucesso, False caso contrário
    """
    tipo = 'auto' if is_auto else 'manual'
    arquivo_legenda = pasta_output / f"{tipo}_{video_id}.txt"
    
    ydl_opts = {
        'skip_download': True,
        'writesubtitles': True,
        'writeautomaticsub': is_auto,
        'subtitleslangs': [idioma],
        'subtitlesformat': 'vtt/srv3/srt',
        'outtmpl': str(pasta_output / video_id),
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Procura arquivo de legenda em qualquer formato (srv3, vtt, srt)
        formatos_possiveis = ['srv3', 'vtt', 'srt']
        for extensao in formatos_possiveis:
            arquivo_legenda_temp = pasta_output / f"{video_id}.{idioma}.{extensao}"
            if arquivo_legenda_temp.exists():
                arquivo_legenda_temp.rename(arquivo_legenda)
                logging.info(f"Legenda salva ({extensao}): {arquivo_legenda.name}")
                return True
        
        # Se não encontrou nenhum formato
        logging.warning(f"Arquivo de legenda não encontrado em nenhum formato: {formatos_possiveis}")
        return False
    
    except Exception as e:
        logging.error(f"Erro ao baixar legenda: {e}")
        return False


# ============================================================================
# DOWNLOAD DE ÁUDIO
# ============================================================================

def baixar_audio(video_id: str, url: str, pasta_output: Path) -> bool:
    """
    Baixa áudio do vídeo conforme configurações
    Retorna True se sucesso, False caso contrário
    """
    formato = DOWNLOADER['audio']['formato']
    bitrate = DOWNLOADER['audio']['bitrate_kbps']
    sample_rate = DOWNLOADER['audio']['sample_rate_hz']
    
    arquivo_audio = pasta_output / f"{video_id}.{formato}"
    
    # Configurações base do yt-dlp
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(pasta_output / video_id),
        'quiet': False,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': formato,
        }],
    }
    
    # Adiciona bitrate se configurado
    if bitrate > 0:
        ydl_opts['postprocessors'][0]['preferredquality'] = str(bitrate)
    
    # Adiciona sample rate se configurado
    if sample_rate > 0:
        ydl_opts['postprocessor_args'] = ['-ar', str(sample_rate)]
    
    try:
        logging.info(f"Iniciando download de áudio: {video_id}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        if arquivo_audio.exists():
            logging.info(f"Áudio baixado com sucesso: {arquivo_audio.name}")
            return True
        else:
            logging.error(f"Arquivo de áudio não foi criado: {arquivo_audio}")
            return False
    
    except Exception as e:
        logging.error(f"Erro ao baixar áudio: {e}")
        return False


# ============================================================================
# PROCESSAMENTO DE VÍDEOS
# ============================================================================

def processar_video(video_info: Dict, url_original: str) -> bool:
    """
    Processa um vídeo individual: valida, baixa legenda e áudio
    Retorna True se sucesso, False caso contrário
    """
    video_id = video_info.get('id')
    titulo = video_info.get('title', 'Sem título')
    duracao = video_info.get('duration', 0)
    
    logging.info("-" * 70)
    logging.info(f"Processando: {titulo}")
    logging.info(f"ID: {video_id} | Duração: {duracao}s")
    
    # Define pasta de output
    pasta_output = PROJECT_ROOT / 'arquivos' / 'audios' / video_id
    
    # Verifica se já existe (pula se configurado)
    if pasta_output.exists() and not DOWNLOADER['comportamento']['sobrescrever']:
        logging.info(f"Pasta {video_id} já existe. Pulando...")
        return True
    
    # Cria pasta de output
    pasta_output.mkdir(parents=True, exist_ok=True)
    
    try:
        # Valida duração
        if not filtrar_por_duracao(duracao):
            registrar_rejeitado(url_original, f"Duração fora dos limites: {duracao}s")
            shutil.rmtree(pasta_output, ignore_errors=True)
            return False
        
        # Baixa legendas (OBRIGATÓRIO)
        tem_legenda, tipo_legenda = verificar_e_baixar_legendas(video_id, video_info, pasta_output)
        
        if not tem_legenda:
            logging.warning(f"Vídeo rejeitado: sem legendas em português")
            registrar_rejeitado(url_original, "Sem legendas em português")
            shutil.rmtree(pasta_output, ignore_errors=True)
            return False
        
        # Baixa áudio
        if not baixar_audio(video_id, video_info['webpage_url'], pasta_output):
            logging.error(f"Falha ao baixar áudio")
            registrar_rejeitado(url_original, "Erro ao baixar áudio")
            shutil.rmtree(pasta_output, ignore_errors=True)
            return False
        
        # Sucesso!
        logging.info(f"Download concluído com sucesso: {video_id}")
        registrar_sucesso(url_original)
        return True
    
    except Exception as e:
        logging.error(f"Erro ao processar vídeo {video_id}: {e}")
        registrar_rejeitado(url_original, f"Erro: {str(e)}")
        shutil.rmtree(pasta_output, ignore_errors=True)
        return False


def processar_playlist_ou_canal(info: Dict, url_original: str, tipo: str) -> None:
    """
    Processa playlist ou canal (múltiplos vídeos)
    """
    entries = info.get('entries', [])
    titulo = info.get('title', 'Sem título')
    limit = DOWNLOADER['quantidade']['limit']
    
    # Aplica limite se configurado
    if limit > 0:
        entries = entries[:limit]
        logging.info(f"Aplicando limite de {limit} vídeos")
    
    total = len(entries)
    logging.info(f"Processando {tipo}: {titulo} ({total} vídeos)")
    
    for idx, entry in enumerate(entries, 1):
        logging.info(f"\n[{idx}/{total}] Processando vídeo da {tipo}...")
        
        # Reconstrói URL do vídeo a partir do ID
        video_id = entry.get('id')
        if not video_id:
            logging.error(f"Vídeo {idx} sem ID. Pulando...")
            continue
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        logging.debug(f"URL reconstruída: {video_url}")
        
        # Extrai informações completas do vídeo
        video_info = extrair_info_video(video_url)
        if not video_info:
            logging.error(f"Falha ao extrair informações do vídeo {idx}")
            continue
        
        # Processa vídeo
        processar_video(video_info, url_original)
        
        # Delay entre vídeos da playlist
        if idx < total:
            delay = random.uniform(
                DOWNLOADER['delays']['entre_videos_playlist']['minimo_segundos'],
                DOWNLOADER['delays']['entre_videos_playlist']['maximo_segundos']
            )
            logging.info(f"Aguardando {delay:.2f}s antes do próximo vídeo...")
            time.sleep(delay)


# ============================================================================
# LOOP PRINCIPAL
# ============================================================================

def executar_downloads() -> None:
    """
    Loop principal de processamento de URLs do CSV
    """
    links = ler_links_csv()
    total_links = len(links)
    
    if total_links == 0:
        logging.info("Nenhum link encontrado no CSV")
        return
    
    logging.info(f"Total de links a processar: {total_links}")
    logging.info("="*70)
    
    for idx, url in enumerate(links, 1):
        logging.info(f"\n{'='*70}")
        logging.info(f"LINK [{idx}/{total_links}]: {url}")
        logging.info(f"{'='*70}")
        
        # Remove linha do CSV imediatamente
        remover_linha_csv(url)
        
        try:
            # Extrai informações
            info = extrair_info_video(url)
            if not info:
                logging.error(f"Falha ao extrair informações do link")
                registrar_rejeitado(url, "Erro ao extrair informações")
                continue
            
            # Identifica tipo
            tipo = identificar_tipo_url(info)
            logging.info(f"Tipo identificado: {tipo}")
            
            # Processa conforme tipo
            if tipo == 'video':
                processar_video(info, url)
            else:
                processar_playlist_ou_canal(info, url, tipo)
            
            # Delay entre links do CSV
            if idx < total_links:
                delay = random.uniform(
                    DOWNLOADER['delays']['entre_links_csv']['minimo_segundos'],
                    DOWNLOADER['delays']['entre_links_csv']['maximo_segundos']
                )
                logging.info(f"\nAguardando {delay:.2f}s antes do próximo link do CSV...")
                time.sleep(delay)
        
        except Exception as e:
            logging.error(f"Erro crítico ao processar link: {e}")
            registrar_rejeitado(url, f"Erro crítico: {str(e)}")
            continue
    
    logging.info("\n" + "="*70)
    logging.info("PROCESSAMENTO CONCLUÍDO")
    logging.info("="*70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Função principal
    """
    # Configura logging
    configurar_logging()
    
    # Valida ambiente
    if not validar_ambiente():
        logging.error("Ambiente inválido. Abortando execução.")
        sys.exit(1)
    
    # Exibe configurações
    logging.info("\nConfigurações ativas:")
    logging.info(f"  Formato áudio: {DOWNLOADER['audio']['formato']}")
    logging.info(f"  Bitrate: {DOWNLOADER['audio']['bitrate_kbps']} kbps")
    logging.info(f"  Sample rate: {DOWNLOADER['audio']['sample_rate_hz']} Hz")
    logging.info(f"  Idiomas legenda: {DOWNLOADER['legendas']['idiomas']}")
    logging.info(f"  Aceitar automáticas: {DOWNLOADER['legendas']['aceitar_automaticas']}")
    logging.info(f"  Duração: {DOWNLOADER['duracao']['minima_segundos']}s - {DOWNLOADER['duracao']['maxima_segundos']}s")
    logging.info(f"  Limite por fonte: {DOWNLOADER['quantidade']['limit']}")
    logging.info("")
    
    # Executa downloads
    try:
        executar_downloads()
    except KeyboardInterrupt:
        logging.warning("\nProcesso interrompido pelo usuário")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Erro fatal: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()