"""
Módulo 04: Segmentador Inteligente de Áudio via VAD (Voice Activity Detection)
Segmenta áudios automaticamente baseado em detecção de voz e pausas naturais
"""

import json
import subprocess
import tempfile
from pathlib import Path
import sys
import torch
import torchaudio

# Importar configurações do projeto
sys.path.append(str(Path(__file__).parent.parent))
from config import SEGMENTADOR_AUDIO_VAD, PROJECT_ROOT

id_video = 'CKidrRu_OEM'

# =============================================================================
# VARIÁVEIS DE TESTE (hardcoded para desenvolvimento)
# =============================================================================

PASTA_ORIGEM = "arquivos/temp/"+id_video+"/01-arquivos_originais"
PASTA_DESTINO = "arquivos/temp/"+id_video+"/02-segmentos_originais"


# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Carregar configurações do VAD
CFG = SEGMENTADOR_AUDIO_VAD

# Sample rate fixo para VAD (Silero-VAD otimizado para 16kHz)
VAD_SAMPLE_RATE = 16000


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

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
        Dict com: formato, bitrate, sample_rate, codec
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
        
        specs = {
            'formato': caminho_audio.suffix[1:],
            'bitrate': stream_audio.get('bit_rate', 'N/A'),
            'sample_rate': stream_audio.get('sample_rate', 'N/A'),
            'codec': stream_audio.get('codec_name', 'N/A'),
            'duracao': float(dados['format'].get('duration', 0))
        }
        
        return specs
        
    except Exception as e:
        print(f"Erro ao detectar specs do áudio: {e}")
        return None


def converter_para_16khz(caminho_audio: Path) -> Path:
    """
    Converte áudio para 16kHz WAV temporário para processamento VAD
    
    Args:
        caminho_audio: Path do áudio original
        
    Returns:
        Path do arquivo temporário 16kHz
    """
    # Criar arquivo temporário
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()
    
    # Converter para 16kHz mono WAV
    cmd = [
        'ffmpeg',
        '-i', str(caminho_audio),
        '-ar', str(VAD_SAMPLE_RATE),
        '-ac', '1',  # Mono
        '-y',
        str(temp_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        print(f"Audio temporário 16kHz criado para VAD")
        return temp_path
    except subprocess.CalledProcessError as e:
        print(f"Erro ao converter áudio para 16kHz: {e.stderr.decode()}")
        return None


def carregar_modelo_vad():
    """
    Carrega modelo Silero-VAD
    
    Returns:
        Tuple (model, utils)
    """
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        print("Modelo Silero-VAD carregado com sucesso")
        return model, utils
        
    except Exception as e:
        print(f"Erro ao carregar modelo VAD: {e}")
        return None, None


def detectar_fala_vad(caminho_audio: Path, model, utils) -> list:
    """
    Detecta segmentos de fala usando Silero-VAD
    
    Args:
        caminho_audio: Path do áudio em 16kHz
        model: Modelo VAD carregado
        utils: Utilitários do VAD
        
    Returns:
        Lista de dicionários com timestamps de fala: [{'start': float, 'end': float}]
    """
    # Extrair funções dos utils
    (get_speech_timestamps, _, read_audio, _, _) = utils
    
    # Carregar áudio
    wav = read_audio(str(caminho_audio), sampling_rate=VAD_SAMPLE_RATE)
    
    # Detectar fala com configurações do config
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=CFG['deteccao']['voice_threshold'],
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=CFG['criterios']['min_speech_duration_ms'],
        min_silence_duration_ms=CFG['criterios']['min_silence_duration_ms'],
        window_size_samples=int(CFG['deteccao']['window_size_seconds'] * VAD_SAMPLE_RATE),
        speech_pad_ms=CFG['padding']['inicio_ms']  # Padding será aplicado depois
    )
    
    # Converter de samples para segundos
    segmentos_fala = []
    for ts in speech_timestamps:
        start_sec = ts['start'] / VAD_SAMPLE_RATE
        end_sec = ts['end'] / VAD_SAMPLE_RATE
        
        segmentos_fala.append({
            'start': start_sec,
            'end': end_sec
        })
    
    print(f"VAD detectou {len(segmentos_fala)} segmentos de fala brutos")
    return segmentos_fala


def agrupar_segmentos_vad(segmentos_fala: list, duracao_total: float) -> list:
    """
    Agrupa segmentos de fala respeitando limites de duração e pausas
    
    Args:
        segmentos_fala: Lista de segmentos detectados pelo VAD
        duracao_total: Duração total do áudio em segundos
        
    Returns:
        Lista de segmentos agrupados com tempo_inicio e tempo_fim
    """
    if not segmentos_fala:
        print("Nenhum segmento de fala detectado")
        return []
    
    # Configurações
    min_seg = CFG['segmentos']['min_seg']
    max_seg = CFG['segmentos']['max_seg']
    tolerancia = CFG['segmentos']['tolerancia']
    min_silence_split = CFG['criterios']['min_silence_for_split']
    padding_inicio = CFG['padding']['inicio_ms'] / 1000.0  # Converter para segundos
    padding_fim = CFG['padding']['fim_ms'] / 1000.0
    
    segmentos_finais = []
    i = 0
    
    while i < len(segmentos_fala):
        # Iniciar novo segmento
        inicio_grupo = max(0, segmentos_fala[i]['start'] - padding_inicio)
        fim_grupo = segmentos_fala[i]['end'] + padding_fim
        j = i
        
        # Tentar acumular segmentos
        while j < len(segmentos_fala) - 1:
            # Verificar pausa até próximo segmento
            pausa = segmentos_fala[j + 1]['start'] - segmentos_fala[j]['end']
            
            # Se pausa é grande demais, força quebra
            if pausa >= min_silence_split:
                break
            
            # Calcular duração se incluir próximo segmento
            fim_tentativo = segmentos_fala[j + 1]['end'] + padding_fim
            duracao_tentativa = fim_tentativo - inicio_grupo
            
            # Se ultrapassar max, para aqui
            if duracao_tentativa > (max_seg + tolerancia):
                break
            
            # Incluir próximo segmento
            fim_grupo = fim_tentativo
            j += 1
            
            # Verificar se atingiu mínimo
            duracao_atual = fim_grupo - inicio_grupo
            if duracao_atual >= (min_seg - tolerancia):
                # Atingiu mínimo, verificar se próxima pausa é grande
                if j < len(segmentos_fala) - 1:
                    proxima_pausa = segmentos_fala[j + 1]['start'] - segmentos_fala[j]['end']
                    if proxima_pausa >= min_silence_split:
                        # Pausa natural grande, finaliza aqui
                        break
        
        # Garantir que não ultrapassa duração total
        fim_grupo = min(fim_grupo, duracao_total)
        
        # Calcular duração final
        duracao_final = fim_grupo - inicio_grupo
        
        # Adicionar segmento se atende critério mínimo (com tolerância)
        if duracao_final >= (min_seg - tolerancia):
            segmentos_finais.append({
                'tempo_inicio': inicio_grupo,
                'tempo_fim': fim_grupo,
                'duracao': duracao_final
            })
        
        # Avançar para próximo segmento não processado
        i = j + 1
    
    print(f"Segmentos agrupados: {len(segmentos_finais)}")
    return segmentos_finais


def segmentar_audio(caminho_audio: Path, segmentos: list, pasta_destino: Path, id_audio: str, formato: str, specs: dict):
    """
    Segmenta áudio original usando ffmpeg baseado nos timestamps calculados
    
    Args:
        caminho_audio: Path do áudio original
        segmentos: Lista de segmentos com tempo_inicio e tempo_fim em segundos
        pasta_destino: Path da pasta onde salvar segmentos
        id_audio: ID do áudio
        formato: Extensão do arquivo
        specs: Especificações do áudio original
    """
    pasta_destino.mkdir(parents=True, exist_ok=True)
    
    for idx, seg in enumerate(segmentos, start=1):
        nome_segmento = f"{id_audio}_{idx:03d}.{formato}"
        caminho_segmento = pasta_destino / nome_segmento
        
        # Timestamps em segundos
        inicio_seg = seg['tempo_inicio']
        duracao_seg = seg['duracao']
        
        # Construir comando ffmpeg preservando qualidade original
        cmd = [
            'ffmpeg',
            '-i', str(caminho_audio),
            '-ss', str(inicio_seg),
            '-t', str(duracao_seg),
        ]
        
        # Preservar qualidade original baseado no formato
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
            # Manter codec original
            if specs['codec'] != 'N/A':
                cmd.extend(['-c:a', specs['codec']])
        
        # Preservar sample rate original
        if specs['sample_rate'] != 'N/A':
            cmd.extend(['-ar', specs['sample_rate']])
        
        cmd.extend(['-y', str(caminho_segmento)])
        
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"Segmento criado: {nome_segmento}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao criar segmento {nome_segmento}: {e.stderr.decode()}")


def gerar_json_tracking(segmentos: list, pasta_destino: Path, id_audio: str, formato: str):
    """
    Gera arquivo JSON com metadados dos segmentos
    
    Args:
        segmentos: Lista de segmentos processados
        pasta_destino: Path da pasta de destino
        id_audio: ID do áudio
        formato: Extensão do arquivo
    """
    tracking = {}
    
    for idx, seg in enumerate(segmentos, start=1):
        nome_segmento = f"{id_audio}_{idx:03d}.{formato}"
        
        tracking[nome_segmento] = {
            'tempo_inicio': segundos_para_timestamp(seg['tempo_inicio']),
            'tempo_fim': segundos_para_timestamp(seg['tempo_fim']),
            'duracao': round(seg['duracao'], 2),
            'texto': None,      # VAD não gera transcrição
            'legenda': None,    # VAD não usa legendas
            'vad': True         # Identifica segmentação por VAD
        }
    
    # Salvar JSON
    nome_json = f"{id_audio}_segments_originais.json"
    caminho_json = pasta_destino / nome_json
    
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, ensure_ascii=False, indent=2)
    
    print(f"JSON de tracking criado: {nome_json}")


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def executar_segmentacao_vad():
    """
    Executa fluxo completo de segmentação via VAD
    """
    # Configurações
    pasta_origem = Path(PASTA_ORIGEM)
    pasta_destino = Path(PASTA_DESTINO)
    
    print(f"\n{'='*70}")
    print(f"SEGMENTAÇÃO VIA VAD")
    print(f"{'='*70}")
    
    # Verificar se deve sobrescrever
    if not CFG['comportamento']['sobrescrever']:
        if pasta_destino.exists() and any(pasta_destino.iterdir()):
            print(f"Pasta de destino já contém arquivos. Pulando (sobrescrever=False)")
            return
    
    # Localizar áudio
    arquivos_audio = [
        f for f in pasta_origem.iterdir() 
        if f.suffix.lower() in ['.mp3', '.flac', '.wav', '.m4a', '.ogg', '.opus']
    ]
    
    if not arquivos_audio:
        print("Nenhum arquivo de áudio encontrado na pasta de origem")
        return
    
    audio_path = arquivos_audio[0]
    formato = audio_path.suffix[1:]
    id_audio = id_video  # Usa ID do vídeo definido no início
    
    print(f"Audio: {audio_path.name}")
    print(f"ID: {id_audio} | Formato: {formato}")
    
    # Detectar especificações do áudio original
    specs = detectar_specs_audio(audio_path)
    if not specs:
        print("Erro ao detectar especificações do áudio")
        return
    
    print(f"Specs originais: {specs['codec']} | {specs['bitrate']} bps | {specs['sample_rate']} Hz | {specs['duracao']:.2f}s")
    
    # Carregar modelo VAD
    print("\nCarregando modelo VAD...")
    model, utils = carregar_modelo_vad()
    if model is None:
        print("Erro ao carregar modelo VAD")
        return
    
    # Converter para 16kHz temporário
    print("\nConvertendo para 16kHz temporário...")
    audio_16khz = converter_para_16khz(audio_path)
    if audio_16khz is None:
        print("Erro ao converter áudio")
        return
    
    try:
        # Detectar fala com VAD
        print("\nDetectando fala com VAD...")
        segmentos_fala = detectar_fala_vad(audio_16khz, model, utils)
        
        if not segmentos_fala:
            print("Nenhum segmento de fala detectado")
            return
        
        # Agrupar segmentos
        print("\nAgrupando segmentos...")
        segmentos_finais = agrupar_segmentos_vad(segmentos_fala, specs['duracao'])
        
        if not segmentos_finais:
            print("Nenhum segmento válido após agrupamento")
            return
        
        print(f"Segmentos válidos: {len(segmentos_finais)}")
        
        # Estatísticas
        duracoes = [s['duracao'] for s in segmentos_finais]
        print(f"Duração média: {sum(duracoes)/len(duracoes):.2f}s")
        print(f"Duração mínima: {min(duracoes):.2f}s")
        print(f"Duração máxima: {max(duracoes):.2f}s")
        
        # Segmentar áudio original
        print("\nSegmentando áudio original...")
        segmentar_audio(audio_path, segmentos_finais, pasta_destino, id_audio, formato, specs)
        
        # Gerar JSON
        print("\nGerando JSON de tracking...")
        gerar_json_tracking(segmentos_finais, pasta_destino, id_audio, formato)
        
        print(f"\nProcessamento concluído com sucesso!")
        
    finally:
        # Limpar arquivo temporário
        if audio_16khz and audio_16khz.exists():
            audio_16khz.unlink()
            print(f"Arquivo temporário 16kHz removido")


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    executar_segmentacao_vad()