"""
Module 04: Smart Audio Segmenter via VAD (Voice Activity Detection)
Automatically segments audio based on voice detection and natural pauses

OPTIMIZATIONS APPLIED:
- [LINE 310] FFmpeg fast seeking: -ss BEFORE -i (~50% gain)
- [LINE 338-342] Optimized logging: progress every 50 segments
"""

import json
import subprocess
import tempfile
from pathlib import Path
import sys
import torch
import torchaudio

# Import project configuration (root) and sibling modules (src/)
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))
from config import SEGMENTADOR_AUDIO_VAD, PROJECT_ROOT, EXTENSOES_AUDIO
# Probing and bits->codec translation come from m02, single source. This
# module used to have its own copy of the probing, which did NOT read
# bits_por_amostra: through the pipeline the cut came out correct and
# through direct execution it came out at 16 bits, a divergence that only
# shows up to whoever is debugging. One function only, one result only.
from m02_diretorios import codec_pcm_para_bits, sondar_specs_origem


# =============================================================================
# TEST VARIABLES (hardcoded for development)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# CONFIGURATION
# =============================================================================

# Load VAD configuration
CFG = SEGMENTADOR_AUDIO_VAD

# Fixed sample rate for VAD (Silero-VAD optimized for 16kHz)
VAD_SAMPLE_RATE = 16000


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def segundos_para_timestamp(segundos: float) -> str:
    """
    Converts seconds to an HH:MM:SS.mmm timestamp

    Args:
        segundos: Float with the total number of seconds

    Returns:
        String in HH:MM:SS.mmm format
    """
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = segundos % 60
    
    return f"{horas:02d}:{minutos:02d}:{segs:06.3f}"


def converter_para_16khz(caminho_audio: Path) -> Path:
    """
    Converts audio to a temporary 16kHz WAV for VAD processing

    Args:
        caminho_audio: Path of the original audio file

    Returns:
        Path of the temporary 16kHz file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    # Convert to 16kHz mono WAV
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
    Loads the Silero-VAD model

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
    Detects speech segments using Silero-VAD

    Args:
        caminho_audio: Path of the audio at 16kHz
        model: Loaded VAD model
        utils: VAD utilities

    Returns:
        List of dictionaries with speech timestamps: [{'start': float, 'end': float}]
    """
    # Extract functions from utils
    (get_speech_timestamps, _, read_audio, _, _) = utils

    # Load audio
    wav = read_audio(str(caminho_audio), sampling_rate=VAD_SAMPLE_RATE)

    # Detect speech using config settings
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        threshold=CFG['deteccao']['voice_threshold'],
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=CFG['criterios']['min_speech_duration_ms'],
        min_silence_duration_ms=CFG['criterios']['min_silence_duration_ms'],
        window_size_samples=int(CFG['deteccao']['window_size_seconds'] * VAD_SAMPLE_RATE),
        speech_pad_ms=CFG['padding']['inicio_ms']  # Padding will be applied later
    )

    # Convert from samples to seconds
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
    Groups speech segments respecting duration limits and pauses

    Args:
        segmentos_fala: List of segments detected by VAD
        duracao_total: Total audio duration in seconds

    Returns:
        List of grouped segments with tempo_inicio and tempo_fim
    """
    if not segmentos_fala:
        print("Nenhum segmento de fala detectado")
        return []
    
    # Configuration
    min_seg = CFG['segmentos']['min_seg']
    max_seg = CFG['segmentos']['max_seg']
    tolerancia = CFG['segmentos']['tolerancia']
    min_silence_split = CFG['criterios']['min_silence_for_split']
    padding_inicio = CFG['padding']['inicio_ms'] / 1000.0  # Convert to seconds
    padding_fim = CFG['padding']['fim_ms'] / 1000.0

    segmentos_finais = []
    i = 0

    while i < len(segmentos_fala):
        # Start new segment
        inicio_grupo = max(0, segmentos_fala[i]['start'] - padding_inicio)
        fim_grupo = segmentos_fala[i]['end'] + padding_fim
        j = i

        # Try to accumulate segments
        while j < len(segmentos_fala) - 1:
            # Check pause until the next segment
            pausa = segmentos_fala[j + 1]['start'] - segmentos_fala[j]['end']

            # If the pause is too long, force a break
            if pausa >= min_silence_split:
                # Checked whether it already reached the minimum
                duracao_atual = fim_grupo - inicio_grupo
                if duracao_atual >= (min_seg - tolerancia):
                    break

            # Calculate duration if the next segment is included
            fim_tentativo = segmentos_fala[j + 1]['end'] + padding_fim
            duracao_tentativa = fim_tentativo - inicio_grupo

            # If it exceeds the maximum, stop
            if duracao_tentativa > max_seg:
                break

            # Include next segment
            fim_grupo = fim_tentativo
            j += 1

            # Check whether it reached the minimum
            duracao_atual = fim_grupo - inicio_grupo
            if duracao_atual >= (min_seg - tolerancia):
                # Reached the minimum, check whether the next pause is long
                if j < len(segmentos_fala) - 1:
                    proxima_pausa = segmentos_fala[j + 1]['start'] - segmentos_fala[j]['end']
                    if proxima_pausa >= min_silence_split:
                        # Long natural pause, finish here
                        break

        # Ensure it does not exceed the total duration
        fim_grupo = min(fim_grupo, duracao_total)

        # Calculate final duration
        duracao_final = fim_grupo - inicio_grupo

        # Add segment if it meets the minimum criterion (with tolerance)
        if duracao_final >= (min_seg - tolerancia):
            segmentos_finais.append({
                'tempo_inicio': inicio_grupo,
                'tempo_fim': fim_grupo,
                'duracao': duracao_final
            })

        # Advance to the next unprocessed segment
        i = j + 1
    
    print(f"Segmentos agrupados: {len(segmentos_finais)}")
    return segmentos_finais


def segmentar_audio(caminho_audio: Path, segmentos: list, pasta_destino: Path, id_audio: str, formato: str, specs: dict):
    """
    Segments the original audio with ffmpeg, at source quality.

    The segment comes out with the SAME bit depth, rate and channel
    count as the original: 02-segmentos_originais is the quality ceiling
    for everything that comes after, and the folder name needs to match
    the content.

    Args:
        caminho_audio: Path of the original audio file
        segmentos: List of segments with tempo_inicio and tempo_fim in seconds
        pasta_destino: Path of the folder to save segments in
        id_audio: Audio ID
        formato: File extension (makes up the segment's name)
        specs: SOURCE audio specs, which the bit depth and sample rate come from
    """
    pasta_destino.mkdir(parents=True, exist_ok=True)
    
    for idx, seg in enumerate(segmentos, start=1):
        nome_segmento = f"{id_audio}_{idx:03d}.{formato}"
        caminho_segmento = pasta_destino / nome_segmento
        
        # Timestamps in seconds
        inicio_seg = seg['tempo_inicio']
        duracao_seg = seg['duracao']

        # OPTIMIZATION: -ss BEFORE -i for fast seeking (~50% gain)
        # Fast seeking looks for the nearest keyframe before the timestamp
        # This is safe because VAD already adds padding at the pauses
        cmd = [
            'ffmpeg',
            '-ss', str(inicio_seg),
            '-i', str(caminho_audio),
            '-t', str(duracao_seg),
        ]

        # SOURCE bit depth, by the same rule m02 used to write
        # 01-arquivos_originais. A fixed constant here would truncate a
        # 24-bit original WITHOUT WARNING, and no later step recovers from it.
        cmd.extend(['-c:a', codec_pcm_para_bits(specs)])

        # Preserve original sample rate
        if specs['sample_rate'] is not None:
            cmd.extend(['-ar', specs['sample_rate']])

        # No '-ac': FFmpeg copies the input's channel count

        cmd.extend(['-y', str(caminho_segmento)])

        try:
            subprocess.run(cmd, capture_output=True, check=True)

            # OPTIMIZATION: Progressive logging (reduces console I/O)
            # Log only first, multiples of 50, and last segment
            if idx == 1 or idx % 50 == 0 or idx == len(segmentos):
                print(f"Progresso segmentacao: {idx}/{len(segmentos)} segmentos")
                
        except subprocess.CalledProcessError as e:
            print(f"ERRO ao criar segmento {nome_segmento}: {e.stderr.decode()}")


def gerar_json_tracking(segmentos: list, pasta_destino: Path, id_audio: str, formato: str, specs: dict):
    """
    Generates a JSON file with segment metadata

    Args:
        segmentos: List of processed segments
        pasta_destino: Path of the destination folder
        id_audio: Audio ID
        formato: File extension
        specs: Source-audio specs (already in memory, no extra read)
    """
    tracking = {}

    for idx, seg in enumerate(segmentos, start=1):
        nome_segmento = f"{id_audio}_{idx:03d}.{formato}"

        tracking[nome_segmento] = {
            'tempo_inicio': segundos_para_timestamp(seg['tempo_inicio']),
            'tempo_fim': segundos_para_timestamp(seg['tempo_fim']),
            'duracao': round(seg['duracao'], 2),
            'vad': True,        # Identifies VAD segmentation
            # Provenance of the SOURCE file (before segmentation and
            # normalization) - defines the segment's quality ceiling
            'origem_codec': specs['codec'],
            'origem_bitrate': specs['bitrate'],
            'origem_sample_rate': specs['sample_rate']
        }

    # Save JSON
    nome_json = f"{id_audio}_segments_originais.json"
    caminho_json = pasta_destino / nome_json
    
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(tracking, f, ensure_ascii=False, indent=2)
    
    print(f"JSON de tracking criado: {nome_json}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def executar_segmentacao_vad(audio_id: str, specs_origem: dict) -> bool:
    """
    Runs the full VAD segmentation flow.

    Args:
        audio_id: ID of the audio file to process
        specs_origem: SOURCE file specs, probed by m02 before the
            conversion to WAV. Probing here would return the internal
            WAV's specs, and the audio's provenance would be lost.

    Returns:
        True if valid segments were found/created, False otherwise
    """
    # Configuration
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    pasta_origem = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "01-arquivos_originais"
    pasta_destino = PROJECT_ROOT / "arquivos" / "temp" / audio_id / "02-segmentos_originais"
    print(f"\n{'='*70}")
    print(f"SEGMENTACAO VIA VAD")
    print(f"{'='*70}")

    # Check whether it should overwrite
    if not CFG['comportamento']['sobrescrever']:
        if pasta_destino.exists() and any(pasta_destino.iterdir()):
            print(f"Pasta de destino ja contem arquivos. Pulando (sobrescrever=False)")
            return True

    # Locate audio
    arquivos_audio = [
        f for f in pasta_origem.iterdir()
        if f.suffix.lower() in EXTENSOES_AUDIO
    ]

    if not arquivos_audio:
        print("Nenhum arquivo de audio encontrado na pasta de origem")
        return False

    audio_path = arquivos_audio[0]
    formato = audio_path.suffix[1:]
    id_audio = audio_id

    print(f"Audio: {audio_path.name}")
    print(f"ID: {id_audio} | Formato: {formato}")

    # SOURCE specs, already in memory (probed by m02 before the conversion)
    print(f"Specs da fonte: {specs_origem['codec'] or 'ausente'} | "
          f"{specs_origem['bitrate'] or 'ausente'} bps | "
          f"{specs_origem['sample_rate'] or 'ausente'} Hz | "
          f"{specs_origem['duracao']:.2f}s")

    # Load VAD model
    print("\nCarregando modelo VAD...")
    model, utils = carregar_modelo_vad()
    if model is None:
        print("Erro ao carregar modelo VAD")
        return False

    # Convert to temporary 16kHz
    print("\nConvertendo para 16kHz temporário...")
    audio_16khz = converter_para_16khz(audio_path)
    if audio_16khz is None:
        print("Erro ao converter áudio")
        return False

    resultado = False
    try:
        # Detect speech with VAD
        print("\nDetectando fala com VAD...")
        segmentos_fala = detectar_fala_vad(audio_16khz, model, utils)

        if not segmentos_fala:
            print("Nenhum segmento de fala detectado")
            return False

        # Group segments
        print("\nAgrupando segmentos...")
        segmentos_finais = agrupar_segmentos_vad(segmentos_fala, specs_origem['duracao'])

        if not segmentos_finais:
            print("Nenhum segmento válido após agrupamento")
            return False

        print(f"Segmentos válidos: {len(segmentos_finais)}")

        # Statistics
        duracoes = [s['duracao'] for s in segmentos_finais]
        print(f"Duração média: {sum(duracoes)/len(duracoes):.2f}s")
        print(f"Duração mínima: {min(duracoes):.2f}s")
        print(f"Duração máxima: {max(duracoes):.2f}s")

        # Segment original audio
        print(f"\nSegmentando áudio original ({len(segmentos_finais)} segmentos)...")
        segmentar_audio(audio_path, segmentos_finais, pasta_destino, id_audio, formato, specs_origem)

        # Generate JSON
        print("\nGerando JSON de tracking...")
        gerar_json_tracking(segmentos_finais, pasta_destino, id_audio, formato, specs_origem)

        print(f"\nProcessamento concluído com sucesso!")
        resultado = True

    finally:
        # Clean up temporary file
        if audio_16khz and audio_16khz.exists():
            audio_16khz.unlink()
            print(f"Arquivo temporário 16kHz removido")

    return resultado


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Direct execution requires audio_id as an argument - no fixed id in the code
    if len(sys.argv) != 2:
        print("Uso: python src/m04_segmentador_audio_vad.py <audio_id>")
        sys.exit(1)

    audio_id_cli = sys.argv[1]

    # In the pipeline the specs come from m02. Outside it, the SOURCE file
    # is probed - never the WAV in 01-arquivos_originais, which has
    # already lost the source format
    pasta_fonte_cli = PROJECT_ROOT / "arquivos" / "audios" / audio_id_cli
    fontes_cli = sorted(
        f for f in pasta_fonte_cli.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSOES_AUDIO
    ) if pasta_fonte_cli.is_dir() else []

    if not fontes_cli:
        print(f"Nenhum audio-fonte encontrado em {pasta_fonte_cli}")
        sys.exit(1)

    specs_cli = sondar_specs_origem(fontes_cli[0])
    if not specs_cli:
        sys.exit(1)

    if not executar_segmentacao_vad(audio_id_cli, specs_cli):
        sys.exit(1)