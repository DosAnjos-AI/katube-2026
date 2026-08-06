import json
import sys
import subprocess
from pathlib import Path
from typing import Optional
import shutil

# Define PROJECT_ROOT in the global scope
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EXTENSOES_AUDIO


def sondar_specs_origem(origem: Path) -> Optional[dict]:
    """
    Probes the SOURCE file with ffprobe, once, before converting.

    After conversion the source format disappears from the pipeline:
    what circulates is the internal WAV. Probing here is the last chance
    to record where the audio came from - these specs travel all the way
    to the origem_* columns of dataset.csv.

    The same result decides the WAV's bit depth: FFmpeg's default is
    pcm_s16le, which would truncate a 24-bit original WITHOUT WARNING.

    Returns:
        Dict with format, codec, bitrate, sample_rate, duration and
        bits_per_sample, or None if ffprobe failed (the caller aborts).
    """
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        str(origem)
    ]

    try:
        resultado = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding='utf-8', check=True)
        dados = json.loads(resultado.stdout)
        stream = next(s for s in dados['streams'] if s['codec_type'] == 'audio')
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError, StopIteration) as e:
        stderr = getattr(e, 'stderr', '') or ''
        print(f"ERRO: ffprobe falhou em '{origem}': {e} {stderr.strip()}")
        return None

    # Absence is None, never the string 'N/A': codec, bitrate and sample_rate
    # travel all the way to the origem_* columns of dataset.csv, where
    # absence has to be an empty cell. 'N/A' would be TEXT in a numeric column.
    return {
        'formato': origem.suffix[1:],
        'codec': stream.get('codec_name'),
        'bitrate': stream.get('bit_rate'),
        'sample_rate': stream.get('sample_rate'),
        'duracao': float(dados['format'].get('duration', 0)),
        'bits_por_amostra': stream.get('bits_per_raw_sample'),
    }


def codec_pcm_para_bits(specs: dict) -> str:
    """
    Translates the SOURCE's bit depth into the corresponding PCM codec.

    Single decision point: it is used both here, in the WAV conversion,
    and by m04, when cutting the segments. Two copies of the rule would
    drift apart from each other, and the segment would end up with a
    different depth than the original it came from.

    Args:
        specs: source specs already probed (see sondar_specs_origem).

    Returns:
        PCM codec name for FFmpeg's '-c:a' parameter.
    """
    # Compressed formats (mp3, m4a, ogg, aac, wma) decode as floating point
    # and do not declare bits per sample - 16 bits is the correct target
    bits = str(specs.get('bits_por_amostra'))
    if bits == '24':
        return 'pcm_s24le'
    if bits == '32':
        return 'pcm_s32le'
    return 'pcm_s16le'


def converter_para_wav(origem: Path, destino: Path, specs: dict) -> bool:
    """
    Converts the audio to WAV preserving the original's parameters.

    No '-ar' and no '-ac': FFmpeg copies the source's sample rate and
    channel count. No resampling, neither up nor down.

    Args:
        specs: source specs already probed, which the bit depth comes
            from.

    Returns:
        True if the WAV was written, False otherwise (with a log entry).
    """
    codec = codec_pcm_para_bits(specs)

    cmd = [
        'ffmpeg',
        '-v', 'error',
        '-i', str(origem),
        '-c:a', codec,
        '-y',
        str(destino)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True,
                       encoding='utf-8', check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        stderr = getattr(e, 'stderr', '') or ''
        print(f"ERRO: ffmpeg falhou ao converter '{origem}': {e} {stderr.strip()}")
        return False

    if not destino.exists() or destino.stat().st_size == 0:
        print(f"ERRO: conversao nao produziu audio utilizavel: '{destino}'")
        return False

    print(f"Convertido para WAV ({codec}): {origem.name} -> {destino.name}")
    return True


def limpar_estado_anterior(audio_id: str):
    """
    Removes all state left by previous runs of this audio_id.

    Without this, whatever the new run does not rewrite survives and
    mixes with the new state: orphaned segments in temp/{id} and
    orphaned .flac files in the delivered dataset. Processing of an id
    always starts from scratch.
    """
    # Mandatory guard: an empty id, '.' or '..' collapses the path to the
    # root and rmtree would end up targeting the entire temp/ and
    # audio_dataset/. Fail loud.
    if not audio_id or audio_id in ('.', '..') or '/' in audio_id or '\\' in audio_id:
        raise ValueError(f"audio_id invalido para limpeza: {audio_id!r}")

    alvos = [
        PROJECT_ROOT / "arquivos" / "temp" / audio_id,
        PROJECT_ROOT / "dataset" / "audio_dataset" / audio_id,
    ]

    for alvo in alvos:
        if not alvo.exists():
            continue

        total_arquivos = sum(1 for item in alvo.rglob('*') if item.is_file())
        shutil.rmtree(alvo)
        print(f"Estado anterior removido: {alvo} ({total_arquivos} arquivo(s))")


def criar_diretorios(audio_id: str) -> Optional[dict]:
    """
    Prepares the audio_id's directory structure and converts the input.

    Returns:
        The SOURCE audio's specs (see sondar_specs_origem), which the
        caller passes on to m04 for the dataset's provenance, or None if
        something failed (missing source folder, no audio, ffprobe or
        ffmpeg).
    """
    #============================================================
    # Clean restart: nothing from a previous run survives
    #============================================================
    limpar_estado_anterior(audio_id)

    #============================================================
    # Creating the audio's general folder where all the subfolders will live
    #============================================================
    pasta = PROJECT_ROOT / "arquivos" / "temp" / audio_id
    pasta.mkdir(parents=True, exist_ok=True)

    #============================================================
    # Creating subfolders for intermediate files
    #============================================================
    # create folder for the dynamic .json files
    pasta1 = pasta / '00-json_dinamico'
    pasta1.mkdir(parents=True, exist_ok=True)

    # create folder with the copies of the original files
    pasta1 = pasta / '01-arquivos_originais'
    pasta1.mkdir(parents=True, exist_ok=True)

    # create folder with the segments at the original sr
    pasta2 = pasta / '02-segmentos_originais'
    pasta2.mkdir(parents=True, exist_ok=True)

    # create folder with the segments at 16 khz sr
    pasta3 = pasta / '03-segments_16khz'
    pasta3.mkdir(parents=True, exist_ok=True)

    # create folder with the MOS files
    pasta4 = pasta / '04-mos_score'
    pasta4.mkdir(parents=True, exist_ok=True)

    # create folder with the overlap 1 files
    pasta5 = pasta / '05-overlap1'
    pasta5.mkdir(parents=True, exist_ok=True)

    # create folder with the stt_whisper files
    pasta6 = pasta / '06-stt_whisper'
    pasta6.mkdir(parents=True, exist_ok=True)

    # create folder with the stt_wav2vec files
    pasta7 = pasta / '07-stt_wav2vec'
    pasta7.mkdir(parents=True, exist_ok=True)

    # create folder with the normalizador_texto files
    pasta8 = pasta / '08-normalizador_texto'
    pasta8.mkdir(parents=True, exist_ok=True)

    # create folder with the similarity validation files
    pasta9 = pasta / '09-validacao_similaridade'
    pasta9.mkdir(parents=True, exist_ok=True)

    # create folder with the denoiser files
    pasta10 = pasta / '10-denoiser'
    pasta10.mkdir(parents=True, exist_ok=True)

    # create folder with the normalizador_audio files
    pasta11 = pasta / '11-normalizador_audio'
    pasta11.mkdir(parents=True, exist_ok=True)

    #########################################################
    #============================================================
    # Preparing the original files: audio becomes WAV
    #============================================================
    pasta_origem = PROJECT_ROOT / "arquivos" / "audios" / audio_id
    pasta_destino = pasta1

    # Ensure the destination exists
    pasta_destino.mkdir(parents=True, exist_ok=True)

    # Missing source folder is a hard failure: without input there is
    # nothing to process
    if not pasta_origem.exists():
        print(f"ERRO: Pasta de origem nao encontrada: {pasta_origem}")
        return None

    # WAV is the pipeline's internal format. The conversion happens here,
    # once per audio file: from that point on the input format no longer
    # circulates, which allows accepting formats that SoX 14.4.2 cannot
    # read (m4a, aac, wma, opus). A file that is not audio is copied as-is.
    audios_convertidos = 0
    outros_copiados = 0
    specs_origem = None

    for item in sorted(pasta_origem.iterdir()):
        if not item.is_file():
            continue

        if item.suffix.lower() in EXTENSOES_AUDIO:
            specs = sondar_specs_origem(item)
            if specs is None:
                return None
            if not converter_para_wav(item, pasta_destino / f"{item.stem}.wav", specs):
                return None
            audios_convertidos += 1
            # The contract is one audio file per folder; in alphabetical
            # order, the first one is what the pipeline will process
            if specs_origem is None:
                specs_origem = specs
        else:
            shutil.copy2(item, pasta_destino / item.name)
            outros_copiados += 1

    # A folder with no audio at all is a hard failure: failing here costs
    # less than failing six modules down the line
    if audios_convertidos == 0:
        print(f"ERRO: nenhum arquivo de audio em {pasta_origem}")
        return None

    print(f"Originais preparados: {audios_convertidos} audio(s) em WAV, "
          f"{outros_copiados} outro(s) arquivo(s) copiado(s)")
    print(f"Proveniencia da fonte: {specs_origem['codec'] or 'ausente'} | "
          f"{specs_origem['bitrate'] or 'ausente'} bps | "
          f"{specs_origem['sample_rate'] or 'ausente'} Hz")

    #########################################################
    #============================================================
    # Creating dataset folders
    #============================================================
    # Create the dataset folder
    dataset = PROJECT_ROOT / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)

    # Create the audio_dataset folder
    audio_dataset = dataset / 'audio_dataset'
    audio_dataset.mkdir(parents=True, exist_ok=True)

    # Create the history folder
    historico_dataset = dataset / 'historico_dataset'
    historico_dataset.mkdir(parents=True, exist_ok=True)

    # Create the log folder
    log = dataset / 'log'
    log.mkdir(parents=True, exist_ok=True)

    return specs_origem


if __name__ == '__main__':
    # Direct execution requires audio_id as an argument - no fixed id in the code
    if len(sys.argv) != 2:
        print("Uso: python src/m02_diretorios.py <audio_id>")
        sys.exit(1)
    if criar_diretorios(sys.argv[1]) is None:
        sys.exit(1)