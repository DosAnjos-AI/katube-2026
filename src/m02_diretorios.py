import json
import sys
import subprocess
from pathlib import Path
from typing import Optional
import shutil

# Definir PROJECT_ROOT no escopo global
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import EXTENSOES_AUDIO


def sondar_specs_origem(origem: Path) -> Optional[dict]:
    """
    Sonda o arquivo-FONTE com ffprobe, uma unica vez, antes de converter.

    Depois da conversao o formato de origem some da pipeline: o que circula e
    o WAV interno. Sondar aqui e a ultima chance de registrar de onde o audio
    veio - essas specs viajam ate as colunas origem_* do dataset.csv.

    O mesmo resultado decide a profundidade de bits do WAV: o padrao do
    FFmpeg e pcm_s16le, que truncaria um original de 24 bits SEM AVISO.

    Returns:
        Dict com formato, codec, bitrate, sample_rate, duracao e
        bits_por_amostra, ou None se o ffprobe falhou (o chamador aborta).
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

    # Ausencia e None, nunca a string 'N/A': codec, bitrate e sample_rate
    # viajam ate as colunas origem_* do dataset.csv, onde ausencia tem que ser
    # celula vazia. 'N/A' seria TEXTO numa coluna numerica.
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
    Traduz a profundidade de bits da FONTE no codec PCM correspondente.

    Ponto unico de decisao: e usada tanto aqui, na conversao para WAV, quanto
    pelo m04, no corte dos segmentos. Duas copias da regra sairiam do ar uma
    da outra, e o segmento acabaria com profundidade diferente do original de
    onde saiu.

    Args:
        specs: specs da fonte ja sondadas (ver sondar_specs_origem).

    Returns:
        Nome do codec PCM para o parametro '-c:a' do FFmpeg.
    """
    # Formatos comprimidos (mp3, m4a, ogg, aac, wma) decodificam em ponto
    # flutuante e nao declaram bits por amostra - 16 bits e o destino correto
    bits = str(specs.get('bits_por_amostra'))
    if bits == '24':
        return 'pcm_s24le'
    if bits == '32':
        return 'pcm_s32le'
    return 'pcm_s16le'


def converter_para_wav(origem: Path, destino: Path, specs: dict) -> bool:
    """
    Converte o audio para WAV preservando os parametros do original.

    Sem '-ar' e sem '-ac': o FFmpeg copia o sample rate e a contagem de
    canais da origem. Nada de resampling, nem para cima nem para baixo.

    Args:
        specs: specs da fonte ja sondadas, de onde sai a profundidade de bits.

    Returns:
        True se o WAV foi escrito, False caso contrario (com log).
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
    Remove todo o estado deixado por rodadas anteriores deste audio_id.

    Sem isso, o que a nova rodada nao regravar sobrevive e se mistura ao
    estado novo: segmentos orfaos em temp/{id} e .flac orfaos no
    dataset entregue. O processamento de um id nasce sempre do zero.
    """
    # Guarda obrigatoria: id vazio, '.' ou '..' colapsa o caminho na raiz e
    # o rmtree passaria a mirar temp/ e audio_dataset/ inteiros. Falha alto.
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
    Prepara a estrutura de diretorios do audio_id e converte a entrada.

    Returns:
        As specs do audio-FONTE (ver sondar_specs_origem), que o chamador
        repassa ao m04 para a proveniencia do dataset, ou None se algo
        falhou (pasta de origem ausente, sem audio, ffprobe ou ffmpeg).
    """
    #============================================================
    # Reinicio limpo: nada de rodada anterior sobrevive
    #============================================================
    limpar_estado_anterior(audio_id)

    #============================================================
    # Criando pasta geral do audio onde estara todas as subpastas
    #============================================================
    pasta = PROJECT_ROOT / "arquivos" / "temp" / audio_id
    pasta.mkdir(parents=True, exist_ok=True)

    #============================================================
    # Criando subpastas para arquivos intermediarios
    #============================================================
    # criar pasta para os .json dinâmicos
    pasta1 = pasta / '00-json_dinamico'
    pasta1.mkdir(parents=True, exist_ok=True)

    # criar pasta com as copias dos arquivos originais
    pasta1 = pasta / '01-arquivos_originais'
    pasta1.mkdir(parents=True, exist_ok=True)

    # criar pasta com os segmentos com sr original
    pasta2 = pasta / '02-segmentos_originais'
    pasta2.mkdir(parents=True, exist_ok=True)

    # criar pasta com os segmetnos com sr a 16 khz
    pasta3 = pasta / '03-segments_16khz'
    pasta3.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos da MOS
    pasta4 = pasta / '04-mos_score'
    pasta4.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do overlap 1
    pasta5 = pasta / '05-overlap1'
    pasta5.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do -stt_whisper
    pasta6 = pasta / '06-stt_whisper'
    pasta6.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do stt_wav2vec
    pasta7 = pasta / '07-stt_wav2vec'
    pasta7.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do normalizador_texto
    pasta8 = pasta / '08-normalizador_texto'
    pasta8.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos da validacao de similaridade
    pasta9 = pasta / '09-validacao_similaridade'
    pasta9.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do denoiser
    pasta10 = pasta / '10-denoiser'
    pasta10.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do normalizador_audio
    pasta11 = pasta / '11-normalizador_audio'
    pasta11.mkdir(parents=True, exist_ok=True)

    #########################################################
    #============================================================
    # Preparando os arquivos originais: audio vira WAV
    #============================================================
    pasta_origem = PROJECT_ROOT / "arquivos" / "audios" / audio_id
    pasta_destino = pasta1

    # Garantir que destino existe
    pasta_destino.mkdir(parents=True, exist_ok=True)

    # Pasta de origem ausente e falha dura: sem entrada nao ha o que processar
    if not pasta_origem.exists():
        print(f"ERRO: Pasta de origem nao encontrada: {pasta_origem}")
        return None

    # WAV e o formato interno da pipeline. A conversao acontece aqui, uma vez
    # por audio: dali em diante o formato de entrada nao circula mais, o que
    # permite aceitar formatos que o SoX 14.4.2 nao le (m4a, aac, wma, opus).
    # Arquivo que nao e audio segue copiado como esta.
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
            # O contrato e um audio por pasta; na ordem alfabetica, o primeiro
            # e o que a pipeline vai processar
            if specs_origem is None:
                specs_origem = specs
        else:
            shutil.copy2(item, pasta_destino / item.name)
            outros_copiados += 1

    # Pasta sem audio nenhum e falha dura: falhar aqui custa menos que falhar
    # seis modulos adiante
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
    # Criando pastas de dataset
    #============================================================
    # Criar a pasta de dataset
    dataset = PROJECT_ROOT / 'dataset'
    dataset.mkdir(parents=True, exist_ok=True)

    # Criar a pasta de audio_dataset
    audio_dataset = dataset / 'audio_dataset'
    audio_dataset.mkdir(parents=True, exist_ok=True)

    # Criar a pasta de historico
    historico_dataset = dataset / 'historico_dataset'
    historico_dataset.mkdir(parents=True, exist_ok=True)

    # Criar a pasta de log
    log = dataset / 'log'
    log.mkdir(parents=True, exist_ok=True)

    return specs_origem


if __name__ == '__main__':
    # Execucao direta exige o audio_id como argumento - sem id fixo no codigo
    if len(sys.argv) != 2:
        print("Uso: python src/m02_diretorios.py <audio_id>")
        sys.exit(1)
    if criar_diretorios(sys.argv[1]) is None:
        sys.exit(1)