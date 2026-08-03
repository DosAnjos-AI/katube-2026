import sys
from pathlib import Path
import shutil

# Definir PROJECT_ROOT no escopo global
PROJECT_ROOT = Path(__file__).resolve().parent.parent



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


def criar_diretorios(audio_id: str) -> bool:
    """
    Prepara a estrutura de diretorios do audio_id e copia a entrada.

    Returns:
        True se a estrutura foi criada e a entrada copiada, False se a
        pasta de origem nao existe (nada a processar).
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

    # criar pasta com arquivos do validacao_levenstein
    pasta9 = pasta / '09-validacao_levenshtein'
    pasta9.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do denoiser
    pasta10 = pasta / '10-denoiser'
    pasta10.mkdir(parents=True, exist_ok=True)

    # criar pasta com arquivos do normalizador_audio
    pasta11 = pasta / '11-normalizador_audio'
    pasta11.mkdir(parents=True, exist_ok=True)

    #########################################################
    #============================================================
    # Criando copia dos arquivos originais
    #============================================================
    pasta_origem = PROJECT_ROOT / "arquivos" / "audios" / audio_id
    pasta_destino = pasta1

    # Garantir que destino existe
    pasta_destino.mkdir(parents=True, exist_ok=True)

    # Pasta de origem ausente e falha dura: sem entrada nao ha o que processar
    if not pasta_origem.exists():
        print(f"ERRO: Pasta de origem nao encontrada: {pasta_origem}")
        return False

    # Copiar TODOS os arquivos (qualquer tipo, qualquer nome)
    for item in pasta_origem.iterdir():
        if item.is_file():
            shutil.copy2(item, pasta_destino / item.name)

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

    return True


if __name__ == '__main__':
    # Execucao direta exige o audio_id como argumento - sem id fixo no codigo
    if len(sys.argv) != 2:
        print("Uso: python src/m02_diretorios.py <audio_id>")
        sys.exit(1)
    if not criar_diretorios(sys.argv[1]):
        sys.exit(1)