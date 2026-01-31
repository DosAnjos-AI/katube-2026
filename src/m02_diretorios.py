from pathlib import Path
import shutil

# Definir PROJECT_ROOT no escopo global
PROJECT_ROOT = Path(__file__).resolve().parent.parent

id_video = 'B4RgpqJhoIo'

print(str(PROJECT_ROOT) + "/arquivos/temp/" + id_video)

def criar_diretorios():
    #============================================================
    # Criando pasta geral do vídeo onde estará todos as subpastas
    #============================================================
    pasta = PROJECT_ROOT / "arquivos" / "temp" / id_video
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
    pasta_origem = PROJECT_ROOT / "arquivos" / "audios" / id_video
    pasta_destino = pasta1

    # Garantir que destino existe
    pasta_destino.mkdir(parents=True, exist_ok=True)

    # Verificar se a pasta de origem existe antes de copiar
    if not pasta_origem.exists():
        print(f"AVISO: Pasta de origem não encontrada: {pasta_origem}")
    else:
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


if __name__ == '__main__':
    criar_diretorios()