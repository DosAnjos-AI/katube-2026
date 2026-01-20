from pathlib import Path
import shutil


id_video= 'CA6TSoMw86k'


#============================================================
# Criando pasta geral do vídeo onde estará todos as subpastas
#============================================================

pasta= Path("arquivos/temp/"+ id_video)
pasta.mkdir(parents=True, exist_ok=True)

#============================================================
# Criando subpastas
#============================================================

# criar pasta com as copias dos arquivos originais
pasta1= Path(str(pasta)+'/01-arquivos_originais')
pasta1.mkdir(parents=True, exist_ok=True)

# criar pasta com os segmentos com sr original
pasta2= Path(str(pasta)+'/02-segmentos_originais')
pasta2.mkdir(parents=True, exist_ok=True)

# criar pasta com os segmetnos com sr a 16 khz
pasta3= Path(str(pasta)+'/03-segments_16khz')
pasta3.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos da MOS
pasta4= Path(str(pasta)+'/04-mos_score')
pasta4.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do overlap 1
pasta5= Path(str(pasta)+ '/05-overlap1')
pasta5.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do overlap 2
pasta6= Path(str(pasta)+ '/06-overlap2')
pasta6.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do stt-whisper
pasta7= Path(str(pasta)+ '/07-stt_whisper')
pasta7.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do stt-wav2vec
pasta8= Path(str(pasta)+ '/08-stt_wav2vec')
pasta8.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do normalizador de texto
pasta9= Path(str(pasta)+ '/09-normalizador_texto')
pasta9.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do validação
pasta10= Path(str(pasta)+ '/10-validacao_levenstein')
pasta10.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do Denoiser
pasta11= Path(str(pasta)+ '/11-denoiser')
pasta11.mkdir(parents=True, exist_ok=True)

# criar pasta com arquivos do normalizador de texto
pasta12= Path(str(pasta)+ '/12-normalizador_audio')
pasta12.mkdir(parents=True, exist_ok=True)


#########################################################
#============================================================
# Criando copia dos arquivos originais
#============================================================

pasta_origem = Path("./arquivos/audios/"+id_video)
pasta_destino = pasta1

# Garantir que destino existe
pasta_destino.mkdir(parents=True, exist_ok=True)

# Copiar TODOS os arquivos (qualquer tipo, qualquer nome)
for item in pasta_origem.iterdir():
    if item.is_file():
        shutil.copy2(item, pasta_destino / item.name)