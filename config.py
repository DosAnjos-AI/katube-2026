from pathlib import Path

# ============================================================================
# DIRETÓRIO RAIZ DO PROJETO
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================================
# FORMATOS DE ÁUDIO - FONTE ÚNICA
# ============================================================================
# Conjunto de extensões que os módulos do pipeline reconhecem como áudio.
# Esta é a ÚNICA definição do projeto: todo módulo importa daqui.
# Comparação sempre com `suffix.lower()`.
#
# ----------------------------------------------------------------------------
# AS TRÊS LISTAS DE FORMATO (confirmadas empiricamente no build instalado
# nesta máquina - relatório 04, Ubuntu 24.04.4)
# ----------------------------------------------------------------------------
#
# 1) ENTRADA - o que o FFmpeg 6.1.1 consegue DECODIFICAR:
#       mp3, wav, flac, ogg, opus, m4a, aac, wma, aiff, au
#
#    ATENÇÃO: a lista EM VIGOR abaixo (EXTENSOES_AUDIO) é mais restrita que
#    essa - ela NÃO inclui `opus`, `aiff` nem `au`. Motivo: hoje não existe
#    conversão para WAV na porta de entrada, então o formato do arquivo de
#    entrada atravessa o pipeline inteiro e precisa ser processável por
#    TODOS os módulos. Um `.opus` era aceito na entrada e rejeitado seis
#    módulos adiante. Enquanto a conversão de entrada não existir, o
#    arquivo é recusado logo na porta, com aviso, em vez de falhar no meio.
#
# 2) INTERMEDIÁRIO - `wav`, fixo e não configurável. É o formato usado
#    internamente entre etapas (VAD sempre trabalha em 16 kHz mono).
#
# 3) SAÍDA - o que o SoX 14.4.2 consegue ESCREVER de forma útil:
#       flac, wav, ogg, mp3   (padrão do projeto: flac - ver SOX_NORMALIZER)
#
#    O SoX também escreve aiff, au, caf, w64, raw, voc e amr-nb, mas estes
#    ficam documentados como NÃO RECOMENDADOS - `amr-nb`, em particular,
#    força reamostragem para 8 kHz. Quem precisar de outro formato converte
#    fora do projeto, com FFmpeg.
#
# ----------------------------------------------------------------------------
# ORIGEM DOS BINÁRIOS (relatório 04)
# ----------------------------------------------------------------------------
# `sox` v14.4.2 e `soxi` vêm de /usr/bin, ou seja, do SISTEMA OPERACIONAL,
# NÃO do env conda `katube-2026`. O mesmo vale para `ffmpeg` e `ffprobe`
# 6.1.1 (/usr/bin). Consequência prática: recriar o env a partir do
# `environment.yml` NÃO reproduz o ambiente - os binários precisam ser
# instalados à parte, no sistema, e a versão deles muda o que o pipeline
# consegue ler e escrever.
EXTENSOES_AUDIO = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aac', '.wma'}


# ============================================================================
# BLOCO MASTER - Controle de Módulos Ativos
# ============================================================================
# Ativa/desativa módulos principais do sistema
# True = módulo será executado | False = módulo será ignorado
MASTER = {
    # Opcoes: 'vad'. O valor '' (audio ja segmentado) esta previsto no codigo
    # mas NAO FUNCIONA: nenhum modulo alimenta 02-segmentos_originais sem o
    # m04, e o m05 aborta a pipeline em 0,00 min por falta do JSON. Nao use
    # ate que exista um caminho de entrada para segmentos prontos.
    'segmentacao': 'vad',
    'mos_filter': True,        # True = utilizar; False = nao utilizar
    'overlap': True,           # Utilizar ou nao o detector de overlap
    'transcricao_whisper': True,
    'transcricao_wav2vec': True,
    'Denoiser': True,
    'cleanup': 'all',          # Opcoes: 'all' (input+temp), 'input' (so input), 'temp' (so temp), 'none' (nao apaga)

    # ------------------------------------------------------------------------
    # FALLBACK PARA CPU - controle unico e global de todos os modelos
    # ------------------------------------------------------------------------
    # Governa o que acontece quando um bloco pede 'gpu' e a GPU nao atende
    # (CUDA ausente, erro de CUDA, falta de VRAM, qualquer excecao no
    # carregamento do modelo).
    #
    #   True  -> a pipeline CAI PARA CPU e continua. A excecao original e o
    #            dispositivo efetivamente usado sao registrados em nivel de
    #            ERRO, no stderr, com o prefixo [ERRO]. O log alto e
    #            obrigatorio: uma rodada que cai para CPU sem ninguem
    #            perceber custa cerca de 9,7x tempo real e passa por normal.
    #
    #   False -> SEM fallback. O erro derruba o modulo, que devolve False
    #            pelo contrato de retorno, e o main.py marca o audio como
    #            falho em processamento_metadados.csv.
    #
    # Vale para os cinco modelos (Whisper, wav2vec, pyannote, SQUIM e
    # DeepFilterNet3). Ver a RESSALVA do DeepFilterNet3 em
    # DEEPFILTERNET_DENOISER['device'].
    'fallback_cpu': True,
}


# =============================================================================
# MÓDULO 01: SEGMENTADOR DE ÁUDIO VAD (VOICE ACTIVITY DETECTION)
# =============================================================================

# Configurações do módulo de segmentação automática usando detecção de voz
# Utilizado quando MASTER['segmentacao'] = 'vad'
# 
# Este módulo utiliza Silero-VAD para detectar automaticamente momentos de
# fala e silêncio no áudio, criando segmentos baseados em pausas naturais.
#
# IMPORTANTE: Sempre processa áudio em 16 kHz internamente (conversão automática)
SEGMENTADOR_AUDIO_VAD = {
    
    # ------------------------------------------------------------------------
    # Detecção de Voz (Voice Activity Detection)
    # ------------------------------------------------------------------------
    'deteccao': {
        # Threshold de confiança para considerar que há voz presente (0.0 a 1.0)
        # - Valores BAIXOS (0.3-0.4): mais sensível, detecta até sussurros/ruídos
        # - Valores MÉDIOS (0.5): equilíbrio, recomendado para maioria dos casos
        # - Valores ALTOS (0.6-0.8): menos sensível, só detecta voz clara
        # Exemplo: 0.5 = confiança média (50%) para considerar como voz
        'voice_threshold': 0.5,
        
        # Tamanho da janela de análise em segundos
        # Define a granularidade temporal da detecção de voz
        # - Valores menores (0.05-0.1): maior precisão, mais processamento
        # - Valores médios (0.15): equilíbrio recomendado
        # - Valores maiores (0.3-0.5): menos precisão, mais rápido
        # Exemplo: 0.15 = analisa o áudio em blocos de 150ms
        'window_size_seconds': 0.15,
    },
    
    # ------------------------------------------------------------------------
    # Critérios de Silêncio e Fala
    # ------------------------------------------------------------------------
    'criterios': {
        # Duração mínima de FALA para ser considerada válida (milissegundos)
        # Falas mais curtas são ignoradas (consideradas ruído)
        # - Valores baixos (100-200ms): captura falas muito breves
        # - Valores médios (250-500ms): filtra ruídos curtos (recomendado)
        # - Valores altos (>500ms): só aceita falas longas
        # Exemplo: 250 = ignora sons com menos de 250ms (0.25s)
        'min_speech_duration_ms': 250,
        
        # Duração mínima de SILÊNCIO entre falas (milissegundos)
        # Silêncios mais curtos não separam falas (continuam no mesmo segmento)
        # - Valores baixos (50-100ms): divide em pausas muito breves
        # - Valores médios (100-200ms): equilíbrio (recomendado)
        # - Valores altos (>300ms): só divide em pausas longas
        # Exemplo: 100 = pausas menores que 100ms não dividem o segmento
        'min_silence_duration_ms': 100,
        
        # Duração mínima de SILÊNCIO para FORÇAR divisão de segmento (segundos)
        # Pausas maiores que este valor sempre criam novo segmento
        # - Valores baixos (0.2-0.3s): sensível a pausas curtas
        # - Valores médios (0.3-0.5s): equilíbrio (recomendado)
        # - Valores altos (>0.5s): só divide em pausas muito longas
        # Exemplo: 0.3 = pausa de 300ms sempre cria novo segmento
        'min_silence_for_split': 0.3,
    },
    
    # ------------------------------------------------------------------------
    # Padding (Margem de Segurança nos Cortes)
    # ------------------------------------------------------------------------
    'padding': {
        # Tempo adicional no INÍCIO de cada segmento (milissegundos)
        # Evita cortar o início da primeira palavra
        # - Valores baixos (10-30ms): corte mais preciso
        # - Valores médios (30-50ms): segurança recomendada
        # - Valores altos (>100ms): pode incluir silêncio extra
        #
        # ATENÇÃO - este valor é aplicado DUAS VEZES no início: uma pelo
        # Silero (o m04 o passa como speech_pad_ms, e o Silero expande os
        # dois lados) e outra pelo próprio m04 ao montar o grupo. Com 30 ms
        # configurados, o início recebe 60 ms no total. O fim recebe
        # inicio_ms (pelo Silero) + fim_ms (pelo m04).
        'inicio_ms': 30,
        
        # Tempo adicional no FIM de cada segmento (milissegundos)
        # Evita cortar o final da última palavra
        # - Valores baixos (10-30ms): corte mais preciso
        # - Valores médios (30-50ms): segurança recomendada
        # - Valores altos (>100ms): pode incluir silêncio extra
        # Exemplo: 30 = adiciona 30ms após o fim detectado da fala
        # Este campo NÃO chega ao Silero: só o m04 o aplica.
        'fim_ms': 30,
    },
    
    # ------------------------------------------------------------------------
    # Limites de Duração dos Segmentos Finais
    # ------------------------------------------------------------------------
    'segmentos': {
        # Duração mínima ALVO de cada segmento, em segundos
        # Segmentos mais curtos são agrupados com próximos
        # - Valores baixos (2-4s): aceita segmentos muito curtos
        # - Valores médios (4-8s): equilíbrio (recomendado)
        # - Valores altos (>10s): força segmentos longos
        #
        # ATENÇÃO: o mínimo EFETIVO é (min_seg - tolerancia), não min_seg.
        # Com min_seg=4.0 e tolerancia=0.5, o mínimo real é 3,5 s.
        'min_seg': 4.0,
        
        # Teto de AGRUPAMENTO, em segundos - NÃO é teto absoluto
        # Impede juntar mais um trecho de fala a um grupo que já alcançou
        # este limite. Um trecho de fala único MAIOR que o limite
        # ATRAVESSA INTEIRO - nada o divide.
        # - Valores baixos (8-12s): força segmentos curtos
        # - Valores médios (15-20s): equilíbrio (recomendado)
        # - Valores altos (>25s): permite segmentos muito longos
        #
        # Medido neste projeto: com max_seg=8.0, um segmento saiu com
        # 8,312 s. Com 15.0, os segmentos ficaram entre 6,5 e 12,3 s.
        'max_seg': 15.0,
        
        # Tolerância nas durações (segundos)
        # Permite pequenas variações nos limites min/max
        # - Valores baixos (0.3-0.5s): mais rigoroso
        # - Valores médios (0.8-1.0s): equilíbrio (recomendado)
        # - Valores altos (>1.5s): mais flexível
        # Exemplo com os valores atuais: 0.5 aceita segmento de 3,5 s
        # (min_seg 4.0 - tolerancia 0.5)
        'tolerancia': 0.5,
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # CAMPO SEM EFEITO HOJE.
        # O código do skip existe (m04), mas o m02 apaga temp/{audio_id}
        # inteiro no início de todo processamento: quando a verificação
        # acontece, a pasta de destino acabou de ser criada vazia, e a
        # resposta é sempre "não há nada para pular". Confirmado em sete
        # rodadas reprocessando o mesmo audio_id - a mensagem de skip nunca
        # apareceu. Ligar isto exige decidir antes qual invariante vence:
        # o reinício limpo ou o reaproveitamento.
        'sobrescrever': False,
    },
}

# =============================================================================
# MÓDULO 02: MOS FILTER (MEAN OPINION SCORE - QUALIDADE DE ÁUDIO)
# =============================================================================

# Configurações do filtro de qualidade de áudio baseado em MOS
# Avalia áudio usando modelo SQUIM (Speech Quality and Intelligibility Measures)
# Classifica segmentos em: baixa, média ou alta qualidade
MOS_FILTER = {
    
    # ------------------------------------------------------------------------
    # Dispositivo de Processamento
    # ------------------------------------------------------------------------
    # Define onde o modelo MOS será executado
    # Opções disponíveis - APENAS DUAS ("auto" foi eliminado):
    # - "gpu": pede GPU/CUDA. Se a GPU não atender, o que acontece é
    #          decidido por MASTER['fallback_cpu'], não por este campo.
    # - "cpu": força uso de CPU (mais lento, mas funciona em qualquer máquina)
    #
    # Qualquer outro valor é RECUSADO com erro explícito no m01 - não há
    # resolução silenciosa nem valor padrão escondido.
    # Nota: GPU acelera significativamente (3-5x mais rápido)
    'device': 'gpu',
    
    # ------------------------------------------------------------------------
    # Limiares de Qualidade (MOS Score)
    # ------------------------------------------------------------------------
    # MOS (Mean Opinion Score) varia de 1.0 (péssimo) a 5.0 (excelente)
    # Define os limiares para classificação de qualidade
    
    'thresholds': {
        # Limiar mínimo aceitável
        # Áudios com MOS < min_threshold são DESCARTADOS
        # Valores típicos: 1.5-2.5
        # Exemplo: 2.0 = descarta áudios muito ruins
        'min_threshold': 2.0,
        
        # Limiar para alta qualidade
        # Áudios com MOS >= max_threshold são classificados como 'alta'
        # Valores típicos: 3.0-4.0
        # Exemplo com o valor atual: MOS >= 3.0 é classificado como 'alta'
        #
        # Este campo NÃO decide quem passa pelo denoiser - quem decide é
        # DEEPFILTERNET_DENOISER['mos_quality_filter'].
        'max_threshold': 3.0,
        
        # Faixa intermediária (calculada automaticamente):
        # min_threshold <= MOS < max_threshold
        # Estes áudios passam por denoising antes do dataset final
    },
    
    # ------------------------------------------------------------------------
    # Batch Processing (Processamento em Lote)
    # ------------------------------------------------------------------------
    # Processa múltiplos áudios simultaneamente para maior eficiência
    
    'batch': {
        # Tamanho do batch (quantos áudios processar juntos)
        # Valores maiores = mais rápido, mas usa mais VRAM
        # 
        # Opções:
        # - "auto": neste bloco NÃO olha a VRAM. O m06 devolve 8 para GPU e
        #           1 para CPU, e só. (O "auto" do STT_WHISPER é diferente:
        #           lá a VRAM é lida de verdade.)
        # - 1-16: Valor fixo (números maiores exigem mais VRAM)
        #
        # Referência de uso de VRAM (aproximado):
        # - batch_size=1:  ~2.0 GB
        # - batch_size=4:  ~3.0 GB
        # - batch_size=8:  ~4.0 GB
        # - batch_size=16: ~6.0 GB
        # 
        # Recomendação:
        # - GPU com 24GB: "auto" ou 16
        # - GPU com 8-16GB: 8
        # - GPU com 4-8GB: 4
        # - CPU: 1-2
        'batch_size': 'auto',
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # CAMPO SEM EFEITO HOJE.
        # Mesmo caso do bloco do VAD: o código do skip existe (m06), mas o
        # m02 apaga temp/{audio_id} antes, então o JSON que a verificação
        # procura nunca está lá. Ligar isto é decisão de projeto, não de
        # implementação.
        'sobrescrever': False,
    },
}

# =============================================================================
# MÓDULO 04: DETECTOR DE OVERLAP 01 (SOBREPOSIÇÃO DE LOCUTORES)
# =============================================================================

# Configurações do detector de overlap usando diarização de speakers
# Detecta se há sobreposição de fala (múltiplos locutores falando simultaneamente)
# Utiliza modelo pyannote para análise de áudio
OVERLAP_DETECTOR = {
    
    # ------------------------------------------------------------------------
    # Dispositivo de Processamento
    # ------------------------------------------------------------------------
    # Define onde o modelo será executado
    # Opções disponíveis - APENAS DUAS ("auto" foi eliminado):
    # - "gpu": pede GPU/CUDA. Se a GPU não atender, o que acontece é
    #          decidido por MASTER['fallback_cpu'], não por este campo.
    # - "cpu": força uso de CPU (mais lento, mas funciona em qualquer máquina)
    #
    # Qualquer outro valor é RECUSADO com erro explícito no m01.
    # Nota: GPU acelera significativamente o processamento
    'device': 'gpu',
    
    # ------------------------------------------------------------------------
    # Modelo de Diarização
    # ------------------------------------------------------------------------
    # Modelo HuggingFace para detecção de overlap.
    # Este campo É lido pelo m01 e define de fato o modelo carregado -
    # antes havia uma constante no código que o ignorava.
    # IMPORTANTE: Requer token HuggingFace configurado em .env
    'modelo': 'pyannote/speaker-diarization-3.1',
    
    # ------------------------------------------------------------------------
    # Batch Processing (Processamento em Lote)
    # ------------------------------------------------------------------------
    'batch': {
        # ÚNICO VALOR SUPORTADO: 1
        #
        # Este módulo processa um segmento por vez, com timeout individual
        # por áudio - não existe caminho de lote no m07. O campo é LIDO e
        # qualquer valor diferente de 1 é RECUSADO com erro explícito, que
        # derruba o módulo. Antes o valor era simplesmente ignorado, e quem
        # configurasse 8 não tinha como perceber.
        #
        # Processamento em lote aqui seria funcionalidade nova, não ajuste
        # de configuração.
        'batch_size': 1,
    },
    'timeout': {
    'por_audio_segundos': 150,  # Timeout máximo por áudio
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # CAMPO SEM EFEITO HOJE - e sem leitor nenhum.
        # Nada no m07 verifica o campo overlap01 antes de processar; este
        # valor não é consultado por linha alguma do projeto.
        'sobrescrever': False,
    },
}

# =============================================================================
# MÓDULO 05: STT WHISPER (SPEECH-TO-TEXT)
# =============================================================================

# Configurações do módulo de transcrição usando Whisper
# Converte segmentos de áudio em texto usando modelo distil-whisper PT-BR
# Modelo: freds0/distil-whisper-large-v3-ptbr
STT_WHISPER = {
    
    # ------------------------------------------------------------------------
    # Dispositivo de Processamento
    # ------------------------------------------------------------------------
    # Define onde o modelo Whisper será executado
    # Opções disponíveis - APENAS DUAS ("auto" foi eliminado):
    # - "gpu": pede GPU/CUDA. Se a GPU não atender, o que acontece é
    #          decidido por MASTER['fallback_cpu'], não por este campo.
    # - "cpu": força uso de CPU (mais lento, mas funciona em qualquer máquina)
    #
    # Qualquer outro valor é RECUSADO com erro explícito no m01.
    # Nota: GPU acelera significativamente (6x mais rápido que large-v3)
    'device': 'gpu',
    
    # ------------------------------------------------------------------------
    # Batch Processing (Processamento em Lote)
    # ------------------------------------------------------------------------
    # Processa múltiplos áudios simultaneamente para maior eficiência
    
    'batch': {
        # Tamanho do batch (quantos áudios transcrever juntos)
        # Valores maiores = mais rápido, mas usa mais VRAM
        # 
        # Opções:
        # - "auto": Calcula automaticamente baseado em VRAM disponível
        # - 1-16: Valor fixo (números maiores exigem mais VRAM)
        # 
        # Referência de uso de VRAM (aproximado):
        # - batch_size=1:  ~2.5 GB
        # - batch_size=4:  ~4.0 GB
        # - batch_size=8:  ~6.0 GB
        # - batch_size=16: ~10.0 GB
        # 
        # Recomendação:
        # - GPU com 24GB: "auto" ou 16
        # - GPU com 8-16GB: 8
        # - GPU com 4-8GB: 4
        # - CPU: 1 automático
        'batch_size': 8,
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # CAMPO SEM EFEITO HOJE - e sem leitor nenhum.
        # Nada no m08 verifica o campo stt_whisper antes de transcrever;
        # este valor não é consultado por linha alguma do projeto.
        'sobrescrever': False,
    },
}

# ============================================================
# MÓDULO 06: STT WAV2VEC2 (SPEECH-TO-TEXT) 
# ============================================================
STT_WAV2VEC2= {
    # Dispositivo de processamento - APENAS "gpu" ou "cpu" ("auto" foi
    # eliminado). Com "gpu", a queda para CPU é decidida por
    # MASTER['fallback_cpu']. Outro valor é recusado com erro no m01.
    "device": "gpu",
}


# ============================================================
# MÓDULO 07: NORMALIZADOR DE TEXTO (TEXT NORMALIZER)
# ============================================================
TEXT_NORMALIZER = {

    # Pontuação que afeta dicção/pronúncia do locutor
    # Remove: . , ; ! ? _
    # Exemplo: "Olá, mundo!" → "Olá mundo" (com remove=True)
    # Exemplo: "Olá, mundo!" → "Olá, mundo!" (com remove=False)
    # Útil para: comparação STT onde pontuação não é transcrita
    "remove_punctuation_diction": True,
    
    # Acentuação gráfica (marcas diacríticas)
    # Remove TODA marca diacrítica, por decomposição Unicode NFD seguida do
    # descarte da categoria Mn: agudo, grave, circunflexo, til E CEDILHA.
    # Exemplos verificados: "josé" → "jose", "coração" → "coracao",
    #                       "açúcar" → "acucar" (a cedilha some junto)
    # Exemplo: "josé" → "josé" (com remove=False)
    # IMPORTANTE: Se False, números por extenso terão acento (três, décimo)
    #             Se True, números por extenso sem acento (tres, decimo)
    # Útil para: normalização para modelos que não lidam bem com acentos
    "remove_accents_graphic": True,
}

# ============================================================
# MÓDULO 08: VALIDADOR DE SIMILARIDADE (SIMILARITY VALIDATOR)
# ============================================================
SIMILARITY_VALIDATOR = {
    
    # Limiar de similaridade para aprovação de segmentos
    # Valor entre 0.0 (totalmente diferente) e 1.0 (idêntico)
    # Segmentos com similaridade >= threshold são aprovados
    # Exemplo com o valor atual (0.85):
    #   - Similaridade 0.90 → APROVADO (>=0.85)
    #   - Similaridade 0.80 → REJEITADO (<0.85)
    # Valores típicos: 0.70 (permissivo), 0.80 (equilibrado), 0.90 (restritivo)
    # Útil para: filtrar segmentos com alta divergência entre modelos STT
    "similarity_threshold": 0.85,
    
    # Tipo de métrica para cálculo de similaridade
    #
    # ATENÇÃO: as três opções produzem HOJE resultados IDÊNTICOS. As três
    # funções do m11 calculam a mesma coisa - 1 - (distância de Levenshtein
    # / comprimento máximo) sobre CARACTERES. O "wer" não mede palavras: ele
    # separa o texto em palavras e imediatamente as rejunta com espaço,
    # voltando a medir caracteres. Verificado: 'casa azul' contra
    # 'casa verde' devolve 0,5000 nas três. Trocar este campo não altera
    # nenhuma nota nem nenhuma aprovação. A descrição abaixo é a intenção
    # de projeto, a ser cumprida no redesenho da similaridade.
    #
    # Opções disponíveis:
    #   "wer" (Word Error Rate) - Padrão da indústria STT
    #       Calcula: 1 - (edições_necessárias / total_palavras)
    #       Exemplo: "casa azul" vs "casa verde" → WER ~0.50
    #       Sensível à ordem das palavras
    #       Melhor para: avaliar qualidade de transcrição completa
    #   
    #   "cer" (Character Error Rate) - Análise em nível de caractere
    #       Calcula: 1 - (edições_necessárias / total_caracteres)
    #       Exemplo: "josé" vs "jose" → CER ~0.80
    #       Mais granular que WER
    #       Melhor para: detectar erros sutis (acentos, typos)
    #   
    #   "levenshtein_norm" - Distância de edição normalizada
    #       Calcula: 1 - (distância_levenshtein / max_length)
    #       Exemplo: "gato" vs "pato" → 0.75 (1 edição em 4 chars)
    #       Não diferencia palavras vs caracteres
    #       Melhor para: comparação genérica de strings
    # 
    # Recomendação: "wer" para STT (alinhado com métricas acadêmicas)
    "metric_type": "wer",
}

# ============================================================
# MÓDULO 09: DENOISER DEEPFILTERNET
# ============================================================
DEEPFILTERNET_DENOISER = {
    
    # Filtro de qualidade MOS para seleção de áudios a processar
    # Array com categorias de qualidade desejadas
    # Opções disponíveis: "alta", "media", "baixa"
    # Exemplos de uso:
    #   ["alta", "media", "baixa"] → Processa TODOS os áudios (independente de MOS)
    #   ["alta"] → Processa APENAS áudios de alta qualidade
    #   ["media", "baixa"] → Processa áudios de média e baixa qualidade
    #   [] → NÃO PROCESSA NENHUM ÁUDIO (array vazio = sem processamento)
    # 
    # ATENÇÃO - o que chega de fato até aqui:
    #   "baixa" NUNCA chega: o m06 descarta esses segmentos antes, por
    #   min_threshold. Só restam "media" e "alta".
    #   Com max_threshold=3.0, os segmentos deste projeto têm saído quase
    #   todos como "alta" (MOS de 3,36 a 3,89). O valor ["media"] selecionou
    #   ZERO segmentos em todas as rodadas de teste - o denoiser carrega,
    #   imprime cabeçalho e não processa nada. Isso é a semântica correta
    #   (áudio de qualidade alta não precisa de tratamento), mas quem
    #   quiser ver o denoiser agir precisa incluir "alta".
    #
    # IMPORTANTE: Os arquivos originais (input) SEMPRE permanecem intactos
    #             O denoising cria novos arquivos processados no output
    "mos_quality_filter": ["media"],
    
    # Dispositivo de processamento - APENAS DUAS opções ("auto" foi eliminado):
    #   "gpu" → pede GPU/CUDA; a queda para CPU é decidida por
    #           MASTER['fallback_cpu'], não por este campo
    #   "cpu" → força processamento em CPU (mais lento, sempre disponível)
    # Qualquer outro valor é RECUSADO com erro explícito no m01.
    #
    # RESSALVA DO DEEPFILTERNET3 (achado A29, medido no build instalado):
    # a biblioteca resolve o dispositivo por conta própria, com get_device(),
    # que escolhe cuda:0 sempre que houver CUDA. O init_df() NÃO aceita
    # parâmetro device, e o config.ini interno do modelo traz [train]
    # device = (vazio). Para submeter a biblioteca a este campo, o m01
    # escreve a chave DEVICE na config interna do DeepFilterNet logo DEPOIS
    # do init_df() - antes não adianta, porque o próprio init_df() recarrega
    # o parser e apaga o ajuste.
    #
    # LIMITAÇÃO CONHECIDA E ACEITA (decisão do Mestre, instrução 20):
    # o init_df() em si continua carregando o modelo no dispositivo que a
    # biblioteca escolher. Consequência: numa GPU que FALHE, o
    # DeepFilterNet3 NÃO consegue reiniciar em CPU, porque o init_df() do
    # retry volta a tentar CUDA. Os outros quatro modelos têm fallback
    # pleno. Resolver isso exigiria uma cópia do modelo dentro do projeto
    # com o config.ini editado - custo não justificado hoje.
    "device": "gpu",
    
    # Liga/desliga o pós-filtro do DeepFilterNet.
    # O parâmetro da biblioteca é BOOLEANO (assinatura verificada no build
    # instalado, DeepFilterNet 0.5.6: post_filter: bool = False):
    #   0 → desliga o pós-filtro
    #   qualquer outro valor → liga
    #
    # NÃO existem três graus de agressividade nesta versão: o valor 2 é
    # idêntico ao 1. O pós-filtro faz uma redução extra e menor de ruído.
    "post_filter": 1,
    
    # Limite de atenuação de ruído, em DECIBÉIS - NÃO é fração de 0 a 1.
    # O valor viaja para o parâmetro `atten_lim_db` do enhance(), que mistura
    # sinal original e sinal tratado assim:
    #
    #     lim = 10^(-|dB|/20)
    #     saída = original * lim + tratado * (1 - lim)
    #
    # Ou seja, a fração do denoising que chega ao áudio é (1 - lim), e
    # valor MAIOR em decibéis significa MAIS denoising - a escala é o
    # inverso do que o comentário antigo descrevia.
    #
    # MEDIDO NESTE PROJETO (tarefa 19, 6 valores x 6 segmentos):
    #    0.95 dB -> aplica  10,4 % do denoising (praticamente inerte)
    #    3.0  dB -> aplica  29,2 %
    #    6.0  dB -> aplica  49,9 %
    #   12.0  dB -> aplica  74,9 %
    #   24.0  dB -> aplica  93,7 %
    #   40.0  dB -> aplica  99,0 %
    #
    # ABAIXO de 6 dB: região inerte - o módulo carrega, roda e quase não
    # altera o áudio (com 0.95 dB a diferença medida ficou 51 dB abaixo do
    # sinal, inaudível).
    # ACIMA de 24 dB: saturação - de 24 para 40 dB o ganho é de 0,5 dB.
    # Equivale a não ter limite nenhum.
    # 0 ou None: sem limite, denoising integral.
    #
    # FAIXA ÚTIL: 6 a 24 dB. Valor de trabalho: 12.0 (75 % do denoising,
    # conservador o bastante para não maltratar a voz num dataset de TTS).
    # A duração do áudio NÃO muda em nenhum valor testado - a invariante
    # de duração não é afetada por este campo.
    "attenuation_limit": 12.0,
    
    # Pular segmentos já processados anteriormente
    # Verifica o campo "utilizou_denoiser" no JSON dinâmico
    #   True → Ignora áudios com flag existente (economiza processamento)
    #   False → Reprocessa todos os áudios elegíveis (sobrescreve outputs)
    # 
    # Caso de uso True: Re-execuções após falhas/interrupções
    # Caso de uso False: Mudança de parâmetros (post_filter, attenuation_limit)
    # 
    # IMPORTANTE: Flag apenas previne reprocessamento, não valida qualidade
    "skip_if_already_processed": True,
}

# ============================================================
# MÓDULO 10: NORMALIZADOR DE ÁUDIO SOX
# ============================================================
SOX_NORMALIZER = {
    
    # Taxa de amostragem (sample rate) do áudio de saída
    # Valores comuns: 8000, 16000, 22050, 44100, 48000 (em Hz)
    # Exemplos de uso por aplicação:
    #   8000  → Telefonia (qualidade mínima)
    #   16000 → STT (Speech-to-Text) -BalanceIO qualidade/performance
    #   22050 → TTS (Text-to-Speech) - Qualidade intermediária
    #   44100 → CD quality - Uso geral alta fidelidade
    #   48000 → Professional audio/broadcasting
    # 
    # Valor atual: 24000 Hz, escolha do projeto para TTS. Está na lista de
    # valores comuns e não contradiz nada - modelos STT/TTS costumam
    # esperar 16000, 22050 ou 24000 Hz.
    # Valores mais altos = maior qualidade mas maior custo computacional
    "sample_rate": 24000,

    # Profundidade de bits (bit depth) do áudio
    # Valores: 16, 24, 32 (em bits)
    #   16 → Padrão para STT/TTS (suficiente para fala)
    #   24 → Maior dinâmica (uso profissional)
    #   32 → Máxima qualidade (raramente necessário para IA)
    # 
    # Recomendação: 16 bits para datasets de treino IA
    # Valores maiores aumentam tamanho sem ganho significativo
    "bit_depth": 16,
    
    # Número de canais de áudio
    # Valores: 1 (mono), 2 (stereo)
    #   1 → OBRIGATÓRIO para STT/TTS (modelos esperam mono)
    #   2 → Apenas se necessário preservar espacialização
    # 
    # ATENCAO - CRITICO: Praticamente todos modelos STT/TTS requerem MONO
    # Stereo aumenta tamanho 2x sem benefício para treino
    "channels": 1,
    
    # Formato do arquivo de áudio de saída
    # Opções: "wav", "flac", "mp3", "ogg"
    #   "wav"  → Sem compressão, máxima qualidade, arquivos grandes
    #   "flac" → Compressão lossless, qualidade=WAV, ~50% menor
    #   "mp3"  → Compressão lossy, qualidade OK, arquivos pequenos
    #   "ogg"  → Compressão lossy, melhor que MP3, menos compatível
    # 
    # Recomendação por caso:
    #   Treino IA → "flac" (qualidade perfeita, economia storage)
    #   Deployment → "mp3" (menor latência de carregamento)
    #   Arquivamento → "wav" (sem perdas, compatibilidade total)
    "output_format": "flac",
    
    # Método de normalização de volume
    # Opções: "peak", "rms", "loudness" - o que CADA UMA emite para o SoX
    # 14.4.2 instalado nesta máquina:
    #   "peak"     → `norm <dB>`: normaliza pelo PICO
    #   "rms"      → `gain -n <dB>`: TAMBÉM normaliza pelo PICO, não pela
    #                energia média. O `sox --help-effect=gain` do build
    #                instalado documenta `-n` como "Norm file to 0dBfs".
    #                RMS de verdade no SoX seria `gain -b` / `-B`.
    #   "loudness" → `loudness <dB>`: curva de audibilidade ISO 226 do SoX.
    #                NÃO é LUFS/EBU R128.
    #
    # Consequência prática: "peak" e "rms" entregam a mesma coisa -
    # normalização por pico, com diferenças menores de implementação. O
    # nome "rms" não descreve o que acontece.
    "normalize_method": "rms",
    
    # Nível alvo de normalização em decibéis
    # Valores típicos: -3.0, -1.0, 0.0 (em dB)
    #   -3.0 → Conservador (headroom para evitar clipping)
    #   -1.0 → Balanceado (padrão broadcasting)
    #   0.0  → Máximo (sem margem de segurança)
    # 
    # ATENCAO: Valores positivos causam distorção (clipping)
    # Valores muito negativos resultam em áudio baixo
    # Recomendação: -3.0 para datasets de treino
    "target_level_db": -3.0,
}