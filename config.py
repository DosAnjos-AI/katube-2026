from pathlib import Path

# ============================================================================
# DIRETÓRIO RAIZ DO PROJETO
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent


# ============================================================================
# BLOCO MASTER - Controle de Módulos Ativos
# ============================================================================
# Ativa/desativa módulos principais do sistema
# True = módulo será executado | False = módulo será ignorado
MASTER = {
    'downloader': True,        # Módulo de download de áudios do YouTube
    'segmentacao': 'vad',  # opções: 'legenda', 'vad', '' para não usar segmentador audio já segmentado.
    'transcricao': False,      # [FUTURO] Módulo de transcrição
    'processamento': False,    # [FUTURO] Módulo de processamento
}


# ============================================================================
# BLOCO 00 - DOWNLOADER
# ============================================================================
# Configurações do módulo de download de áudios do YouTube
DOWNLOADER = {
    
    # ------------------------------------------------------------------------
    # Controle de Quantidade
    # ------------------------------------------------------------------------
    'quantidade': {
        # Limite de vídeos por playlist/canal
        # 0 = baixa todos | N = baixa no máximo N vídeos de cada fonte
        # Exemplo: limit=5 em playlist(100) → baixa apenas 5 vídeos
        'limit': 0,
    },
    
    # ------------------------------------------------------------------------
    # Qualidade do Áudio
    # ------------------------------------------------------------------------
    'audio': {
        # Formato de saída do áudio
        # Opções disponíveis: "mp3", "wav", "m4a", "flac", "opus"
        # Recomendado: "flac" (melhor qualidade) ou "mp3" (menor tamanho)
        'formato': 'flac',
        
        # Taxa de bits (bitrate) em kbps
        # 0 = melhor qualidade disponível (recomendado)
        # Valores comuns: 128, 192, 256, 320
        # Nota: "flac" e "wav" ignoram esta configuração (sem perda)
        'bitrate_kbps': 0,
        
        # Taxa de amostragem (sample rate) em Hz
        # 0 = usa a taxa original do vídeo (recomendado)
        # Valores comuns: 44100 (CD quality), 48000 (padrão YouTube), 96000 (high-res)
        # Nota: valores maiores = melhor qualidade, mas arquivos maiores
        'sample_rate_hz': 0,
    },
    
    # ------------------------------------------------------------------------
    # Legendas (OBRIGATÓRIAS)
    # ------------------------------------------------------------------------
    'legendas': {
        # Lista de prioridades de legendas (ordem = prioridade de tentativa)
        # Formato: 'idioma-tipo' onde tipo pode ser 'manual' ou 'auto'
        # O sistema tentará baixar na ordem da lista até encontrar uma disponível
        # Se nenhuma legenda da lista for encontrada, o vídeo é REJEITADO
        #
        # Opções disponíveis (exemplos):
        # 'pt-BR-manual'  - Português Brasil (legenda manual/criada por humano)
        # 'pt-BR-auto'    - Português Brasil (legenda automática/gerada pelo YouTube)
        # 'pt-manual'     - Português genérico (legenda manual)
        # 'pt-auto'       - Português genérico (legenda automática)
        # 'pt-PT-manual'  - Português Portugal (legenda manual)
        # 'pt-PT-auto'    - Português Portugal (legenda automática)
        # 'en-manual'     - Inglês (legenda manual)
        # 'en-auto'       - Inglês (legenda automática)
        # 'es-manual'     - Espanhol (legenda manual)
        # 'es-auto'       - Espanhol (legenda automática)
        # 'fr-manual'     - Francês (legenda manual)
        # 'fr-auto'       - Francês (legenda automática)
        # ... [qualquer código de idioma ISO 639-1 ou variante regional]
        #
        # Dica: Para aceitar apenas legendas manuais, remova as opções '-auto'
        #       Para priorizar automáticas, coloque '-auto' antes de '-manual'
        'prioridade': [
            'pt-BR-manual',
            'pt-BR-auto',
            'pt-manual',
            'pt-auto',
        ],
    },
    
    # ------------------------------------------------------------------------
    # Filtros de Seleção de Vídeos
    # ------------------------------------------------------------------------
    'filtros': {
        # Filtros de duração do vídeo
        'duracao': {
            # Duração mínima do vídeo em segundos
            # Vídeos mais curtos serão rejeitados
            # Exemplo: 60 = rejeita vídeos com menos de 1 minuto
            'minima_segundos': 60,
            
            # Duração máxima do vídeo em segundos
            # Vídeos mais longos serão rejeitados
            # Exemplo: 3600 = rejeita vídeos com mais de 1 hora (60 minutos)
            'maxima_segundos': 3600,
        },
        
        # Filtros de data de upload (formato: DD-MM-AAAA)
        'data_upload': {
            # Data mínima de upload
            # '0' = sem filtro mínimo (aceita qualquer data antiga)
            # Formato: 'DD-MM-AAAA'
            # Exemplos: '01-01-2024' (apenas vídeos de 2024 em diante)
            #           '15-06-2023' (vídeos de 15/06/2023 em diante)
            #           '0' (sem filtro, aceita todos)
            'minima': '12-12-2020',
            
            # Data máxima de upload
            # '0' = sem filtro máximo (aceita até hoje)
            # Formato: 'DD-MM-AAAA'
            # Exemplos: '31-12-2024' (apenas vídeos até 31/12/2024)
            #           '30-09-2025' (vídeos até 30/09/2025)
            #           '0' (sem filtro, aceita todos)
            'maxima': '0',
        },
    },
    
    # ------------------------------------------------------------------------
    # Anti-Bloqueio (Delays entre Downloads)
    # ------------------------------------------------------------------------
    'delays': {
        # Delay entre LINKS DO CSV (vídeos individuais, playlists, canais)
        # O tempo real será aleatório entre mínimo e máximo (valores contínuos)
        'entre_links_csv': {
            'minimo_segundos': 5,   # Recomendado mínimo: 5s
            'maximo_segundos': 20,  # Recomendado máximo: 20-30s
        },
        
        # Delay entre VÍDEOS DENTRO de playlists/canais (links compostos)
        # Aplica-se apenas quando baixando múltiplos vídeos de uma mesma fonte
        'entre_videos_playlist': {
            'minimo_segundos': 3,   # Recomendado mínimo: 3s
            'maximo_segundos': 10,  # Recomendado máximo: 10-15s
        },
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # Sobrescrever arquivos existentes
        # False = pula vídeos já baixados (verifica se pasta {video_id} existe)
        # True = re-baixa mesmo que já exista
        'sobrescrever': False,
    },
}


# =============================================================================
# MÓDULO 01: SEGMENTADOR DE ÁUDIO (BASEADO EM LEGENDAS)
# =============================================================================

# Configurações do módulo de segmentação de áudio usando legendas
# Utilizado quando MASTER['segmentacao'] = 'legenda'
SEGMENTADOR_AUDIO = {
    
    # ------------------------------------------------------------------------
    # Controle do tamanho dos segmentos criados
    # ------------------------------------------------------------------------
    # min_seg: Duração mínima de cada segmento em segundos
    # - Segmentos menores serão agrupados com próximos (respeitando locutores)
    # - Tolerância: aceita até 0.8s a menos
    'min_seg': 12,
    
    # max_seg: Duração máxima de cada segmento em segundos
    # - Segmentos não ultrapassam este limite
    # - Tolerância: aceita até 0.8s a mais
    'max_seg': 25,
}


# =============================================================================
# MÓDULO 01: SEGMENTADOR DE ÁUDIO VAD (VOICE ACTIVITY DETECTION)
# =============================================================================

# Configurações do módulo de segmentação automática usando detecção de voz
# Utilizado quando MASTER['segmentacao'] = 'vad'
# 
# Este módulo utiliza Silero-VAD para detectar automaticamente momentos de
# fala e silêncio no áudio, criando segmentos baseados em pausas naturais
# (sem depender de legendas).
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
        # Exemplo: 30 = adiciona 30ms antes do início detectado da fala
        'inicio_ms': 30,
        
        # Tempo adicional no FIM de cada segmento (milissegundos)
        # Evita cortar o final da última palavra
        # - Valores baixos (10-30ms): corte mais preciso
        # - Valores médios (30-50ms): segurança recomendada
        # - Valores altos (>100ms): pode incluir silêncio extra
        # Exemplo: 30 = adiciona 30ms após o fim detectado da fala
        'fim_ms': 30,
    },
    
    # ------------------------------------------------------------------------
    # Limites de Duração dos Segmentos Finais
    # ------------------------------------------------------------------------
    'segmentos': {
        # Duração MÍNIMA de cada segmento em segundos
        # Segmentos mais curtos são agrupados com próximos
        # - Valores baixos (2-4s): aceita segmentos muito curtos
        # - Valores médios (4-8s): equilíbrio (recomendado)
        # - Valores altos (>10s): força segmentos longos
        # Exemplo: 4.0 = todos os segmentos terão no mínimo 4 segundos
        'min_seg': 4.0,
        
        # Duração MÁXIMA de cada segmento em segundos
        # Segmentos mais longos são divididos em pausas naturais
        # - Valores baixos (8-12s): força segmentos curtos
        # - Valores médios (15-20s): equilíbrio (recomendado)
        # - Valores altos (>25s): permite segmentos muito longos
        # Exemplo: 15.0 = nenhum segmento ultrapassará 15 segundos
        'max_seg': 15.0,
        
        # Tolerância nas durações (segundos)
        # Permite pequenas variações nos limites min/max
        # - Valores baixos (0.3-0.5s): mais rigoroso
        # - Valores médios (0.8-1.0s): equilíbrio (recomendado)
        # - Valores altos (>1.5s): mais flexível
        # Exemplo: 0.8 = aceita segmento de 3.2s (min=4.0 - tolerância=0.8)
        'tolerancia': 0.8,
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # Sobrescrever segmentos existentes
        # False = pula áudios já segmentados (verifica pasta segments/)
        # True = re-segmenta mesmo que já exista
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
    # Opções disponíveis:
    # - "auto": Detecta automaticamente (GPU se disponível, senão CPU)
    # - "gpu": Força uso de GPU/CUDA (falha se GPU não disponível)
    # - "cpu": Força uso de CPU (mais lento, mas funciona em qualquer máquina)
    # 
    # Recomendação: "auto" para máxima compatibilidade
    # Nota: GPU acelera significativamente (3-5x mais rápido)
    'device': 'auto',
    
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
        'min_threshold': 2.5,
        
        # Limiar para alta qualidade
        # Áudios com MOS >= max_threshold são considerados ÓTIMOS
        # Não precisam de denoising posterior
        # Valores típicos: 3.0-4.0
        # Exemplo: 3.5 = áudios acima disso vão direto pro dataset
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
        # - "auto": Calcula automaticamente baseado em VRAM disponível
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
        # Sobrescrever análises existentes
        # False = pula segmentos já analisados (verifica se JSON existe)
        # True = re-analisa todos os segmentos
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
    # Opções disponíveis:
    # - "auto": Detecta automaticamente (GPU se disponível, senão CPU)
    # - "gpu": Força uso de GPU/CUDA (falha se GPU não disponível)
    # - "cpu": Força uso de CPU (mais lento, mas funciona em qualquer máquina)
    # 
    # Recomendação: "auto" para máxima compatibilidade
    # Nota: GPU acelera significativamente o processamento
    'device': 'auto',
    
    # ------------------------------------------------------------------------
    # Modelo de Diarização
    # ------------------------------------------------------------------------
    # Modelo HuggingFace para detecção de overlap
    # pyannote/speaker-diarization-community-1CC-BY-4.0
    # Licença: CC-BY-4.0 (permissiva para uso acadêmico)
    # IMPORTANTE: Requer token HuggingFace configurado em .env
    'modelo': 'pyannote/speaker-diarization-3.1',
    
    # ------------------------------------------------------------------------
    # Batch Processing (Processamento em Lote)
    # ------------------------------------------------------------------------
    # Processa múltiplos áudios simultaneamente para maior eficiência
    
    'batch': {
        # Tamanho do batch (quantos áudios processar juntos)
        # Valores maiores = mais rápido, mas usa mais VRAM
        # 
        # Opções:
        # - "auto": Calcula automaticamente baseado em VRAM disponível
        # - 1-16: Valor fixo (números maiores exigem mais VRAM)
        # 
        # Referência de uso de VRAM (aproximado):
        # - batch_size=1:  ~3.0 GB
        # - batch_size=4:  ~5.0 GB
        # - batch_size=8:  ~8.0 GB
        # - batch_size=16: ~12.0 GB
        # 
        # Recomendação:
        # - GPU com 24GB: "auto" ou 16
        # - GPU com 8-16GB: 8
        # - GPU com 4-8GB: 4
        # - CPU: 1
        'batch_size': 'auto',
    },
    'timeout': {
    'por_audio_segundos': 60,  # Timeout máximo por áudio
    },
    
    # ------------------------------------------------------------------------
    # Comportamento Geral
    # ------------------------------------------------------------------------
    'comportamento': {
        # Sobrescrever análises existentes
        # False = pula segmentos já analisados (verifica se campo overlap01 existe)
        # True = re-analisa todos os segmentos
        'sobrescrever': False,
    },
}