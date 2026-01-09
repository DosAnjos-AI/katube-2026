from pathlib import Path

# ============================================================================
# DIRETÓRIO RAIZ DO PROJETO
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()


# ============================================================================
# BLOCO MASTER - Controle de Módulos Ativos
# ============================================================================
# Ativa/desativa módulos principais do sistema
# True = módulo será executado | False = módulo será ignorado
MASTER = {
    'downloader': True,      # Módulo de download de áudios do YouTube
    'transcricao': False,    # [FUTURO] Módulo de transcrição
    'processamento': False,  # [FUTURO] Módulo de processamento
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
        'sample_rate_hz': 24000,
    },
    
    # ------------------------------------------------------------------------
    # Legendas (OBRIGATÓRIAS)
    # ------------------------------------------------------------------------
    'legendas': {
        # Idiomas de legenda aceitos (ordem de prioridade)
        # Primeiro tenta "pt-BR", depois "pt"
        # IMPORTANTE: O vídeo só é REJEITADO se não encontrar NENHUMA legenda
        # (nem manual nem automática, caso automáticas estejam ativadas)
        'idiomas': ['pt-BR', 'pt'],
        
        # Aceitar legendas automáticas (auto-geradas pelo YouTube)
        # True = aceita legendas automáticas se manuais não existirem
        # False = aceita APENAS legendas criadas manualmente
        # ATENÇÃO: Se False e sem legenda manual → vídeo é REJEITADO
        'aceitar_automaticas': True,
    },
    
    # ------------------------------------------------------------------------
    # Filtros de Duração
    # ------------------------------------------------------------------------
    'duracao': {
        # Duração mínima do vídeo em segundos
        # Vídeos mais curtos serão rejeitados
        # Exemplo: 30 = rejeita vídeos com menos de 30 segundos
        'minima_segundos': 60,
        
        # Duração máxima do vídeo em segundos
        # Vídeos mais longos serão rejeitados
        # Exemplo: 3600 = rejeita vídeos com mais de 1 hora (60 minutos)
        'maxima_segundos': 3600,
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
            'maximo_segundos': 6,  # Recomendado máximo: 10-15s
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