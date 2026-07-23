#!/usr/bin/env bash
#
# executor.sh — automacao completa da pipeline spotify2026 (KISS)
#
# Processa, em loop, todos os lotes de ~/copias_spotify/dados/.
# Para cada lote:
#   1. move o conteudo do lote  -> arquivos/audios/  (sem estourar ARG_MAX)
#   2. apaga a pasta-lote vazia (rmdir = confirmacao do move)
#   3. processa ate audios/ esvaziar (limpar_temp -> main.py, em loop)
#   4. arquiva a saida em ~/Dataset_Spotify_Processado/<lote>/
#   5. recria o ambiente de trabalho vazio pro proximo lote
#
# Execucao BLOQUEANTE: nunca ha duas execucoes do main.py ao mesmo tempo.
# Erro silencioso tratado: a unica verdade e a contagem de itens em audios/.
#
# --- USO (sobrevive ao logout) ---
#   chmod +x executor.sh
#   nohup ./executor.sh > supervisor.log 2>&1 &
#   tail -f supervisor.log
#
# --- PARAR ---
#   pkill -f executor.sh
#   pkill -f "python main.py"

# ---------- caminhos ----------
PROJ_DIR="$HOME/spotify2026"
DADOS_DIR="$HOME/copias_spotify/dados"
AUDIOS_DIR="$PROJ_DIR/arquivos/audios"
DEST_DIR="$HOME/Dataset_Spotify_Processado"
MARCADOR="$PROJ_DIR/pasta_atual.txt"
CONDA_ENV="katube-2026"

# ---------- ancoragem do diretorio (roda sempre a partir da raiz) ----------
cd "$PROJ_DIR" || { echo "ERRO: nao encontrei $PROJ_DIR"; exit 1; }

# ---------- ativa o conda (1x, antes do loop) ----------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "ERRO: falha ao ativar env $CONDA_ENV"; exit 1; }

# ==================== LOOP POR LOTE ====================
while true; do

    # pega o primeiro lote disponivel (ordem nao importa)
    LOTE_PATH=$(find "$DADOS_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -z "$LOTE_PATH" ]; then
        echo "===================================================="
        echo "[$(date '+%F %T')] FIM: nao ha mais lotes em $DADOS_DIR."
        break
    fi
    LOTE=$(basename "$LOTE_PATH")

    echo "===================================================="
    echo "[$(date '+%F %T')] LOTE '$LOTE' — iniciando"
    echo "$LOTE" > "$MARCADOR"

    # ---- 1. move o conteudo do lote -> audios/ (seguro contra ARG_MAX) ----
    echo "[$(date '+%F %T')] Movendo conteudo de '$LOTE' para audios/..."
    find "$LOTE_PATH" -mindepth 1 -maxdepth 1 -exec mv -t "$AUDIOS_DIR" {} +

    # ---- 2. apaga a pasta-lote vazia (rmdir confirma que o move terminou) ----
    if ! rmdir "$LOTE_PATH" 2>/dev/null; then
        echo "[$(date '+%F %T')] ERRO: '$LOTE' nao ficou vazia apos o move. Abortando para investigar."
        break
    fi

    # ---- 3. processamento interno: roda ate audios/ esvaziar ----
    iter=0
    while true; do
        iter=$((iter + 1))
        echo "[$(date '+%F %T')] [$LOTE] processamento iteracao $iter"

        # ACAO 01: remove os audios quebrados antes do main
        python limpar_temp.py

        # ACAO 03: pipeline em FOREGROUND (bloqueante). Exit code ignorado de proposito.
        date > tempo.txt
        python main.py > output.log 2>&1
        date >> tempo.txt

        # ACAO 04: condicao de parada real = audios/ vazia (itens sao pastas de 1o nivel)
        restantes=$(ls "$AUDIOS_DIR" | wc -l)
        echo "[$(date '+%F %T')] [$LOTE] audios restantes: $restantes"
        [ "$restantes" -eq 0 ] && break

        echo "[$(date '+%F %T')] [$LOTE] ainda ha audios -> reiniciando (provavel erro silencioso)"
        sleep 2
    done

    # ---- 4. arquiva a saida do lote ----
    echo "[$(date '+%F %T')] [$LOTE] arquivando saida em $DEST_DIR/$LOTE/"
    mkdir -p "$DEST_DIR/$LOTE"
    mv "$PROJ_DIR/dataset" "$DEST_DIR/$LOTE/"

    # ---- 5. recria o ambiente de trabalho vazio pro proximo lote ----
    mkdir -p "$PROJ_DIR/dataset/audio_dataset"
    mkdir -p "$PROJ_DIR/dataset/historico_dataset"
    mkdir -p "$PROJ_DIR/dataset/log"
    cp "$PROJ_DIR/sumary_results.py" "$PROJ_DIR/dataset/"

    # ---- limpa o marcador para o proximo lote ----
    > "$MARCADOR"
    echo "[$(date '+%F %T')] LOTE '$LOTE' — concluido."

done

echo "[$(date '+%F %T')] Todos os lotes processados. Encerrando."
