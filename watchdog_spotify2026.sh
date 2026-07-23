#!/bin/bash
# ══════════════════════════════════════════════════════════════════
#  WATCHDOG — spotify2026
#  Reexecuta main.py até que todo o processamento esteja concluído
#  Condição de parada: audios (A) == temp (B)
# ══════════════════════════════════════════════════════════════════

PROJECT_DIR="/home/ubuntu/spotify2026"
AUDIOS_DIR="$PROJECT_DIR/arquivos/audios"
TEMP_DIR="$PROJECT_DIR/arquivos/temp"
LOCKFILE="/tmp/watchdog_spotify2026.lock"

# ── Proteção contra múltiplas instâncias ──────────────────────────
if [ -f "$LOCKFILE" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  Watchdog já está rodando (lock existe: $LOCKFILE). Abortando."
    exit 1
fi

touch "$LOCKFILE"
trap "rm -f $LOCKFILE; echo '[$(date '+%Y-%m-%d %H:%M:%S')] Lock removido.'" EXIT
# ──────────────────────────────────────────────────────────────────

# Ativa o ambiente conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate katube-2026

cd "$PROJECT_DIR"

while true; do

    A=$(ls "$AUDIOS_DIR" | wc -l)
    B=$(ls "$TEMP_DIR"   | wc -l)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] audios=$A | temp=$B"

    # Condição de parada legítima
    if [ "$A" -le "$B" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✅ Processamento concluído (A=$A == B=$B). Encerrando."
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Iniciando main.py..."

    # Execução bloqueante — só avança quando o processo terminar
    bash -c "date > tempo.txt; python main.py > output.log 2>&1; date >> tempo.txt"

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] main.py encerrou. Rechecando condição..."

done
