#!/bin/bash
set -e

# inicializa o conda no contexto do script
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate katube-2026
echo "Ambiente conda ativado: $CONDA_DEFAULT_ENV"

DS="/home/ubuntu/Dataset_Spotify_Processado"
SP="/home/ubuntu/spotify2026"

processar_iteracao() {
    local N=$1
    echo ""
    echo "=========================================="
    echo "Iniciando iteracao $N"
    echo "=========================================="

    # --- ETAPA 1 - PRE-PROCESSAMENTO ---
    echo "[$N] Etapa 1: pre-processamento e ingestao dos dados"
    cd "$DS"
    python nomecao_spotify${N}.py

    mkdir -p "$DS/$N"

    cp -f "$DS/relacao_ids.csv" "$DS/$N/"
    cp -f "$DS/relacao_ids.csv" "$SP/dataset/"

    echo "[$N] Etapa 1: rodando main.py (aguardando finalizacao...)"
    cd "$SP"
    date > tempo.txt
    python main.py > output.log 2>&1
    date >> tempo.txt
    echo "[$N] main.py finalizado"

    # --- ETAPA 2 - LIMPEZA E SALVAMENTO ---
    echo "[$N] Etapa 2: salvando resultados e limpando ambiente"

    rm -rf "$SP/arquivos/audios/"
    mkdir -p "$SP/arquivos/audios/"

    rm -rf "$SP/arquivos/temp/"
    mkdir -p "$SP/arquivos/temp/"

    mv "$SP/dataset/" "$DS/$N/"

    mkdir -p "$SP/dataset/audio_dataset"
    mkdir -p "$SP/dataset/historico_dataset"
    mkdir -p "$SP/dataset/log"

    mv -f "$SP/output.log" "$DS/$N/"
    mv -f "$SP/tempo.txt"  "$DS/$N/"

    cp -rf "$DS/$N/dataset/sumary_results.py" "$SP/dataset/"

    echo "[$N] Iteracao $N concluida com sucesso."
}

for N in 14 15; do
    processar_iteracao $N
done

echo ""
echo "Pipeline completo: iteracoes 14 e 15 finalizadas."