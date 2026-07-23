#!/usr/bin/env python3
"""
Analise de segmentos do dataset
Uso: python result_segments.py
"""

import pandas as pd
import subprocess
from pathlib import Path
import sys


def contar_arquivos(diretorio: str) -> int:
    """Conta arquivos em um diretório (equivalente a ls | wc -l)."""
    result = subprocess.run(
        f"ls {diretorio} | wc -l",
        shell=True,
        capture_output=True,
        text=True
    )
    return int(result.stdout.strip())


def contar_linhas(diretorio: str) -> int:
    result = subprocess.run(
        f"cat {diretorio} | wc -l",
        shell=True,
        capture_output=True,
        text=True
    )
    return int(result.stdout.strip())


def analisar_dataset(csv_path: str):
    try:
        df = pd.read_csv(csv_path, sep='|')

        df_seg = df['duracao']
        df_segmentos = df['arquivo_nome']

        duracao_hrs = df_seg.sum() / 3600
        media = df_seg.mean()
        contagem = df_seg.count()
        seg_max = df_seg.max()
        seg_min = df_seg.min()
        unique = set(df_segmentos)

        total_audios = contar_arquivos("/home/ubuntu/spotify2026/arquivos/audios")
        total_temp   = contar_arquivos("/home/ubuntu/spotify2026/arquivos/temp")
        status = contar_linhas("/home/ubuntu/spotify2026/tempo.txt")

        # Corrigido: if/elif/else para evitar sobrescrita do valor
        if status == 1:
            situacao = "em execução"
        elif status == 2:
            situacao = "finalizado"
        else:
            situacao = "não determinado"

        print(f"  duracao (hrs): {duracao_hrs:.2f}")
        print(f"  media: {media:.2f}")
        print(f"  contagem: {contagem:.0f}")
        print(f"  seg_max: {seg_max:.2f}")
        print(f"  seg_min: {seg_min:.2f}")
        print(f"  segmentos de audios unicos: {len(unique)}")
        print(f"  quantidade de segmentos repetidos: {contagem - len(unique)}")
        print(f"  arquivos_audios: {total_audios}")
        print(f"  arquivos_temp: {total_temp}")
        print(f"  em_execucao: {situacao}")

    except FileNotFoundError:
        print(f"Erro: Arquivo nao encontrado: {csv_path}")
        sys.exit(1)
    except KeyError:
        print(f"Erro: Coluna nao encontrada no CSV")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao processar {csv_path}: {e}")
        sys.exit(1)


def main():
    analisar_dataset(csv_path="dataset.csv")


if __name__ == "__main__":
    main()