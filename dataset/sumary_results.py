#!/usr/bin/env python3
"""
Analise de segmentos do dataset
Uso: python result_segments.py
"""

import pandas as pd
from pathlib import Path
import sys


def analisar_dataset(csv_path: str):
    """
    Analisa estatisticas de segmentos de um dataset CSV.
    
    Args:
        csv_path: Caminho para o arquivo CSV
    """
    try:
        # Carregar CSV
        df = pd.read_csv(csv_path, sep='|')
        
        # Extrair coluna de tamanho_segmento
        df_seg = df['duracao']
        
        # Calcular estatisticas
        duracao_hrs = df_seg.sum() / 3600
        media = df_seg.mean()
        contagem = df_seg.count()
        seg_max = df_seg.max()
        seg_min = df_seg.min()
        
        # Exibir resultado

        print(f"  duracao (hrs): {duracao_hrs:.2f}")
        print(f"  media: {media:.2f}")
        print(f"  contagem: {contagem:.0f}")
        print(f"  seg_max: {seg_max:.2f}")
        print(f"  seg_min: {seg_min:.2f}")
        
    except FileNotFoundError:
        print(f"Erro: Arquivo nao encontrado: {csv_path}")
        sys.exit(1)
    except KeyError:
        print(f"Erro: Coluna 'tamanho_segmento' nao encontrada no CSV")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao processar {csv_path}: {e}")
        sys.exit(1)


def main():
    analisar_dataset(
        csv_path="dataset.csv")


if __name__ == "__main__":
    main()