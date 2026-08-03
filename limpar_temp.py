#!/usr/bin/env python3
"""
limpar_temp.py - libera a entrada dos audios que ja tem estado em temp/

Para cada pasta encontrada em arquivos/temp/, remove a pasta de mesmo nome
em arquivos/audios/. E o passo que o executor.sh roda antes de cada
main.py, para que a fila de entrada nao reprocesse o que ja entrou.

ESTE SCRIPT APAGA PASTAS. Por isso:
  - nada roda no import: tudo vive sob a guarda de __main__;
  - a raiz do projeto e derivada deste arquivo, nunca escrita como
    caminho absoluto (antes era /home/ubuntu/..., que apontava para fora
    do projeto em qualquer usuario diferente de 'ubuntu');
  - todo alvo passa por guarda de caminho antes de ser removido: precisa
    resolver para dentro de arquivos/audios/ e ter nome valido.
"""

import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIOS_DIR = PROJECT_ROOT / "arquivos" / "audios"
TEMP_DIR = PROJECT_ROOT / "arquivos" / "temp"


def nome_valido(nome: str) -> bool:
    """Recusa nome vazio, '.', '..' ou com separador de caminho."""
    if not nome or nome in ('.', '..'):
        return False
    return '/' not in nome and '\\' not in nome


def alvo_seguro(alvo: Path, raiz: Path) -> bool:
    """
    Guarda de caminho: o alvo tem de resolver para dentro da raiz e nao
    pode ser a propria raiz.
    """
    try:
        alvo_resolvido = alvo.resolve()
        raiz_resolvida = raiz.resolve()
    except OSError:
        return False
    return alvo_resolvido != raiz_resolvida and raiz_resolvida in alvo_resolvido.parents


def limpar() -> int:
    """
    Remove de arquivos/audios/ as pastas que ja tem estado em
    arquivos/temp/.

    Returns:
        0 se concluiu, 1 se faltou pre-condicao (pasta ausente).
    """
    if not TEMP_DIR.is_dir():
        print(f"ERRO: pasta temp nao encontrada: {TEMP_DIR}")
        return 1

    if not AUDIOS_DIR.is_dir():
        print(f"ERRO: pasta audios nao encontrada: {AUDIOS_DIR}")
        return 1

    temp_pastas = {p.name for p in TEMP_DIR.iterdir() if p.is_dir()}
    audios_antes = sum(1 for _ in AUDIOS_DIR.iterdir())

    print(f"Pastas em temp:   {len(temp_pastas)}")
    print(f"Pastas em audios: {audios_antes}")
    print("Iniciando limpeza...\n")

    removidas = 0
    recusadas = 0

    for nome in sorted(temp_pastas):
        if not nome_valido(nome):
            print(f"RECUSADO: nome invalido em temp: {nome!r}")
            recusadas += 1
            continue

        alvo = AUDIOS_DIR / nome

        if not alvo.is_dir():
            continue

        if not alvo_seguro(alvo, AUDIOS_DIR):
            print(f"RECUSADO: alvo fora de {AUDIOS_DIR}: {alvo}")
            recusadas += 1
            continue

        shutil.rmtree(alvo)
        print(f"Removida: {alvo}")
        removidas += 1

    print(f"\nPastas removidas:       {removidas}")
    print(f"Pastas recusadas:       {recusadas}")
    print(f"Pastas em audios agora: {sum(1 for _ in AUDIOS_DIR.iterdir())}")

    return 0


if __name__ == "__main__":
    sys.exit(limpar())
