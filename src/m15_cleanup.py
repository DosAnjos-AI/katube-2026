#!/usr/bin/env python3
"""
Modulo m15_cleanup.py
Realiza limpeza de arquivos temporarios e/ou input conforme configuracao
"""

import sys
import shutil
from pathlib import Path

# ==============================================================================
# CONFIGURACAO DE PATHS
# ==============================================================================

# Adicionar pasta raiz ao path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Importar configuracoes
from config import MASTER

# ==============================================================================
# FUNCOES DE LIMPEZA
# ==============================================================================

def excluir_pasta(pasta: Path, tipo: str) -> bool:
    """
    Exclui pasta e todo seu conteudo
    
    Args:
        pasta: Path da pasta a excluir
        tipo: Tipo de pasta ('temp' ou 'input') para logs
        
    Returns:
        True se excluiu com sucesso, False caso contrario
    """
    if not pasta.exists():
        print(f"[AVISO] Pasta {tipo} nao encontrada: {pasta}")
        return False
    
    try:
        shutil.rmtree(pasta)
        print(f"[OK] Pasta {tipo} excluida: {pasta}")
        return True
    except Exception as e:
        print(f"[ERRO] Falha ao excluir pasta {tipo}: {e}")
        return False

# ==============================================================================
# FUNCAO PRINCIPAL DE CLEANUP
# ==============================================================================

def executar_cleanup(audio_id: str) -> bool:
    """
    Executa limpeza conforme configuracao MASTER['cleanup'].

    Args:
        audio_id: ID do audio a processar

    Returns:
        True se a limpeza pedida foi concluida (ou estava desabilitada),
        False se alguma pasta nao pode ser removida ou o modo e invalido.
        Falha aqui e AVISO para o chamador: o dataset ja esta gravado.
    """
    # Definir caminhos baseados no audio_id
    PASTA_TEMP = PROJECT_ROOT / "arquivos" / "temp" / audio_id
    PASTA_INPUT = PROJECT_ROOT / "arquivos" / "audios" / audio_id

    modo = MASTER.get('cleanup', 'none')

    print(f"\n{'='*70}")
    print(f"INICIANDO CLEANUP - Modo: {modo}")
    print(f"ID do audio: {audio_id}")
    print(f"{'='*70}\n")
    
    if modo == 'none':
        print("[INFO] Cleanup desabilitado (mode='none')")
        return True

    # Executar limpeza conforme modo
    if modo == 'all':
        print("[INFO] Modo 'all': Excluindo input e temporarios")
        ok_input = excluir_pasta(PASTA_INPUT, 'input')
        ok_temp = excluir_pasta(PASTA_TEMP, 'temp')
        sucesso = ok_input and ok_temp

    elif modo == 'input':
        print("[INFO] Modo 'input': Excluindo apenas arquivos de entrada")
        sucesso = excluir_pasta(PASTA_INPUT, 'input')

    elif modo == 'temp':
        print("[INFO] Modo 'temp': Excluindo apenas arquivos temporarios")
        sucesso = excluir_pasta(PASTA_TEMP, 'temp')

    else:
        print(f"[ERRO] Modo de cleanup invalido: {modo}")
        print("[INFO] Valores validos: 'all', 'input', 'temp', 'none'")
        sucesso = False

    print(f"\n{'='*70}")
    print("CLEANUP FINALIZADO")
    print(f"{'='*70}\n")

    return sucesso

# ==============================================================================
# EXECUCAO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    # Guarda de execucao direta: este modulo APAGA pastas reais.
    # Exige audio_id como argumento (sem id fixo no codigo) e confirmacao
    # digitada. Nada disso roda quando o main.py chama executar_cleanup().
    if len(sys.argv) != 2:
        print("Uso: python src/m15_cleanup.py <audio_id>")
        sys.exit(1)

    audio_id_alvo = sys.argv[1]
    print("ATENCAO: cleanup em execucao direta apaga as pastas abaixo")
    print(f"  temp : {PROJECT_ROOT / 'arquivos' / 'temp' / audio_id_alvo}")
    print(f"  input: {PROJECT_ROOT / 'arquivos' / 'audios' / audio_id_alvo}")
    print(f"  modo configurado em MASTER['cleanup']: {MASTER.get('cleanup', 'none')}")

    if input("Confirma? (digite SIM): ") != "SIM":
        print("Cancelado - nada foi apagado")
        sys.exit(1)

    executar_cleanup(audio_id_alvo)