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
# CONFIGURACAO DE INPUTS/OUTPUTS
# ==============================================================================

# ID do video a processar
id_video = 'QN7gUP7nYhQ'

# Caminhos para limpeza
PASTA_TEMP = PROJECT_ROOT / "arquivos" / "temp" / id_video
PASTA_INPUT = PROJECT_ROOT / "arquivos" / "audios" / id_video

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

def executar_cleanup():
    """Executa limpeza conforme configuracao MASTER['cleanup']"""
    modo = MASTER.get('cleanup', 'none')
    
    print(f"\n{'='*70}")
    print(f"INICIANDO CLEANUP - Modo: {modo}")
    print(f"{'='*70}\n")
    
    if modo == 'none':
        print("[INFO] Cleanup desabilitado (mode='none')")
        return
    
    # Executar limpeza conforme modo
    if modo == 'all':
        print("[INFO] Modo 'all': Excluindo input e temporarios")
        excluir_pasta(PASTA_INPUT, 'input')
        excluir_pasta(PASTA_TEMP, 'temp')
        
    elif modo == 'input':
        print("[INFO] Modo 'input': Excluindo apenas arquivos de entrada")
        excluir_pasta(PASTA_INPUT, 'input')
        
    elif modo == 'temp':
        print("[INFO] Modo 'temp': Excluindo apenas arquivos temporarios")
        excluir_pasta(PASTA_TEMP, 'temp')
        
    else:
        print(f"[ERRO] Modo de cleanup invalido: {modo}")
        print("[INFO] Valores validos: 'all', 'input', 'temp', 'none'")
    
    print(f"\n{'='*70}")
    print("CLEANUP FINALIZADO")
    print(f"{'='*70}\n")

# ==============================================================================
# EXECUCAO PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    executar_cleanup()