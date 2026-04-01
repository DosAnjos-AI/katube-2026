#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Processor - Processamento sequencial automatico de audios
Processa todos os audios da pasta audios/ chamando run_pipeline.py para cada um
Filosofia KISS - Simples e funcional
"""

import subprocess
import logging
from pathlib import Path
from datetime import datetime
import json
import sys
import argparse
from typing import List, Dict, Tuple


class BatchProcessor:
    """
    Processador em lote para pipeline de audio.
    Descobre automaticamente pastas em audios/ e processa sequencialmente.
    """
    
    def __init__(self, 
                 input_dir: Path = None,
                 log_dir: Path = None,
                 historico_dir: Path = None,
                 dry_run: bool = False):
        """
        Inicializa o processador em lote.
        
        Args:
            input_dir: Diretorio com pastas de audios (default: audios/)
            log_dir: Diretorio para logs (default: dataset/log/)
            historico_dir: Diretorio de historico (default: dataset/historico_dataset/)
            dry_run: Se True, apenas simula sem processar
        """
        # Diretorios (caminhos relativos)
        self.input_dir = Path(input_dir) if input_dir else Path("audios")
        self.log_dir = Path(log_dir) if log_dir else Path("dataset/log")
        self.historico_dir = Path(historico_dir) if historico_dir else Path("dataset/historico_dataset")
        
        # Modo dry-run (apenas simula)
        self.dry_run = dry_run
        
        # Criar diretorios se nao existirem
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.historico_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        self._setup_logging()
        
        # Estatisticas
        self.stats = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0,
            'errors': []
        }
    
    def _setup_logging(self):
        """Configura sistema de logging."""
        log_file = self.log_dir / "batch_processing.log"
        
        # Formato do log
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Configurar logging para arquivo e console
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("="*70)
        self.logger.info("BATCH PROCESSOR INICIADO")
        self.logger.info(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Input: {self.input_dir.absolute()}")
        self.logger.info(f"Log: {log_file.absolute()}")
        self.logger.info(f"Historico: {self.historico_dir.absolute()}")
        if self.dry_run:
            self.logger.info("MODO DRY-RUN: Nenhum processamento sera executado")
        self.logger.info("="*70)
    
    def discover_audio_folders(self) -> List[Path]:
        """
        Descobre todas as pastas com audios no diretorio de entrada.
        
        Returns:
            Lista de caminhos das pastas encontradas
        """
        if not self.input_dir.exists():
            self.logger.error(f"Diretorio de entrada nao existe: {self.input_dir}")
            return []
        
        # Listar todas as subpastas
        folders = [f for f in self.input_dir.iterdir() if f.is_dir()]
        
        # Filtrar pastas que contem arquivos de audio
        audio_folders = []
        for folder in folders:
            # Verificar se tem arquivo de audio dentro
            audio_files = list(folder.glob("*.flac")) + list(folder.glob("*.wav")) + \
                         list(folder.glob("*.mp3")) + list(folder.glob("*.mp4"))
            
            if audio_files:
                audio_folders.append(folder)
        
        # Ordenar por nome
        audio_folders.sort()
        
        self.logger.info(f"Descobertas {len(audio_folders)} pastas com audio")
        for folder in audio_folders:
            self.logger.info(f"  - {folder.name}")
        
        return audio_folders
    
    def is_already_processed(self, folder_id: str) -> bool:
        """
        Verifica se uma pasta ja foi processada checando historico.
        
        Args:
            folder_id: ID da pasta (nome da pasta)
            
        Returns:
            True se ja foi processado, False caso contrario
        """
        # Verificar se existe JSON no historico
        json_file = self.historico_dir / f"{folder_id}.json"
        
        if json_file.exists():
            self.logger.info(f"  [SKIP] {folder_id} - Ja processado (encontrado {json_file.name})")
            return True
        
        return False
    
    def process_single_audio(self, folder_path: Path) -> Tuple[bool, str]:
        """
        Processa um unico audio chamando run_pipeline.py.
        
        Args:
            folder_path: Caminho da pasta com audio
            
        Returns:
            Tupla (sucesso, mensagem)
        """
        folder_id = folder_path.name
        
        self.logger.info(f"Processando: {folder_id}")
        
        if self.dry_run:
            self.logger.info(f"  [DRY-RUN] Simulando processamento de {folder_id}")
            return True, "Dry-run - nao processado"
        
        try:
            # Construir comando
            cmd = [
                sys.executable,  # python ou python3
                "run_pipeline.py",
                str(folder_path),
                "--session-name",
                folder_id
            ]
            
            # Criar arquivo de log individual para este audio
            pipeline_log_file = self.log_dir / f"{folder_id}_pipeline.log"
            
            self.logger.info(f"  Executando: {' '.join(cmd)}")
            self.logger.info(f"  Log detalhado: {pipeline_log_file}")
            
            # Executar run_pipeline.py redirecionando output para arquivo
            with open(pipeline_log_file, 'w', encoding='utf-8') as log_file:
                result = subprocess.run(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,  # Redireciona stderr para stdout (mesmo arquivo)
                    text=True,
                    cwd=Path.cwd()  # Executar no diretorio atual
                )
            
            # Verificar resultado
            if result.returncode == 0:
                self.logger.info(f"  [OK] {folder_id} processado com sucesso")
                return True, "Sucesso"
            else:
                error_msg = f"Erro no processamento (exit code {result.returncode})"
                self.logger.error(f"  [ERRO] {folder_id}: {error_msg}")
                if result.stderr:
                    self.logger.error(f"  STDERR: {result.stderr[:500]}")
                return False, error_msg
        
        except Exception as e:
            error_msg = f"Excecao durante processamento: {str(e)}"
            self.logger.error(f"  [ERRO] {folder_id}: {error_msg}")
            return False, error_msg
    
    def process_all(self) -> Dict:
        """
        Processa todos os audios encontrados.
        
        Returns:
            Dicionario com estatisticas do processamento
        """
        # Descobrir pastas
        audio_folders = self.discover_audio_folders()
        
        if not audio_folders:
            self.logger.warning("Nenhuma pasta com audio encontrada")
            return self.stats
        
        self.stats['total'] = len(audio_folders)
        
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info(f"INICIANDO PROCESSAMENTO DE {self.stats['total']} AUDIOS")
        self.logger.info("="*70)
        
        # Processar cada pasta
        for i, folder_path in enumerate(audio_folders, 1):
            folder_id = folder_path.name
            
            self.logger.info("")
            self.logger.info(f"[{i}/{self.stats['total']}] {folder_id}")
            self.logger.info("-"*70)
            
            # Verificar se ja foi processado
            if self.is_already_processed(folder_id):
                self.stats['skipped'] += 1
                continue
            
            # Processar
            success, message = self.process_single_audio(folder_path)
            
            if success:
                self.stats['processed'] += 1
            else:
                self.stats['failed'] += 1
                self.stats['errors'].append({
                    'folder': folder_id,
                    'error': message
                })
        
        # Resumo final
        self._print_summary()
        
        return self.stats
    
    def _print_summary(self):
        """Imprime resumo do processamento."""
        self.logger.info("")
        self.logger.info("="*70)
        self.logger.info("RESUMO DO PROCESSAMENTO")
        self.logger.info("="*70)
        self.logger.info(f"Total de pastas encontradas: {self.stats['total']}")
        self.logger.info(f"Processados com sucesso: {self.stats['processed']}")
        self.logger.info(f"Pulados (ja processados): {self.stats['skipped']}")
        self.logger.info(f"Falharam: {self.stats['failed']}")
        
        if self.stats['errors']:
            self.logger.info("")
            self.logger.info("ERROS ENCONTRADOS:")
            for error in self.stats['errors']:
                self.logger.info(f"  - {error['folder']}: {error['error']}")
        
        self.logger.info("="*70)
        self.logger.info(f"Processamento concluido em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("="*70)


def main():
    """Ponto de entrada principal."""
    parser = argparse.ArgumentParser(
        description='Batch Processor - Processa automaticamente todos os audios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  python batch_processor.py                    # Usa defaults (audios/)
  python batch_processor.py --dry-run          # Simula sem processar
  python batch_processor.py --input-dir pasta  # Pasta customizada
        """
    )
    
    parser.add_argument(
        '--input-dir',
        default='audios',
        help='Diretorio com pastas de audio (default: audios/)'
    )
    
    parser.add_argument(
        '--log-dir',
        default='dataset/log',
        help='Diretorio para logs (default: dataset/log/)'
    )
    
    parser.add_argument(
        '--historico-dir',
        default='dataset/historico_dataset',
        help='Diretorio de historico (default: dataset/historico_dataset/)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simula processamento sem executar'
    )
    
    args = parser.parse_args()
    
    # Criar e executar processador
    processor = BatchProcessor(
        input_dir=args.input_dir,
        log_dir=args.log_dir,
        historico_dir=args.historico_dir,
        dry_run=args.dry_run
    )
    
    # Processar todos
    stats = processor.process_all()
    
    # Exit code baseado em resultados
    if stats['failed'] > 0:
        sys.exit(1)  # Erro se algum falhou
    else:
        sys.exit(0)  # Sucesso


if __name__ == "__main__":
    main()
