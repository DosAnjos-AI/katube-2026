#!/usr/bin/env python3
"""
Command-line interface to run the Audio Processing Pipeline on local directories.
"""
import os
import shutil
import sys
import logging
from pathlib import Path
import argparse
import json
from dotenv import load_dotenv # Para carregar variáveis de ambiente de um arquivo .env (opcional)

# Adiciona o diretório 'src' ao path para encontrar os módulos da pipeline
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Importa a classe principal da pipeline e a configuração
try:
    from pipeline import AudioProcessingPipeline
    from config import Config
except ModuleNotFoundError as e:
    print(f"Erro: Não foi possível importar módulos da pipeline. Verifique se está no diretório correto.")
    print(f"Detalhe: {e}")
    sys.exit(1)

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def main():
    # --- Configuração dos Argumentos de Linha de Comando ---
    parser = argparse.ArgumentParser(description="Executa a pipeline de processamento de áudio em um diretório local.")

    # Argumento OBRIGATÓRIO: Caminho para o diretório de entrada
    parser.add_argument(
        "input_directory",
        type=Path, # Converte automaticamente para um objeto Path
        help="Caminho completo para o diretório que contém o arquivo .flac a ser processado."
    )

    # Argumentos OPCIONAIS (espelhando as opções do HTML/pipeline)
    parser.add_argument(
        "--session-name",
        type=str,
        default=None,
        help="Nome personalizado para a sessão de processamento (padrão: nome do diretório de entrada)."
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Número esperado de locutores (para diarização, opcional)."
    )
    # Exemplo de flag booleana (liga/desliga)
    parser.add_argument(
        '--enhance-audio',
        action=argparse.BooleanOptionalAction, # Cria --enhance-audio e --no-enhance-audio
        default=True,
        help="Habilita/Desabilita a melhoria de áudio na separação (padrão: habilitado)."
    )
    parser.add_argument(
        '--intelligent-segmentation',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Usa segmentação inteligente baseada em VAD (padrão: habilitado)."
    )
    # Adicione outros argumentos opcionais conforme necessário (min_duration, max_duration, mos_threshold, etc.)
    parser.add_argument("--min-duration", type=float, default=4.0, help="Duração mínima do segmento (s).")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Duração máxima do segmento (s).")
    parser.add_argument("--mos-threshold", type=float, default=2.5, help="Limiar mínimo do filtro MOS.")
    parser.add_argument("--hf-token", type=str, default=os.getenv('HUGGINGFACE_TOKEN'), help="Token de acesso do Hugging Face (pode ser via variável de ambiente HUGGINGFACE_TOKEN).")
    parser.add_argument("--output-base-dir", type=Path, default=Config.OUTPUT_DIR, help="Diretório base onde as sessões de saída serão criadas.")
    parser.add_argument('--cleanup-policy', type=str, default='final_dataset', choices=['final_dataset', 'intermediate', 'all_except_raw_data', 'none'], help="Política de limpeza a ser aplicada no final.")
    parser.add_argument('--use-cuda', action=argparse.BooleanOptionalAction, default=True, help="Habilita o uso de GPU (CUDA) se disponível.")

    # --- Parsing dos Argumentos ---
    args = parser.parse_args()

    # --- Carregar Variáveis de Ambiente (Opcional, mas recomendado) ---
    load_dotenv() # Procura por um arquivo .env no diretório atual ou superior

    # Verifica se o token do Hugging Face foi fornecido (via argumento ou variável de ambiente)
    huggingface_token = args.hf_token or os.getenv('HUGGINGFACE_TOKEN')
    if not huggingface_token:
         logger.warning("⚠️ Token do Hugging Face não fornecido (--hf-token ou HUGGINGFACE_TOKEN). Modelos protegidos (como Pyannote) podem falhar.")
         # Você pode decidir parar aqui se o token for essencial:
         # parser.error("Token do Hugging Face é obrigatório.")

    logger.info("🚀 Iniciando a Pipeline de Processamento de Áudio Local...")
    logger.info(f"Diretório de Entrada: {args.input_directory}")
    logger.info(f"Diretório Base de Saída: {args.output_base_dir}")
    logger.info(f"Política de Cleanup: {args.cleanup_policy}")
    logger.info(f"Usar CUDA: {args.use_cuda}")


    # --- Execução da Pipeline ---
    try:
        # Instancia a pipeline, passando as configurações relevantes do argparse
        pipeline = AudioProcessingPipeline(
            output_base_dir=args.output_base_dir,
            huggingface_token=huggingface_token,
            segment_min_duration=args.min_duration,
            segment_max_duration=args.max_duration,
            mos_threshold=args.mos_threshold,
            use_cuda=args.use_cuda
            # Passe outras configurações do __init__ que você tenha adicionado
        )

        # Chama o método para processar o diretório local
        results = pipeline.process_local_audio(
            audio_path=args.input_directory,
            session_name=args.session_name,
            num_speakers=args.num_speakers,
            enhance_audio=args.enhance_audio,
            use_intelligent_segmentation=args.intelligent_segmentation
        )

        logger.info("✅ Pipeline concluída com sucesso!")
        logger.info("Resultados:")
        # Imprime os resultados de forma legível
        print(json.dumps(results, indent=2, ensure_ascii=False, default=str))

        #Apagando diretório original
        logger.info("🧹 Executando cleanup final...")
        shutil.rmtree(args.input_directory)

    except FileNotFoundError as e:
        logger.error(f"❌ ERRO DE ARQUIVO: {e}")
        sys.exit(1)
    except NotADirectoryError as e:
        logger.error(f"❌ ERRO DE DIRETÓRIO: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ ERRO INESPERADO NA PIPELINE: {e}", exc_info=True) # exc_info=True mostra o traceback
        sys.exit(1)

# --- Ponto de Entrada Padrão do Python ---
if __name__ == "__main__":
    main()