#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisador Espectral de Áudio
Detecta sample rate real vs declarado (upsampling artificial)
Independente da pipeline - uso standalone
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal

# ========================================
# CONFIGURAÇÕES
# ========================================

# Caminhos (relativo ou absoluto)
INPUT_DIR = "/home/ubuntu/z_projeto_2026/dataset/audio_dataset/QN7gUP7nYhQ"
OUTPUT_DIR = "/home/ubuntu/z_projeto_2026/dataset/audio_dataset/QN7gUP7nYhQ"
# Parâmetros de análise
ENERGY_THRESHOLD_DB = -60  # Threshold para detectar corte espectral
FORMATOS_SUPORTADOS = ['.flac', '.mp3', '.wav']

# Visualização
SHOW_PLOTS = True  # Mostrar gráficos na tela
SAVE_PLOTS = True  # Salvar PNG
PAUSE_BETWEEN_FILES = False  # Pausar entre arquivos (aguardar fechar janela)
DPI = 150  # Qualidade das imagens

# Logging
LOG_LEVEL = logging.INFO

# ========================================
# CONFIGURAÇÃO DE LOGGING
# ========================================

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ========================================
# FUNÇÕES DE ANÁLISE
# ========================================

def carregar_audio(audio_path: Path) -> Tuple[np.ndarray, int, int]:
    """
    Carrega arquivo de áudio com múltiplos backends (suporte 24-bit FLAC)
    
    Args:
        audio_path: Caminho do arquivo de áudio
        
    Returns:
        Tuple (audio_data, sample_rate, num_channels)
    """
    # Tentativa 1: soundfile direto (melhor para 16-bit)
    try:
        import soundfile as sf
        audio, sr = sf.read(str(audio_path), always_2d=False, dtype='float32')
        
        # Detectar canais
        if len(audio.shape) == 1:
            num_channels = 1
            audio = audio.reshape(1, -1)  # Garantir formato (channels, samples)
        else:
            num_channels = audio.shape[0] if audio.shape[0] < audio.shape[1] else 1
            if num_channels == 1 and len(audio.shape) > 1:
                audio = audio.T
        
        logger.info(f"Áudio carregado (soundfile): {audio_path.name}")
        logger.info(f"  Sample rate: {sr} Hz")
        logger.info(f"  Canais: {num_channels}")
        logger.info(f"  Duração: {audio.shape[-1] / sr:.2f}s")
        
        return audio, sr, num_channels
        
    except Exception as e:
        logger.warning(f"soundfile falhou: {e}")
    
    # Tentativa 2: librosa (melhor para 24-bit e arquivos grandes)
    try:
        audio, sr = librosa.load(str(audio_path), sr=None, mono=False, dtype=np.float32)
        
        # Detectar canais
        if len(audio.shape) == 1:
            num_channels = 1
            audio = audio.reshape(1, -1)
        else:
            num_channels = audio.shape[0]
        
        logger.info(f"Áudio carregado (librosa): {audio_path.name}")
        logger.info(f"  Sample rate: {sr} Hz")
        logger.info(f"  Canais: {num_channels}")
        logger.info(f"  Duração: {audio.shape[-1] / sr:.2f}s")
        
        return audio, sr, num_channels
        
    except Exception as e:
        logger.warning(f"librosa falhou: {e}")
    
    # Tentativa 3: Conversão via subprocess (fallback final)
    try:
        import subprocess
        import tempfile
        
        logger.info("Tentando conversão via subprocess...")
        
        # Criar arquivo WAV temporário
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Converter usando sox (se disponível) ou ffmpeg
        try:
            # Tentar sox primeiro (mais rápido)
            cmd = ['sox', str(audio_path), tmp_path]
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info("Conversão via sox bem-sucedida")
        except:
            # Fallback para ffmpeg
            cmd = [
                'ffmpeg', '-i', str(audio_path),
                '-acodec', 'pcm_s16le',
                '-y', tmp_path
            ]
            subprocess.run(cmd, capture_output=True, check=True, timeout=300)
            logger.info("Conversão via ffmpeg bem-sucedida")
        
        # Carregar arquivo convertido
        audio, sr = librosa.load(tmp_path, sr=None, mono=False, dtype=np.float32)
        
        # Limpar arquivo temporário
        Path(tmp_path).unlink()
        
        # Detectar canais
        if len(audio.shape) == 1:
            num_channels = 1
            audio = audio.reshape(1, -1)
        else:
            num_channels = audio.shape[0]
        
        logger.info(f"Áudio carregado (subprocess): {audio_path.name}")
        logger.info(f"  Sample rate: {sr} Hz")
        logger.info(f"  Canais: {num_channels}")
        logger.info(f"  Duração: {audio.shape[-1] / sr:.2f}s")
        
        return audio, sr, num_channels
        
    except Exception as e:
        logger.error(f"Todas as tentativas falharam para {audio_path}: {e}")
        raise RuntimeError(f"Não foi possível carregar o áudio: {e}")


def detectar_sample_rate_efetivo(audio: np.ndarray, sr: int, 
                                 threshold_db: float = -60) -> Dict:
    """
    Detecta sample rate efetivo analisando espectro de frequência
    Otimizado para áudios 24-bit e arquivos grandes
    
    Args:
        audio: Array de áudio (mono ou stereo)
        sr: Sample rate declarado
        threshold_db: Threshold de energia para detectar corte
        
    Returns:
        Dicionário com análise espectral
    """
    # Converter para mono se stereo (média dos canais)
    if len(audio.shape) > 1:
        audio_mono = np.mean(audio, axis=0)
    else:
        audio_mono = audio
    
    # Para arquivos muito grandes (>60s), usar amostra representativa
    max_samples = sr * 60  # 60 segundos
    if len(audio_mono) > max_samples:
        logger.info(f"Áudio longo detectado ({len(audio_mono)/sr:.0f}s), usando amostra de 60s")
        # Usar parte do meio (mais representativo)
        start_idx = len(audio_mono) // 2 - max_samples // 2
        audio_mono = audio_mono[start_idx:start_idx + max_samples]
    
    # FFT do áudio
    fft_result = np.fft.rfft(audio_mono)
    freqs = np.fft.rfftfreq(len(audio_mono), 1/sr)
    magnitude_db = 20 * np.log10(np.abs(fft_result) + 1e-10)
    
    # Normalizar magnitude (0 dB = pico)
    magnitude_db = magnitude_db - np.max(magnitude_db)
    
    # Encontrar frequência de corte (onde energia cai abaixo do threshold)
    above_threshold = magnitude_db > threshold_db
    
    # Procurar último ponto acima do threshold
    indices_above = np.where(above_threshold)[0]
    
    if len(indices_above) > 0:
        cutoff_idx = indices_above[-1]
        cutoff_freq = freqs[cutoff_idx]
    else:
        cutoff_freq = 0
    
    # Estimar sample rate efetivo (Nyquist: freq_max * 2)
    effective_sr = cutoff_freq * 2
    
    # Calcular energia por bandas
    bandas = {
        '0-8kHz': (0, 8000),
        '8-16kHz': (8000, 16000),
        '16-24kHz': (16000, 24000)
    }
    
    energia_bandas = {}
    for nome_banda, (f_min, f_max) in bandas.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(mask):
            energia = np.mean(10 ** (magnitude_db[mask] / 10))
            energia_bandas[nome_banda] = float(energia)
        else:
            energia_bandas[nome_banda] = 0.0
    
    # Determinar status
    tolerancia = 0.9  # 90% do SR declarado
    is_upsampled = effective_sr < (sr * tolerancia)
    
    if is_upsampled:
        status = "upsampled"
        qualidade_real = f"{int(effective_sr/1000)}kHz"
    else:
        status = "real"
        qualidade_real = f"{int(sr/1000)}kHz"
    
    return {
        'sample_rate_declarado': sr,
        'sample_rate_efetivo': int(effective_sr),
        'frequencia_corte_hz': float(cutoff_freq),
        'status': status,
        'qualidade_real': qualidade_real,
        'energia_por_banda': energia_bandas,
        'freqs': freqs,
        'magnitude_db': magnitude_db
    }


def gerar_espectrograma(audio: np.ndarray, sr: int, output_path: Path):
    """
    Gera espectrograma visual (similar ao Audacity)
    
    Args:
        audio: Array de áudio
        sr: Sample rate
        output_path: Caminho para salvar PNG
    """
    plt.figure(figsize=(14, 8))
    
    # Se stereo, plotar ambos os canais
    if len(audio.shape) > 1:
        num_channels = audio.shape[0]
        
        for i in range(num_channels):
            plt.subplot(num_channels, 1, i + 1)
            
            # Calcular espectrograma
            D = librosa.stft(audio[i])
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Plot
            librosa.display.specshow(
                S_db, 
                sr=sr, 
                x_axis='time', 
                y_axis='hz',
                cmap='viridis'
            )
            
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Espectrograma - Canal {i+1}')
            plt.ylabel('Frequência (Hz)')
            
            if i == num_channels - 1:
                plt.xlabel('Tempo (s)')
    else:
        # Mono
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        librosa.display.specshow(
            S_db, 
            sr=sr, 
            x_axis='time', 
            y_axis='hz',
            cmap='viridis'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title('Espectrograma')
        plt.ylabel('Frequência (Hz)')
        plt.xlabel('Tempo (s)')
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"  Espectrograma salvo: {output_path.name}")
    
    if SHOW_PLOTS and not PAUSE_BETWEEN_FILES:
        plt.draw()
        plt.pause(0.1)
    elif SHOW_PLOTS and PAUSE_BETWEEN_FILES:
        plt.show()
    
    if not PAUSE_BETWEEN_FILES:
        plt.close()


def gerar_plot_espectro(analise: Dict, output_path: Path):
    """
    Gera plot do espectro de frequência
    
    Args:
        analise: Dicionário com análise espectral
        output_path: Caminho para salvar PNG
    """
    plt.figure(figsize=(12, 6))
    
    freqs = analise['freqs']
    magnitude_db = analise['magnitude_db']
    cutoff_freq = analise['frequencia_corte_hz']
    
    # Plot espectro
    plt.plot(freqs / 1000, magnitude_db, linewidth=0.5, alpha=0.7)
    
    # Linha de corte
    plt.axvline(cutoff_freq / 1000, color='red', linestyle='--', 
                linewidth=2, label=f'Corte detectado: {cutoff_freq/1000:.1f} kHz')
    
    # Linha de threshold
    plt.axhline(ENERGY_THRESHOLD_DB, color='orange', linestyle='--', 
                linewidth=1, alpha=0.5, label=f'Threshold: {ENERGY_THRESHOLD_DB} dB')
    
    plt.xlabel('Frequência (kHz)')
    plt.ylabel('Magnitude (dB)')
    plt.title('Espectro de Frequência')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, (analise['sample_rate_declarado'] / 2000))  # até Nyquist
    plt.ylim(-80, 5)
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"  Espectro salvo: {output_path.name}")
    
    if SHOW_PLOTS and not PAUSE_BETWEEN_FILES:
        plt.draw()
        plt.pause(0.1)
    elif SHOW_PLOTS and PAUSE_BETWEEN_FILES:
        plt.show()
    
    if not PAUSE_BETWEEN_FILES:
        plt.close()


def gerar_plot_bandas(analise: Dict, output_path: Path):
    """
    Gera gráfico de barras com energia por banda de frequência
    
    Args:
        analise: Dicionário com análise espectral
        output_path: Caminho para salvar PNG
    """
    plt.figure(figsize=(10, 6))
    
    bandas = list(analise['energia_por_banda'].keys())
    energias = list(analise['energia_por_banda'].values())
    
    # Normalizar energias para 0-1
    max_energia = max(energias) if max(energias) > 0 else 1
    energias_norm = [e / max_energia for e in energias]
    
    # Cores baseadas na energia
    colors = ['green' if e > 0.5 else 'orange' if e > 0.1 else 'red' 
              for e in energias_norm]
    
    plt.bar(bandas, energias_norm, color=colors, alpha=0.7)
    
    # Adicionar valores no topo das barras
    for i, (banda, energia) in enumerate(zip(bandas, energias_norm)):
        plt.text(i, energia + 0.02, f'{energia:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Banda de Frequência')
    plt.ylabel('Energia Normalizada')
    plt.title('Energia por Banda de Frequência')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adicionar legenda de status
    status_text = f"Status: {analise['status'].upper()}\n"
    status_text += f"SR Declarado: {analise['sample_rate_declarado']/1000:.0f} kHz\n"
    status_text += f"SR Efetivo: {analise['sample_rate_efetivo']/1000:.0f} kHz"
    
    plt.text(0.98, 0.97, status_text, 
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if SAVE_PLOTS:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        logger.info(f"  Análise de bandas salva: {output_path.name}")
    
    if SHOW_PLOTS and not PAUSE_BETWEEN_FILES:
        plt.draw()
        plt.pause(0.1)
    elif SHOW_PLOTS and PAUSE_BETWEEN_FILES:
        plt.show()
    
    if not PAUSE_BETWEEN_FILES:
        plt.close()


def analisar_audio(audio_path: Path, output_dir: Path) -> Dict:
    """
    Análise completa de um arquivo de áudio
    
    Args:
        audio_path: Caminho do arquivo de áudio
        output_dir: Diretório de saída para resultados
        
    Returns:
        Dicionário com resultados da análise
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Analisando: {audio_path.name}")
    logger.info(f"{'='*60}")
    
    # Criar pasta de output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar áudio
    audio, sr, num_channels = carregar_audio(audio_path)
    
    # Análise espectral
    logger.info("Detectando sample rate efetivo...")
    analise = detectar_sample_rate_efetivo(audio, sr, ENERGY_THRESHOLD_DB)
    
    # Adicionar informações extras
    analise['arquivo'] = audio_path.name
    analise['canais'] = num_channels
    analise['duracao_segundos'] = float(len(audio.T) / sr)
    
    # Gerar gráficos
    logger.info("Gerando visualizações...")
    
    gerar_espectrograma(
        audio, sr, 
        output_dir / 'espectrograma.png'
    )
    
    gerar_plot_espectro(
        analise,
        output_dir / 'espectro_frequencia.png'
    )
    
    gerar_plot_bandas(
        analise,
        output_dir / 'analise_bandas.png'
    )
    
    # Remover dados de plot do JSON (muito grandes)
    analise_json = {k: v for k, v in analise.items() 
                   if k not in ['freqs', 'magnitude_db']}
    
    # Salvar JSON
    json_path = output_dir / 'resultado.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analise_json, f, indent=2, ensure_ascii=False)
    
    logger.info(f"  Resultado JSON salvo: {json_path.name}")
    
    # Log de resultado
    logger.info(f"\nResultado:")
    logger.info(f"  Sample Rate Declarado: {analise['sample_rate_declarado']} Hz")
    logger.info(f"  Sample Rate Efetivo: {analise['sample_rate_efetivo']} Hz")
    logger.info(f"  Frequência de Corte: {analise['frequencia_corte_hz']:.0f} Hz")
    logger.info(f"  Status: {analise['status'].upper()}")
    logger.info(f"  Qualidade Real: {analise['qualidade_real']}")
    
    return analise_json


def processar_pasta(input_dir: Path, output_dir: Path) -> List[Dict]:
    """
    Processa todos os áudios de uma pasta
    
    Args:
        input_dir: Pasta com arquivos de áudio
        output_dir: Pasta de saída
        
    Returns:
        Lista com resultados de todos os áudios
    """
    # Validar pasta de entrada
    if not input_dir.exists():
        logger.error(f"Pasta de entrada não encontrada: {input_dir}")
        raise FileNotFoundError(f"Pasta não encontrada: {input_dir}")
    
    # Buscar arquivos de áudio (apenas na raiz)
    audio_files = []
    for formato in FORMATOS_SUPORTADOS:
        audio_files.extend(list(input_dir.glob(f"*{formato}")))
    
    if not audio_files:
        logger.warning(f"Nenhum arquivo de áudio encontrado em: {input_dir}")
        logger.warning(f"Formatos suportados: {FORMATOS_SUPORTADOS}")
        return []
    
    logger.info(f"\nEncontrados {len(audio_files)} arquivos de áudio")
    
    # Processar cada arquivo
    resultados = []
    
    for i, audio_path in enumerate(sorted(audio_files), 1):
        logger.info(f"\n[{i}/{len(audio_files)}] Processando: {audio_path.name}")
        
        # Criar pasta de output para este áudio
        audio_output_dir = output_dir / f"{audio_path.stem}_analise"
        
        try:
            resultado = analisar_audio(audio_path, audio_output_dir)
            resultados.append(resultado)
            
        except Exception as e:
            logger.error(f"Erro ao processar {audio_path.name}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    return resultados


def main():
    """Função principal"""
    logger.info("="*60)
    logger.info("ANALISADOR ESPECTRAL DE ÁUDIO")
    logger.info("Detecção de Upsampling Artificial")
    logger.info("="*60)
    
    # Converter caminhos para Path (aceita relativo ou absoluto)
    input_path = Path(INPUT_DIR).resolve()
    output_path = Path(OUTPUT_DIR).resolve()
    
    logger.info(f"\nConfiguração:")
    logger.info(f"  Pasta de entrada: {input_path}")
    logger.info(f"  Pasta de saída: {output_path}")
    logger.info(f"  Threshold de energia: {ENERGY_THRESHOLD_DB} dB")
    logger.info(f"  Formatos suportados: {FORMATOS_SUPORTADOS}")
    
    # Processar
    inicio = datetime.now()
    
    resultados = processar_pasta(input_path, output_path)
    
    fim = datetime.now()
    tempo_total = (fim - inicio).total_seconds()
    
    # Resumo geral
    if resultados:
        logger.info(f"\n{'='*60}")
        logger.info("RESUMO GERAL")
        logger.info(f"{'='*60}")
        logger.info(f"Total processado: {len(resultados)} áudios")
        logger.info(f"Tempo total: {tempo_total:.2f}s")
        
        # Estatísticas
        upsampled = sum(1 for r in resultados if r['status'] == 'upsampled')
        real = len(resultados) - upsampled
        
        logger.info(f"\nStatus:")
        logger.info(f"  Real: {real}")
        logger.info(f"  Upsampled: {upsampled}")
        
        # Salvar resumo geral
        resumo_path = output_path / 'resumo_geral.json'
        resumo = {
            'data_analise': inicio.isoformat(),
            'total_arquivos': len(resultados),
            'tempo_processamento_s': tempo_total,
            'configuracao': {
                'threshold_db': ENERGY_THRESHOLD_DB,
                'formatos': FORMATOS_SUPORTADOS
            },
            'estatisticas': {
                'real': real,
                'upsampled': upsampled
            },
            'resultados': resultados
        }
        
        with open(resumo_path, 'w', encoding='utf-8') as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nResumo geral salvo: {resumo_path}")
        logger.info(f"Resultados em: {output_path}")
    
    else:
        logger.warning("\nNenhum arquivo foi processado.")
    
    logger.info("\n" + "="*60)
    logger.info("Processamento concluído!")
    logger.info("="*60)
    
    # Manter janelas abertas se configurado
    if SHOW_PLOTS and PAUSE_BETWEEN_FILES:
        plt.show()


if __name__ == "__main__":
    main()