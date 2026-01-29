"""
Módulo 06: MOS Filter (Mean Opinion Score - Qualidade de Áudio)
Avalia qualidade de segmentos de áudio usando modelo SQUIM (Speech Quality and Intelligibility Measures)
Classifica segmentos em: baixa, média ou alta qualidade baseado em limiares configuráveis
"""

import torch
import torchaudio
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import time
import sys
import shutil

sys.path.append(str(Path(__file__).parent.parent))

from config import MOS_FILTER, PROJECT_ROOT


# ==================== CONFIGURAÇÃO MANUAL ====================
# ID do vídeo a ser processado
video_id = '0aICqierMVA'


# ==================== CONFIGURAÇÃO DE LOGGING ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== FUNÇÕES AUXILIARES ====================

def detectar_device() -> torch.device:
    """
    Detecta dispositivo disponível baseado na configuração.
    
    Returns:
        torch.device: Dispositivo a ser usado (CPU ou CUDA)
    """
    device_config = MOS_FILTER['device']
    
    if device_config == 'cpu':
        device = torch.device('cpu')
        logger.info("Dispositivo configurado: CPU (forçado)")
        
    elif device_config == 'gpu':
        if not torch.cuda.is_available():
            raise RuntimeError("GPU configurada mas não disponível")
        device = torch.device('cuda')
        logger.info("Dispositivo configurado: GPU/CUDA (forçado)")
        
    elif device_config == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info("Dispositivo detectado: GPU/CUDA (automático)")
        else:
            device = torch.device('cpu')
            logger.info("Dispositivo detectado: CPU (automático - GPU não disponível)")
    else:
        raise ValueError(f"Configuração de device inválida: '{device_config}'. Use 'auto', 'cpu' ou 'gpu'")
    
    return device


def calcular_batch_size(device: torch.device) -> int:
    """
    Calcula batch size baseado em configuração e dispositivo.
    
    Args:
        device: Dispositivo sendo usado
        
    Returns:
        int: Batch size a ser usado
    """
    batch_size_config = MOS_FILTER['batch']['batch_size']
    
    if batch_size_config == 'auto':
        if device.type == 'cuda':
            batch_size = 8
            logger.info(f"Batch size automático (GPU): {batch_size}")
        else:
            batch_size = 1
            logger.info(f"Batch size automático (CPU): {batch_size}")
    else:
        batch_size = int(batch_size_config)
        logger.info(f"Batch size configurado: {batch_size}")
    
    return batch_size


def carregar_modelo(device: torch.device) -> torch.nn.Module:
    """
    Carrega modelo SQUIM no dispositivo especificado.
    
    Args:
        device: Dispositivo onde carregar o modelo
        
    Returns:
        torch.nn.Module: Modelo SQUIM carregado
    """
    logger.info("Carregando modelo SQUIM...")
    start_time = time.time()
    
    try:
        model = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model().to(device)
        model.eval()
        
        elapsed = time.time() - start_time
        logger.info(f"Modelo SQUIM carregado com sucesso ({elapsed:.2f}s)")
        
        return model
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo SQUIM: {e}")
        raise


def carregar_json_input(json_path: Path) -> Dict:
    """
    Carrega e valida JSON de entrada.
    
    Args:
        json_path: Caminho para o JSON de entrada
        
    Returns:
        Dict: Dados do JSON carregado
        
    Raises:
        FileNotFoundError: Se JSON não existe
        json.JSONDecodeError: Se JSON está corrompido
        ValueError: Se campos obrigatórios estão ausentes
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON de entrada não encontrado: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            dados = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON corrompido: {json_path}", e.doc, e.pos)
    
    # Valida campos obrigatórios em cada segmento
    campos_obrigatorios = ['tempo_inicio', 'tempo_fim', 'duracao']
    
    for nome_arquivo, metadados in dados.items():
        for campo in campos_obrigatorios:
            if campo not in metadados:
                raise ValueError(f"Campo obrigatório '{campo}' ausente em '{nome_arquivo}'")
    
    logger.info(f"JSON de entrada carregado: {len(dados)} segmentos")
    return dados


def preparar_audio(audio_path: Path, device: torch.device) -> torch.Tensor:
    """
    Carrega e prepara áudio para processamento SQUIM.
    
    Args:
        audio_path: Caminho para o arquivo de áudio
        device: Dispositivo onde colocar tensor
        
    Returns:
        torch.Tensor: Áudio preparado (1, samples) em 16kHz
    """
    # Carrega áudio (já deve estar em 16kHz)
    audio, sr = torchaudio.load(audio_path)
    
    # Converte para mono se necessário
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Verifica sample rate (deve ser 16kHz)
    if sr != 16000:
        logger.warning(f"Sample rate inesperado: {sr}Hz (esperado 16kHz) - {audio_path.name}")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    
    # Ajusta para exatamente 12s (192000 samples @ 16kHz)
    target_samples = 192000
    
    if audio.shape[1] < target_samples:
        # Padding se muito curto
        padding = target_samples - audio.shape[1]
        audio = torch.nn.functional.pad(audio, (0, padding), value=0.0)
    elif audio.shape[1] > target_samples:
        # Trunca se muito longo
        audio = audio[:, :target_samples]
    
    return audio.to(device)


def calcular_mos_batch(
    model: torch.nn.Module,
    audios: List[torch.Tensor],
    device: torch.device
) -> List[Dict[str, float]]:
    """
    Calcula MOS para um batch de áudios.
    
    Args:
        model: Modelo SQUIM
        audios: Lista de tensores de áudio
        device: Dispositivo de processamento
        
    Returns:
        List[Dict]: Lista de dicionários com métricas para cada áudio
    """
    # Empilha áudios em batch (batch, 192000)
    batch = torch.cat(audios, dim=0).to(device)
    
    # Processa batch
    with torch.no_grad():
        stoi, pesq, si_sdr = model(batch)
    
    # Converte resultados para lista de dicionários
    resultados = []
    for i in range(len(audios)):
        resultados.append({
            'mos_score': float(pesq[i].cpu().item()),      # PESQ (1-5) como score principal
            'mos_stoi': float(stoi[i].cpu().item()),       # STOI (0-1)
            'mos_si_sdr': float(si_sdr[i].cpu().item())    # SI-SDR (dB)
        })
    
    return resultados


def classificar_qualidade(mos_score: float) -> str:
    """
    Classifica qualidade do áudio baseado no MOS score (PESQ).
    
    Args:
        mos_score: Score PESQ (1-5)
        
    Returns:
        str: 'baixa', 'media' ou 'alta'
    """
    min_threshold = MOS_FILTER['thresholds']['min_threshold']
    max_threshold = MOS_FILTER['thresholds']['max_threshold']
    
    if mos_score < min_threshold:
        return 'baixa'
    elif mos_score >= max_threshold:
        return 'alta'
    else:
        return 'media'


def processar_segmentos(
    input_dir: Path,
    dados_json: Dict,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int
) -> Dict:
    """
    Processa todos os segmentos calculando MOS scores.
    
    Args:
        input_dir: Diretório com os áudios de entrada
        dados_json: Dados do JSON original
        model: Modelo SQUIM
        device: Dispositivo de processamento
        batch_size: Tamanho do batch
        
    Returns:
        Dict: Dados JSON atualizados com MOS scores
    """
    total_segmentos = len(dados_json)
    logger.info(f"Processando {total_segmentos} segmentos (batch_size={batch_size})")
    
    # Prepara estrutura de resultados
    resultados = {}
    
    # Lista de arquivos a processar
    arquivos = list(dados_json.keys())
    
    # Processa em batches
    for i in range(0, len(arquivos), batch_size):
        batch_files = arquivos[i:i+batch_size]
        batch_audios = []
        batch_names = []
        
        # Carrega áudios do batch
        for nome_arquivo in batch_files:
            audio_path = input_dir / nome_arquivo
            
            if not audio_path.exists():
                logger.error(f"Arquivo de áudio não encontrado: {audio_path}")
                raise FileNotFoundError(f"Arquivo de áudio não encontrado: {audio_path}")
            
            audio = preparar_audio(audio_path, device)
            batch_audios.append(audio)
            batch_names.append(nome_arquivo)
        
        # Calcula MOS para o batch
        batch_resultados = calcular_mos_batch(model, batch_audios, device)
        
        # Atualiza dados com resultados
        for nome_arquivo, mos_metrics in zip(batch_names, batch_resultados):
            # Copia dados originais
            resultados[nome_arquivo] = dados_json[nome_arquivo].copy()
            
            # Adiciona métricas MOS
            resultados[nome_arquivo].update(mos_metrics)
            
            # Adiciona classificação de qualidade
            mos_score = mos_metrics['mos_score']
            resultados[nome_arquivo]['mos_qualidade'] = classificar_qualidade(mos_score)
        
        # Log de progresso
        processados = min(i + batch_size, total_segmentos)
        logger.info(f"Processados {processados}/{total_segmentos} segmentos")
    
    return resultados


def filtrar_segmentos_aprovados(dados_completos: Dict) -> Dict:
    """
    Filtra apenas segmentos com qualidade média ou alta.
    
    Args:
        dados_completos: Dados com todos os segmentos
        
    Returns:
        Dict: Apenas segmentos aprovados (média/alta)
    """
    aprovados = {}
    
    for nome_arquivo, metadados in dados_completos.items():
        qualidade = metadados.get('mos_qualidade')
        if qualidade in ['media', 'alta']:
            aprovados[nome_arquivo] = metadados
    
    total = len(dados_completos)
    aprovados_count = len(aprovados)
    rejeitados = total - aprovados_count
    
    logger.info(f"Segmentos aprovados: {aprovados_count}/{total} ({rejeitados} rejeitados)")
    
    return aprovados


def salvar_json_com_validacao(dados: Dict, output_path: Path) -> bool:
    """
    Salva JSON e valida pós-escrita.
    
    Args:
        dados: Dados a serem salvos
        output_path: Caminho de saída
        
    Returns:
        bool: True se salvou e validou com sucesso
    """
    try:
        # Salva JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dados, f, indent=2, ensure_ascii=False)
        
        # Validação pós-escrita: tenta carregar
        with open(output_path, 'r', encoding='utf-8') as f:
            json.load(f)
        
        logger.info(f"JSON salvo e validado: {output_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao salvar/validar JSON {output_path.name}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def gerar_estatisticas(dados_completos: Dict) -> None:
    """
    Gera e loga estatísticas do processamento.
    
    Args:
        dados_completos: Dados com todos os segmentos processados
    """
    total = len(dados_completos)
    alta = sum(1 for d in dados_completos.values() if d['mos_qualidade'] == 'alta')
    media = sum(1 for d in dados_completos.values() if d['mos_qualidade'] == 'media')
    baixa = sum(1 for d in dados_completos.values() if d['mos_qualidade'] == 'baixa')
    
    logger.info("=" * 60)
    logger.info("ESTATISTICAS DE QUALIDADE MOS")
    logger.info("=" * 60)
    logger.info(f"Total de segmentos: {total}")
    logger.info(f"Alta qualidade: {alta} ({alta/total*100:.1f}%)")
    logger.info(f"Media qualidade: {media} ({media/total*100:.1f}%)")
    logger.info(f"Baixa qualidade (rejeitados): {baixa} ({baixa/total*100:.1f}%)")
    logger.info(f"Aprovados (media+alta): {alta+media} ({(alta+media)/total*100:.1f}%)")
    logger.info("=" * 60)


# ==================== FUNÇÃO PRINCIPAL ====================

def processar_mos(video_id: str) -> bool:
    """
    Processa MOS para todos os segmentos de um vídeo.
    
    Args:
        video_id: ID do vídeo (ex: "CA6TSoMw86k")
        
    Returns:
        bool: True se processado com sucesso, False caso contrário
    """
    try:
        logger.info("=" * 60)
        logger.info(f"INICIANDO PROCESSAMENTO MOS - Video ID: {video_id}")
        logger.info("=" * 60)
        
        # Define paths
        input_dir = PROJECT_ROOT / 'arquivos' / 'temp' / video_id / '03-segments_16khz'
        output_dir = PROJECT_ROOT / 'arquivos' / 'temp' / video_id / '04-mos_score'
        
        json_input_path = input_dir / f"{video_id}_segments_originais.json"
        json_acompanhamento_path = output_dir / f"{video_id}_segments_acompanhamento.json"
        json_mos_path = output_dir / f"{video_id}_segments_mos.json"
        
        # Verifica se deve processar (sobrescrever)
        sobrescrever = MOS_FILTER['comportamento']['sobrescrever']
        
        if json_acompanhamento_path.exists() and not sobrescrever:
            logger.info(f"MOS já processado para {video_id} e sobrescrever=False. Pulando...")
            return True
        
        # Cria diretório de output se não existe
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Carrega JSON de entrada
        logger.info(f"Carregando JSON de entrada: {json_input_path.name}")
        dados_json = carregar_json_input(json_input_path)
        
        # Detecta device e configura batch size
        device = detectar_device()
        batch_size = calcular_batch_size(device)
        
        # Carrega modelo
        model = carregar_modelo(device)
        
        # Processa todos os segmentos
        start_time = time.time()
        dados_completos = processar_segmentos(
            input_dir=input_dir,
            dados_json=dados_json,
            model=model,
            device=device,
            batch_size=batch_size
        )
        elapsed = time.time() - start_time
        logger.info(f"Processamento concluído em {elapsed:.2f}s")
        
        # Gera estatísticas
        gerar_estatisticas(dados_completos)
        
        # Salva JSON de acompanhamento (todos os segmentos)
        logger.info("Salvando JSON de acompanhamento...")
        if not salvar_json_com_validacao(dados_completos, json_acompanhamento_path):
            logger.error("Falha ao salvar JSON de acompanhamento")
            return False
        
        # Filtra segmentos aprovados (média/alta)
        dados_aprovados = filtrar_segmentos_aprovados(dados_completos)
        
        # Salva JSON MOS (apenas aprovados)
        logger.info("Salvando JSON MOS (segmentos aprovados)...")
        if not salvar_json_com_validacao(dados_aprovados, json_mos_path):
            logger.error("Falha ao salvar JSON MOS")
            return False
        
        # Copia JSONs para pasta 00-json_dinamico
        logger.info("Copiando JSONs para 00-json_dinamico...")
        json_dinamico_dir = PROJECT_ROOT / 'arquivos' / 'temp' / video_id / '00-json_dinamico'
        
        # Copia acompanhamento (mesmo nome)
        shutil.copy2(
            json_acompanhamento_path,
            json_dinamico_dir / f"{video_id}_segments_acompanhamento.json"
        )
        
        # Copia mos (renomeia para {id}.json)
        shutil.copy2(
            json_mos_path,
            json_dinamico_dir / f"{video_id}.json"
        )
        
        logger.info("JSONs copiados para 00-json_dinamico")
        
        logger.info("=" * 60)
        logger.info(f"MOS processado com sucesso: {video_id}")
        logger.info("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        logger.error(f"Arquivo não encontrado: {e}")
        return False
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON corrompido ou inválido: {e}")
        return False
        
    except ValueError as e:
        logger.error(f"Erro de validação: {e}")
        return False
        
    except Exception as e:
        logger.error(f"Erro inesperado ao processar MOS: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==================== TESTE MANUAL ====================

if __name__ == "__main__":
    sucesso = processar_mos(video_id)
    
    if sucesso:
        logger.info("Processamento MOS concluido com sucesso")
    else:
        logger.error("Erro ao processar MOS")