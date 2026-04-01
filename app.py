#!/usr/bin/env python3
"""
Flask web interface for YouTube Audio Processing Pipeline
"""
# Suppress warnings first
from src.warning_suppression import *

import os
import sys
import json
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.pipeline import AudioProcessingPipeline
from src.config import Config

app = Flask(__name__)
import secrets
app.secret_key = secrets.token_urlsafe(32)

# Global variables for job tracking
active_jobs = {}
job_lock = threading.Lock()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobStatus:
    def __init__(self, job_id: str, url: str):
        self.job_id = job_id
        self.url = url
        self.status = "waiting"
        self.progress = 0
        self.message = "Iniciando processamento..."
        self.start_time = datetime.now()
        self.end_time = None
        self.results = None
        self.error = None

    def update(self, status: str, progress: int, message: str):
        self.status = status
        self.progress = progress
        self.message = message
        logger.info(f"Job {self.job_id}: {status} - {progress}% - {message}")

    def complete(self, results: dict):
        self.status = "completed"
        self.progress = 100
        self.message = "Processamento concluído com sucesso!"
        self.end_time = datetime.now()
        self.results = results

    def fail(self, error: str):
        self.status = "failed"
        self.progress = 0
        self.message = f"Erro: {error}"
        self.end_time = datetime.now()
        self.error = error


def process_youtube_url_background(job_id: str, audio_path: Path, options: dict):
    """Background task to process audio file"""
    job = active_jobs[job_id]
    try:
        pipeline = AudioProcessingPipeline(
            output_base_dir=Config.OUTPUT_DIR,
            huggingface_token=os.getenv('HUGGINGFACE_TOKEN'),
            segment_min_duration=options.get('min_duration', 4.0),
            segment_max_duration=options.get('max_duration', 18.0)
        )
        job.update("downloading", 10, "Baixando áudio do YouTube...")

        session_name = options.get('session_name') or audio_path.stem
        session_dir = pipeline.create_session(session_name)
        segments = pipeline.segment_audio(audio_path, options.get('intelligent_segmentation', True))
        job.update("filtering", 30, "Aplicando filtros de qualidade...")

        if pipeline.enable_mos_filter:
            mos_rejected_dir = session_dir / 'audio_descartado_mos'
            mos_result = pipeline.apply_mos_filter(segments, rejected_dir=mos_rejected_dir)
            segments = mos_result['filtered_segments']
            job.update("filtering", 40, f"Filtro MOS: {len(segments)} segmentos aprovados")

            segments_aprovados_dir = session_dir / 'segments_aprovados'
            segments_aprovados_dir.mkdir(exist_ok=True)
            approved_segments = []

            for segment in segments:
                approved_path = segments_aprovados_dir / segment.name
                import shutil
                shutil.copy2(segment, approved_path)
                approved_segments.append(approved_path)

            segments = approved_segments

        job.update("diarizing", 50, "Executando diarização...")
        diarization_results = pipeline.perform_diarization(segments, options.get('num_speakers'))

        job.update("overlap", 60, "Detectando sobreposições...")
        clean_segments, overlapping_segments = pipeline.detect_overlaps(segments)

        job.update("separating", 70, "Separando por locutor...")
        separation_results = pipeline.separate_speakers(diarization_results, options.get('enhance_audio', True))

        job.update("preparing", 75, "Preparando arquivos para STT...")
        stt_files = pipeline.prepare_for_stt(separation_results)

        if pipeline.enable_stt:
            if stt_files:
                all_stt_files = []
                for speaker_files in stt_files.values():
                    all_stt_files.extend(speaker_files)
                stt_result = pipeline.transcribe_audio_segments(all_stt_files)
            else:
                stt_result = pipeline.transcribe_audio_segments(segments)

            validation_info = ""
            filter_info = ""

            if 'validation' in stt_result and stt_result['validation'].get('success'):
                avg_sim = stt_result['validation'].get('average_similarity', 0)
                validation_info = f" (Validação: {avg_sim:.3f} similaridade)"

            if 'filter_and_denoise' in stt_result and stt_result['filter_and_denoise'].get('success'):
                validated_count = stt_result['filter_and_denoise'].get('validated_count', 0)
                denoised_count = stt_result['filter_and_denoise'].get('denoised_count', 0)
                filter_info = f" | Filtro 80%: {validated_count} validados, {denoised_count} denoised"

            job.update("filtering", 85, f"STT: {stt_result.get('whisper_count', 0)} Whisper + {stt_result.get('wav2vec2_count', 0)} WAV2VEC2{validation_info}{filter_info}")

            if 'filter_and_denoise' in stt_result and stt_result['filter_and_denoise'].get('success'):
                job.update("finalizing", 90, "Criando dataset final com normalização Sox...")
                denoised_audio_paths = stt_result['filter_and_denoise'].get('denoised_audio_paths', [])

                if denoised_audio_paths:
                    final_dataset_result = pipeline.create_final_dataset(
                        denoised_audio_paths=denoised_audio_paths,
                        stt_results_dir=session_dir / 'stt_results',
                        output_dir=session_dir
                    )
                    job.update("finalizing", 95, f"Dataset final: {final_dataset_result['success_count']} áudios normalizados")
                else:
                    final_dataset_result = {'success_count': 0, 'failure_count': 0}
            else:
                final_dataset_result = {'success_count': 0, 'failure_count': 0}

        job.update("clean_up", 99, "Executando limpeza de arquivos intermediários...")
        pipeline.cleanup(stages_to_clean=[
            "downloads", "segments", "stt_ready",
            "audios_abaixo_2,5_MOS", "audios_acima_3,0_MOS",
            "audios_validados_tts", "audios_denoiser", "clean",
            "audios_entre_2,5_e_3,0_MOS", "diarization", "overlapping", "speakers"
        ])

        processing_time = time.time() - job.start_time.timestamp()
        results = {
            'session_name': session_name,
            'session_dir': str(session_dir),
            'url': audio_path,
            'processing_time': processing_time
        }

        results_file = session_dir / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        job.complete(results)
    except Exception as e:
        logger.error(f"Background job {job_id} failed: {e}")
        job.fail(str(e))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Start processing a audio file"""
    try:
        job_id = str(uuid.uuid4())
        data = request.get_json()
        url = data.get('url')

        options = {
            'filename': data.get('filename'),
            'num_speakers': data.get('num_speakers'),
            'min_duration': data.get('min_duration', 4.0),
            'max_duration': data.get('max_duration', 18.0),
            'enhance_audio': data.get('enhance_audio', True),
            'intelligent_segmentation': data.get('intelligent_segmentation', True),
            'session_name': data.get('session_name')
        }

        with job_lock:
            active_jobs[job_id] = JobStatus(job_id, url)

        thread = threading.Thread(
            target=process_youtube_url_background,
            args=(job_id, Path(url), options),
            daemon=True
        )
        thread.start()

        return jsonify({'job_id': job_id})
    except Exception as e:
        logger.error(f"Process URL error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get job status"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'progress': job.progress,
            'message': job.message,
            'start_time': job.start_time.isoformat(),
            'end_time': job.end_time.isoformat() if job.end_time else None,
            'error': job.error
        })

@app.route('/result/<job_id>')
def get_result(job_id):
    """Get job results"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job:
            return jsonify({'error': 'Job não encontrado'}), 404
        
        if job.status != 'completed':
            return jsonify({'error': 'Job ainda não foi concluído'}), 400
        
        # Clean results for JSON serialization
        def clean_for_json(obj):
            if isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_for_json(item) for item in obj]
            elif hasattr(obj, '__module__') and 'pyannote' in str(obj.__module__):
                return f"pyannote.{obj.__class__.__name__}"  # Handle pyannote objects
            elif hasattr(obj, '__dict__'):
                return str(obj)  # Convert complex objects to string
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif hasattr(obj, '__class__'):
                return f"{obj.__class__.__module__}.{obj.__class__.__name__}"  # Better class representation
            else:
                return str(obj)  # Convert anything else to string
        
        return jsonify({
            'job_id': job.job_id,
            'status': job.status,
            'results': clean_for_json(job.results),
            'processing_time': (job.end_time - job.start_time).total_seconds() if job.end_time and job.start_time else 0
        })

@app.route('/results/<job_id>')
def results_page(job_id):
    """Results page"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job:
            return "Job não encontrado", 404
            
    return render_template('result.html', job_id=job_id)

@app.route('/download/<job_id>/<path:file_type>')
def download_file(job_id, file_type):
    """Download processed files"""
    with job_lock:
        job = active_jobs.get(job_id)
        
        if not job or job.status != 'completed':
            return "Arquivo não disponível", 404
        
        session_dir = Path(job.results['session_dir'])
        
        try:
            if file_type == 'results.json':
                file_path = session_dir / 'pipeline_results.json'
                return send_file(file_path, as_attachment=True, download_name=f'results_{job_id}.json')
            
            elif file_type.startswith('speaker_'):
                # Download specific speaker files as ZIP
                import zipfile
                import tempfile
                
                speaker_id = file_type.replace('speaker_', '')
                speaker_dir = session_dir / 'stt_ready' / f'speaker_{speaker_id}'
                
                if not speaker_dir.exists():
                    return "Speaker não encontrado", 404
                
                # Create temporary ZIP file
                temp_zip = tempfile.mktemp(suffix='.zip')
                
                with zipfile.ZipFile(temp_zip, 'w') as zipf:
                    for file_path in speaker_dir.glob('*'):
                        if file_path.is_file():
                            zipf.write(file_path, file_path.name)
                
                return send_file(temp_zip, as_attachment=True, download_name=f'speaker_{speaker_id}_{job_id}.zip')
            
            else:
                return "Tipo de arquivo inválido", 400
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return "Erro no download", 500

@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup_job(job_id):
    """Clean up job data"""
    with job_lock:
        if job_id in active_jobs:
            del active_jobs[job_id]
    
    return jsonify({'message': 'Job removido'})

@app.route('/jobs')
def list_jobs():
    """List all active jobs (for debugging)"""
    with job_lock:
        jobs_info = []
        for job_id, job in active_jobs.items():
            jobs_info.append({
                'job_id': job_id,
                'status': job.status,
                'progress': job.progress,
                'url': job.url,
                'start_time': job.start_time.isoformat()
            })
    
    return jsonify(jobs_info)

if __name__ == '__main__':
    
    # Ensure directories exist
    Config.create_directories()
    
    # Create additional directories
    Config.AUDIOS_BAIXADOS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("🚀 YouTube Audio Processing Pipeline - Web Interface")
    print("============================================================")
    print("📁 Processamento local - Arquivos salvos no disco")
    print("🎯 Aceita: Canais YouTube OU vídeos individuais")
    print("🔍 Pipeline: Download → Normalização → Segmentação → MOS → Diarização → OSD → Separação → STT → Validação → Denoiser → Normalização Final → Dataset")
    print("🌐 Acesse: http://localhost:5000")
    print("============================================================")
    print()
    print("Recursos disponíveis:")
    print("• Download direto do YouTube em FLAC 24kHz Mono")
    print("• Normalização de áudio com FFmpeg")
    print("• Segmentação natural baseada em pausas da fala (10s-1min)")
    print("• Filtro de qualidade MOS (3-tier: ≥3.0, 2.5-3.0, <2.5)")
    print("• Diarização com pyannote.audio 3.1")
    print("• Detecção de sobreposição de vozes (OSD) com pyannote/segmentation-3.0")
    print("• Separação por locutor")
    print("• STT dual: Distil-Whisper + WAV2VEC2-CORAA")
    print("• Validação STT com Levenshtein (80% threshold)")
    print("• Denoiser com DeepFilterNet3")
    print("• Normalização final com Sox")
    print("• Dataset final com nomenclatura padronizada")
    print("• Todos os arquivos salvos localmente")
    print()
    print("Pressione Ctrl+C para parar o servidor")
    print("============================================================")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
