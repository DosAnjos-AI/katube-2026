"""
Main pipeline that orchestrates the complete YouTube audio processing workflow.
"""
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
import json
from datetime import datetime
import shutil
import csv

from config import Config
from audio_segmenter import AudioSegmenter
from diarizer import EnhancedDiarizer
from overlap_detector import OverlapDetector
from speaker_separator import SpeakerSeparator
from stt_whisper import WhisperSTTTranscriber
from stt_wav2vec2 import WAV2VEC2STTTranscriber
from audio_normalizer import AudioNormalizer
from src.marcos_validation.text_normalizer import process_stt_results as normalize_stt_texts
from denoiser import Denoiser
from sox_normalizer import SoxNormalizer
from mos_filter import MOSQualityFilter

logger = logging.getLogger(__name__)

class AudioProcessingPipeline:
    """
    Complete pipeline for YouTube audio processing:
    1. Segment audio intelligently
    2. Perform speaker diarization
    3. Detect voice overlaps
    4. Separate audio by speakers
    5. Prepare for STT processing
    """
    
    def __init__(self, 
                 output_base_dir: Optional[Path] = None,
                 huggingface_token: Optional[str] = None,
                 segment_min_duration: float = 4.0,
                 segment_max_duration: float = 18.0,
                 mos_threshold: float = 2.5,
                 enable_mos_filter: bool = True,
                 use_cuda: bool = True):
        
        # Set up directories
        self.output_base_dir = output_base_dir or Config.OUTPUT_DIR
        Config.create_directories()

        # Use intelligent segmenter with VAD for quality cuts
        self.segmenter = AudioSegmenter(segment_min_duration, segment_max_duration)
        self.diarizer = EnhancedDiarizer(huggingface_token)
        self.overlap_detector = OverlapDetector()
        self.speaker_separator = SpeakerSeparator()
        
        # Initialize filters
        # Completeness filter moved to separate file (src/audio_completeness_filter.py)
        self.enable_completeness_filter = False  # DISABLED - moved to separate file
        
        logger.info("ðŸ” Filtros de Ã¡udio:")
        logger.info("   - Filtro de completude: DESABILITADO (arquivo separado)")
        
        # Initialize MOS quality filter (OBRIGATÃ“RIO)
        self.enable_mos_filter = True  # Sempre habilitado
        logger.info("ðŸ” Inicializando filtro MOS (OBRIGATÃ“RIO)...")
        
        try:
            self.mos_filter = MOSQualityFilter(
                mos_threshold=mos_threshold,
                use_cuda=use_cuda
            )
            logger.info("âœ… Filtro MOS inicializado com sucesso")
        except Exception as e:
            logger.error(f"âŒ ERRO CRÃTICO: Falha ao inicializar filtro MOS: {e}")
            raise RuntimeError(f"Filtro MOS Ã© OBRIGATÃ“RIO e falhou: {e}")
        
        # Initialize STT transcribers (separated models)
        self.enable_stt = True  # Sempre habilitado
        logger.info("ðŸ” Inicializando STT transcribers separados...")
        
        try:
            # Initialize Whisper STT
            self.whisper_stt = WhisperSTTTranscriber(
                whisper_model_name="freds0/distil-whisper-large-v3-ptbr",  # Modelo especializado em PT-BR
                device="cuda" if use_cuda else "cpu",
                huggingface_token=huggingface_token
            )
            logger.info("âœ… Whisper STT transcriber inicializado com sucesso")
            
            # Initialize WAV2VEC2 STT
            self.wav2vec2_stt = WAV2VEC2STTTranscriber(
                wav2vec2_model_name="lgris/wav2vec2-large-xlsr-open-brazilian-portuguese-v2",  # Modelo especializado em PT-BR
                device="cuda" if use_cuda else "cpu"
            )
            logger.info("âœ… WAV2VEC2 STT transcriber inicializado com sucesso")
            
        except Exception as e:
            logger.warning(f"âš ï¸ STT transcribers falharam: {e}")
            logger.warning("âš ï¸ Continuando sem STT - pipeline funcionarÃ¡ normalmente")
            self.enable_stt = False
            self.whisper_stt = None
            self.wav2vec2_stt = None
        
        # Initialize audio normalizer
        self.audio_normalizer = AudioNormalizer(
            target_sample_rate=24000,
            target_format="flac",
            target_channels=1  # Mono
        )
        
        # Initialize denoiser
        self.denoiser = Denoiser(model_name="DeepFilterNet3")
        logger.info("âœ… Denoiser (DeepFilterNet3) inicializado com sucesso")
        
        # Initialize Sox normalizer for final processing
        self.sox_normalizer = SoxNormalizer(
            target_sample_rate=48000,
            target_format="flac",
            target_channels=1,
            normalize_gain=True
        )
        logger.info("âœ… Sox normalizer inicializado com sucesso")
        
        # Pipeline state
        self.current_session = None
        self.session_dir = None
        
    def create_session(self, session_name: Optional[str] = None) -> Path:
        """Create a new processing session directory."""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"
        
        self.current_session = session_name
        self.session_dir = self.output_base_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create minimal subdirectories (only for temporary processing)
        subdirs = ['segments', 'diarization', 'speakers', 'clean', 'overlapping', 'stt_ready']
        for subdir in subdirs:
            (self.session_dir / subdir).mkdir(exist_ok=True)
        
        logger.info(f"ðŸ“ Session local criada: {self.current_session}")
        logger.info(f"Created session: {self.current_session}")
        return self.session_dir
    


    def _save_timestamps_metadata(self, metadata: Dict[str, Any], update: bool = False):
            """
            Save or update timestamps metadata to JSON file.
            
            Args:
                metadata: Metadata dictionary to save
                update: If True, update existing file; if False, create new
            """
            if not self.session_dir:
                logger.error("Session directory not initialized")
                return
            
            # JSON file location: {session_dir}/segments/segments_timestamps.json
            json_path = self.session_dir / 'segments' / 'segments_timestamps.json'
            
            if update and json_path.exists():
                # Load existing data
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                    
                    # Deep merge metadata
                    existing_data.update(metadata)
                    metadata = existing_data
                except Exception as e:
                    logger.warning(f"Failed to load existing metadata: {e}")
            
            # Save metadata
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                logger.info(f"Timestamps metadata saved to: {json_path}")
            except Exception as e:
                logger.error(f"Failed to save timestamps metadata: {e}")
    def cleanup(self, stages_to_clean: Optional[List[str]] = None):
        for stage in stages_to_clean:
            logger.info(f'\n\n\n ==== Limpando a pasta {stage} ===')
            diretories_to_delete = self.session_dir / stage
            if diretories_to_delete.exists():
                try:
                    logger.info(f"\n\n[FINAL CLEAN-UP] Deletando a pasta: {diretories_to_delete}")
                    shutil.rmtree(diretories_to_delete)
                    logger.info(f"âœ… Sucesso: DiretÃ³rio de downloads deletado: {diretories_to_delete}")
                except Exception as e:
                    logger.error(f"âŒ Falha ao deletar o diretÃ³rio de downloads: {e}")
    
    def segment_audio(self, audio_path: Path, use_intelligent_segmentation: bool = True) -> List[Tuple[Path, float, float]]:
        """
        Step 2: Segment audio into manageable chunks for local processing.
        
        Args:
            audio_path: Path to input audio file
            use_intelligent_segmentation: Use intelligent segmentation vs simple chunking
            
        Returns:
            List of tuples (segment_path, absolute_start_time, absolute_end_time)
        """
        logger.info("=== STEP 2: SEGMENTING AUDIO ===")
        
        segments_dir = self.session_dir / 'segments'
        
        if use_intelligent_segmentation:
            # Use intelligent segmentation with VAD for quality cuts - returns timestamps
            segments_with_timestamps = self.segmenter.segment_audio(audio_path, segments_dir)
        else:
            # Simple time-based segmentation fallback
            segments_with_timestamps = self._simple_segment_audio(audio_path, segments_dir)
        
        logger.info(f"Created {len(segments_with_timestamps)} segments")
        
        # Save timestamps metadata to JSON
        self._save_segmentation_metadata(audio_path, segments_with_timestamps)
        
        return segments_with_timestamps


    def _save_segmentation_metadata(self, audio_path: Path, segments_with_timestamps: List[Tuple[Path, float, float]]):
        """
        Save segmentation metadata with absolute timestamps to JSON.
        
        Args:
            audio_path: Path to original audio file
            segments_with_timestamps: List of (segment_path, start_time, end_time) tuples
        """
        import soundfile as sf
        
        # Get audio duration
        try:
            audio_info = sf.info(audio_path)
            total_duration = audio_info.duration
        except Exception as e:
            logger.warning(f"Could not get audio duration: {e}")
            total_duration = 0.0
        
        # Build metadata structure
        metadata = {
            "original_audio": {
                "path": str(audio_path),
                "duration": total_duration
            },
            "segments": {}
        }
        
        # Add each segment info
        for segment_path, start_time, end_time in segments_with_timestamps:
            segment_id = segment_path.stem  # e.g., "segment_000"
            
            metadata["segments"][segment_id] = {
                "file_path": str(segment_path),
                "absolute_start": start_time,
                "absolute_end": end_time,
                "duration": end_time - start_time,
                "speakers": {}  # Will be populated after diarization
            }
        
        # Save to JSON
        self._save_timestamps_metadata(metadata, update=False)
        logger.info(f"Saved segmentation metadata for {len(segments_with_timestamps)} segments")



    def apply_mos_filter(self, segment_paths: List[Path], rejected_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Apply MOS quality filter to audio segments.
        
        Args:
            segment_paths: List of audio segment paths
            rejected_dir: Directory to save rejected segments (optional)
            
        Returns:
            Dictionary with filtering results
        """
        logger.info("=== STEP 4: APPLYING MOS QUALITY FILTER ===")
        
        if not segment_paths:
            logger.warning("âš ï¸ No segments to filter")
            return {
                'filtered_segments': [],
                'rejected_segments': [],
                'total_segments': 0,
                'accepted_count': 0,
                'rejected_count': 0,
                'quality_rate': 0.0
            }
        
        # Apply MOS filter with 3-tier classification
        # COLE AQUI - MudanÃ§a 3
        # Extrair video_id do primeiro segmento
        video_id = None
        if segment_paths:
            first_segment = segment_paths[0].stem
            video_id = first_segment.split('_segment_')[0] if '_segment_' in first_segment else None

        approved_segments, intermediate_segments, rejected_segments = self.mos_filter.filter_audio_segments(
            segment_paths, 
            output_dir=self.session_dir,
            video_id=video_id
        )
        # For pipeline continuation, use approved + intermediate segments (Ã¢â€°Â¥2.5)
        # Intermediate segments will go through denoising
        accepted_segments = approved_segments + intermediate_segments
        # Generate quality report
        quality_report = self.mos_filter.get_quality_report(segment_paths)
        logger.info(f"ðŸ“Š MOS Quality Report: {quality_report}")
        
        # Log detailed results
        logger.info(f"ðŸŽ¯ MOS filtering results:")
        logger.info(f"   - Total segments analyzed: {len(segment_paths)}")
        logger.info(f"   - Accepted segments: {len(accepted_segments)}")
        logger.info(f"   - Rejected segments: {len(rejected_segments)}")
        logger.info(f"   - Quality acceptance rate: {len(accepted_segments)/len(segment_paths):.1%}")
        
        return {
            'filtered_segments': accepted_segments,
            'rejected_segments': rejected_segments,
            'total_segments': len(segment_paths),
            'accepted_count': len(accepted_segments),
            'rejected_count': len(rejected_segments),
            'quality_rate': len(accepted_segments)/len(segment_paths) if segment_paths else 0.0,
            'quality_report': quality_report
        }
    
    def filter_segments_by_quality(self, segment_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Filter audio segments based on MOS quality scores (OBRIGATÃ“RIO).
        
        Args:
            segment_paths: List of audio segment paths
            
        Returns:
            Tuple of (accepted_segments, rejected_segments)
        """
        if self.mos_filter is None:
            raise RuntimeError("âŒ Filtro MOS nÃ£o foi inicializado (OBRIGATÃ“RIO)")
        
        logger.info(f"ðŸ” Filtrando {len(segment_paths)} segmentos por qualidade MOS (OBRIGATÃ“RIO)...")
        
        # Criar pastas especÃ­ficas para Ã¡udios descartados
        rejected_completeness_dir = self.session_dir / 'audio_descartado_completude'
        rejected_mos_dir = self.session_dir / 'audio_descartado_mos'
        
        # Criar as pastas
        rejected_completeness_dir.mkdir(exist_ok=True)
        rejected_mos_dir.mkdir(exist_ok=True)
        
        # Apply MOS filter with 3-tier classification
        # Extrair video_id do primeiro segmento
        video_id = None
        if segment_paths:
            first_segment = segment_paths[0].stem
            video_id = first_segment.split('_segment_')[0] if '_segment_' in first_segment else None

        approved_segments, intermediate_segments, rejected_segments = self.mos_filter.filter_audio_segments(
            segment_paths, 
            output_dir=self.session_dir,
            video_id=video_id
        )
        # For pipeline continuation, use approved + intermediate segments (Ã¢â€°Â¥2.5)
        # Intermediate segments will go through denoising
        accepted_segments = approved_segments + intermediate_segments
        
        # Generate quality report
        quality_report = self.mos_filter.get_quality_report(segment_paths)
        logger.info(f"ðŸ“Š RelatÃ³rio de Qualidade MOS: {quality_report}")
        
        return accepted_segments, rejected_segments
        
        def process_video_callback(video_url: str, total_videos: int, current_index: int) -> bool:
            """Callback to process each video from the channel."""
            try:
                logger.info(f"ðŸ“¹ Processing video {current_index}/{total_videos}: {video_url}")
                
                # Process single video through pipeline
                result = self.process_single_video(video_url)
                
                # Update progress if callback provided
                if progress_callback:
                    # Consider it success if we processed it, even if no segments
                    # Local processing success check
                    is_success = result.get('success', False) or result.get('warning') is not None
                    progress_callback(video_url, is_success, total_videos, current_index)
                
                return result.get('success', False)
                
            except Exception as e:
                logger.error(f"âŒ Error processing video {video_url}: {e}")
                if progress_callback:
                    progress_callback(video_url, False, total_videos, current_index)
                return False
        
        return result
    
    def perform_diarization(self, segments: List[Path], num_speakers: Optional[int] = None) -> Dict[str, Any]:
        """
        Step 3: Perform speaker diarization on segments.
        
        Args:
            segments: List of audio segment paths
            num_speakers: Hint for number of speakers
            
        Returns:
            Dictionary with diarization results
        """
        logger.info("=== STEP 3: PERFORMING SPEAKER DIARIZATION ===")
        
        diarization_dir = self.session_dir / 'diarization'
        
        # Process segments in batch
        results = self.diarizer.diarize_batch(
            segments, 
            diarization_dir, 
            save_rttm=True
        )
        
        # Summarize results
        successful = [k for k, v in results.items() if 'error' not in v]
        failed = [k for k, v in results.items() if 'error' in v]
        
        logger.info(f"Diarization completed: {len(successful)} successful, {len(failed)} failed")
        
        if failed:
            logger.warning(f"Failed files: {failed}")
        
        return results
    
    def detect_overlaps(self, segments: List[Path]) -> Tuple[List[Path], List[Path]]:
        """
        Step 4: Detect and separate overlapping vs clean segments.
        
        Args:
            segments: List of segment paths
            
        Returns:
            Tuple of (clean_segments, overlapping_segments)
        """
        logger.info("=== STEP 4: DETECTING VOICE OVERLAPS ===")
        
        overlap_dir = self.session_dir / 'overlapping'
        clean_dir = self.session_dir / 'clean'
        
        # Filter segments based on overlap detection
        clean_segments, overlapping_segments = self.overlap_detector.filter_overlapping_segments(
            segments, self.session_dir
        )
        
        logger.info(f"Overlap detection: {len(clean_segments)} clean, {len(overlapping_segments)} overlapping")
        
        return clean_segments, overlapping_segments
    
    
    def separate_speakers(self, diarization_results: Dict[str, Any], enhance_audio: bool = True) -> Dict[str, Any]:
        """
        Step 5: Separate audio by speakers using diarization results.
        
        Args:
            diarization_results: Results from diarization step
            enhance_audio: Apply audio enhancement
            
        Returns:
            Dictionary with speaker separation results
        """
        logger.info("=== STEP 5: SEPARATING SPEAKERS ===")
        
        speakers_dir = self.session_dir / 'speakers'
        separation_results = {}
        
        # Load timestamps metadata
        json_path = self.session_dir / 'segments' / 'segments_timestamps.json'
        timestamps_metadata = {}
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    timestamps_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load timestamps metadata: {e}")
        
        for audio_path_str, diar_result in diarization_results.items():
            if 'error' in diar_result:
                continue
            
            try:
                audio_path = Path(audio_path_str)
                rttm_path = Path(diar_result['rttm_path']) if diar_result.get('rttm_path') else None
                
                if not rttm_path or not rttm_path.exists():
                    logger.warning(f"No RTTM file for {audio_path.name}")
                    continue
                
                # Get segment offset from metadata
                segment_id = audio_path.stem
                segment_offset = 0.0
                
                if 'segments' in timestamps_metadata and segment_id in timestamps_metadata['segments']:
                    segment_offset = timestamps_metadata['segments'][segment_id].get('absolute_start', 0.0)
                    logger.info(f"Segment {segment_id} offset: {segment_offset:.2f}s")
                
                # Process with speaker separator (with offset)
                result = self.speaker_separator.process_audio_file(
                    audio_path, 
                    rttm_path, 
                    speakers_dir / audio_path.stem,
                    enhance=enhance_audio,
                    create_compilations=True,
                    segment_offset=segment_offset
                )
                
                separation_results[audio_path_str] = result
                
                # Update metadata with speaker information
                self._update_metadata_with_speakers(segment_id, result, rttm_path)
                
            except Exception as e:
                logger.error(f"Speaker separation failed for {audio_path_str}: {e}")
                separation_results[audio_path_str] = {'error': str(e)}
        
        # Summarize results
        total_speakers = sum(r.get('num_speakers', 0) for r in separation_results.values() if 'error' not in r)
        total_segments = sum(r.get('total_segments', 0) for r in separation_results.values() if 'error' not in r)
        
        logger.info(f"Speaker separation: {total_speakers} speakers, {total_segments} segments")
        
        return separation_results
    
    def _update_metadata_with_speakers(self, segment_id: str, separation_result: Dict[str, Any], rttm_path: Path):
        """
        Update timestamps metadata with speaker information after diarization.
        
        Args:
            segment_id: Segment identifier (e.g., "segment_000")
            separation_result: Result from speaker_separator.process_audio_file()
            rttm_path: Path to RTTM file with diarization data
        """
        import pandas as pd
        
        # Load existing metadata
        json_path = self.session_dir / 'segments' / 'segments_timestamps.json'
        if not json_path.exists():
            logger.warning("Timestamps metadata file not found")
            return
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return
        
        # Get segment offset
        segment_offset = 0.0
        if 'segments' in metadata and segment_id in metadata['segments']:
            segment_offset = metadata['segments'][segment_id].get('absolute_start', 0.0)
        else:
            logger.warning(f"Segment {segment_id} not found in metadata")
            return
        
        # Load RTTM to get speaker timestamps
        try:
            speaker_data = self.speaker_separator.load_diarization_dataframe(rttm_path)
            if speaker_data.empty:
                logger.warning(f"No speaker data in RTTM: {rttm_path}")
                return
            
            # Merge consecutive segments
            merged_data = self.speaker_separator.merge_consecutive_segments(speaker_data)
            
        except Exception as e:
            logger.error(f"Failed to load RTTM data: {e}")
            return
        
        # Build speaker information
        speakers_info = {}
        
        for speaker in merged_data['SPEAKER'].unique():
            speaker_segments = merged_data[merged_data['SPEAKER'] == speaker]
            
            segments_list = []
            for idx, row in speaker_segments.iterrows():
                relative_start = row['START']
                relative_end = row['END']
                
                segments_list.append({
                    "relative_start": relative_start,
                    "relative_end": relative_end,
                    "absolute_start": segment_offset + relative_start,
                    "absolute_end": segment_offset + relative_end,
                    "duration": relative_end - relative_start
                })
            
            speakers_info[speaker] = segments_list
        
        # Update metadata
        metadata['segments'][segment_id]['speakers'] = speakers_info
        
        # Save updated metadata
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Updated metadata with {len(speakers_info)} speakers for segment {segment_id}")
        except Exception as e:
            logger.error(f"Failed to save updated metadata: {e}")

    def prepare_for_stt(self, separation_results: Dict[str, Any]) -> Dict[str, List[Path]]:
        """
        Step 6: Prepare final audio files for STT processing.
        
        Args:
            separation_results: Results from speaker separation
            
        Returns:
            Dictionary of STT-ready files organized by speaker
        """
        logger.info("=== STEP 6: PREPARING FOR STT ===")
        
        stt_dir = self.session_dir / 'stt_ready'
        stt_files = {}
        
        # Collect all speaker files
        for audio_result in separation_results.values():
            if 'error' in audio_result:
                continue
            
            # Use individual segments for STT (better for validation), not compilations
            if 'speaker_files' in audio_result:
                for speaker, speaker_file_list in audio_result['speaker_files'].items():
                    if speaker not in stt_files:
                        stt_files[speaker] = []
                    stt_files[speaker].extend(speaker_file_list)
        
        # Copy files to STT directory and organize
        organized_files = {}
        for speaker, files in stt_files.items():
            speaker_stt_dir = stt_dir / f"speaker_{speaker}"
            speaker_stt_dir.mkdir(exist_ok=True)
            
            organized_files[speaker] = []
            for file_path in files:
                if isinstance(file_path, Path) and file_path.exists():
                    # Copy to STT directory
                    dest_path = speaker_stt_dir / file_path.name
                    if not dest_path.exists():
                        import shutil
                        shutil.copy2(file_path, dest_path)
                    organized_files[speaker].append(dest_path)
        
        # Log summary
        total_files = sum(len(files) for files in organized_files.values())
        logger.info(f"STT preparation: {len(organized_files)} speakers, {total_files} files ready")
        
        return organized_files
    
    def transcribe_audio_segments(self, 
                                 segment_paths: List[Path]) -> Dict[str, Any]:
        """
        Step 7: Transcribe audio segments using Whisper and WAV2VEC2 (separated models).
        
        Args:
            segment_paths: List of audio segment paths to transcribe
            
        Returns:
            Dictionary with transcription results
        """
        logger.info("=== STEP 7: TRANSCRIBING AUDIO SEGMENTS ===")
        
        if not self.enable_stt:
            logger.warning("STT transcribers are disabled, skipping transcription")
            return {"error": "STT transcribers are disabled"}
        
        try:
            # Create STT output directory
            stt_output_dir = self.session_dir / 'stt_results'
            
            # Transcribe with Whisper
            whisper_results = {}
            if self.whisper_stt:
                logger.info("ðŸŽ¤ Transcribing with Whisper...")
                whisper_results = self.whisper_stt.transcribe_segments(
                    segment_paths=segment_paths,
                    output_dir=stt_output_dir
                )
                logger.info(f"âœ… Whisper transcription completed: {whisper_results['whisper_count']} segments")
            
            # Transcribe with WAV2VEC2
            wav2vec2_results = {}
            if self.wav2vec2_stt:
                logger.info("ðŸŽ¤ Transcribing with WAV2VEC2...")
                wav2vec2_results = self.wav2vec2_stt.transcribe_segments(
                    segment_paths=segment_paths,
                    output_dir=stt_output_dir
                )
                logger.info(f"âœ… WAV2VEC2 transcription completed: {wav2vec2_results['wav2vec2_count']} segments")
            
            # Combine results
            combined_results = {
                "whisper_results": whisper_results.get("whisper_results", []),
                "wav2vec2_results": wav2vec2_results.get("wav2vec2_results", []),
                "whisper_dir": whisper_results.get("whisper_dir", ""),
                "wav2vec2_dir": wav2vec2_results.get("wav2vec2_dir", ""),
                "total_segments": len(segment_paths),
                "whisper_count": whisper_results.get("whisper_count", 0),
                "wav2vec2_count": wav2vec2_results.get("wav2vec2_count", 0)
            }
            
            logger.info(f"Transcription completed: {combined_results['whisper_count']} Whisper, {combined_results['wav2vec2_count']} WAV2VEC2")
            
            # Step 7.5: Normalize STT texts for validation
            logger.info("=== STEP 7.5: NORMALIZING STT TEXTS ===")
            try:
                normalization_result = normalize_stt_texts(str(self.session_dir))
                
                if normalization_result.get('success'):
                    logger.info(f"Text normalization completed:")
                    logger.info(f"   - Total videos: {normalization_result.get('total_videos', 0)}")
                    logger.info(f"   - Total segments: {normalization_result.get('total_segments', 0)}")
                    logger.info(f"   - Output files: {len(normalization_result.get('output_files', []))}")
                    
                    combined_results['normalization'] = normalization_result
                else:
                    logger.warning(f"Text normalization failed: {normalization_result.get('error')}")
                    combined_results['normalization'] = {"error": normalization_result.get('error')}
                    
            except Exception as e:
                logger.error(f"Error in text normalization: {e}")
                combined_results['normalization'] = {"error": str(e)}

            # Step 8: Validate STT transcriptions with Levenshtein + MOS
            logger.info("=== STEP 8: VALIDATING STT TRANSCRIPTIONS ===")
            if normalization_result.get('success'):
                try:
                    # Importar funcao de validacao
                    from src.marcos_validation.validador_transcricao import validate_normalized_texts
                    
                    # Buscar arquivo JSON normalizado
                    normalized_json_path = normalization_result.get('output_file')
                    
                    if normalized_json_path:
                        # Executar validacao (Levenshtein + MOS)
                        validation_result = validate_normalized_texts(normalized_json_path)
                        
                        if validation_result.get('success'):
                            combined_results['validation'] = validation_result
                            logger.info(f"STT validation completed:")
                            logger.info(f"   - Average similarity: {validation_result.get('average_similarity', 0):.3f}")
                            logger.info(f"   - MOS scores found: {validation_result.get('mos_scores_found', 0)}/{validation_result.get('validated_segments', 0)}")
                            logger.info(f"   - Output file: {validation_result.get('output_file')}")
                        else:
                            logger.warning(f"Text validation failed: {validation_result.get('error')}")
                            combined_results['validation'] = {"error": validation_result.get('error')}
                    else:
                        logger.warning("Normalization output file not found, skipping validation")
                        combined_results['validation'] = {"error": "No normalized file to validate"}
                        
                except Exception as e:
                    logger.error(f"Error in text validation: {e}")
                    combined_results['validation'] = {"error": str(e)}
            else:
                logger.warning("Skipping validation - normalization failed")
                combined_results['validation'] = {"error": "Normalization failed"}
            

            # Step 9: Filter by similarity threshold and MOS, then apply denoising
            if combined_results.get('validation', {}).get('success'):
                try:
                    validation_json_path = combined_results['validation'].get('output_file')
                    
                    if validation_json_path:
                        filter_result = self.filter_and_denoise_segments(
                            validation_json_path=validation_json_path,
                            output_dir=self.session_dir,
                            source_audio_path=self.source_audio_path,
                            similarity_threshold=0.80,
                            mos_range=(2.5, 3.0)
                        )
                        combined_results['filter_and_denoise'] = filter_result
                        logger.info(f"Filtering and denoising completed:")
                        logger.info(f"   - Approved count: {filter_result.get('approved_count', 0)}")
                        logger.info(f"   - Denoised count: {filter_result.get('denoised_success', 0)}")
                        
                        # Step 10: Sox normalization of approved audios
                        if filter_result.get('success'):
                            try:
                                audios_finais_dir = Path(filter_result.get('audios_finais_dir'))
                                video_id = filter_result.get('video_id')
                                
                                sox_result = self.sox_normalize_approved_audios(
                                    audios_finais_dir=audios_finais_dir,
                                    video_id=video_id
                                )
                                combined_results['sox_normalization'] = sox_result
                                
                                if sox_result.get('success'):
                                    logger.info(f"Sox normalization completed:")
                                    logger.info(f"   - Normalized: {sox_result.get('success_count', 0)}/{sox_result.get('total_files', 0)}")
                                    logger.info(f"   - Output: {sox_result.get('output_dir')}")

                                # Step 11: Criar/atualizar dataset.csv
                                    try:
                                        # Caminho base do dataset (assumindo estrutura katube-novo/dataset)
                                        project_root = Path(__file__).parent.parent
                                        dataset_base = project_root / 'dataset'
                                        
                                        # Criar pasta dataset se nao existir
                                        dataset_base.mkdir(parents=True, exist_ok=True)
                                        
                                        # Caminho do JSON final
                                        final_json_path = Path(filter_result.get('final_json_path'))
                                        
                                        logger.info("Iniciando criacao/atualizacao do dataset.csv...")
                                        
                                        csv_result = self.create_or_update_dataset_csv(
                                            final_json_path=final_json_path,
                                            video_id=video_id,
                                            audio_dataset_base=dataset_base
                                        )
                                        
                                        combined_results['dataset_csv'] = csv_result
                                        
                                        if csv_result.get('success'):
                                            logger.info(f"Dataset CSV atualizado:")
                                            logger.info(f"   - Segmentos adicionados: {csv_result.get('approved_count', 0)}")
                                            logger.info(f"   - Arquivo: {csv_result.get('csv_path')}")
                                            logger.info("Copiando JSON final para historico...")
                                            
                                            historico_dir = dataset_base / 'historico_dataset'
                                            historico_dir.mkdir(parents=True, exist_ok=True)
                                            
                                            # Caminho de destino: apenas {video_id}.json
                                            json_dest = historico_dir / f"{video_id}.json"
                                            
                                            # Validar existencia do JSON origem
                                            if not final_json_path.exists():
                                                error_msg = f"ERRO CRITICO: JSON final nao encontrado para copiar: {final_json_path}"
                                                logger.error(error_msg)
                                                raise FileNotFoundError(error_msg)
                                            
                                            # Se destino ja existe, gerar nome com sufixo
                                            if json_dest.exists():
                                                counter = 2
                                                while json_dest.exists():
                                                    json_dest = historico_dir / f"{video_id}_v{counter}.json"
                                                    counter += 1
                                                logger.warning(f"JSON ja existe, salvando como: {json_dest.name}")
                                            
                                            # Copiar JSON
                                            shutil.copy2(final_json_path, json_dest)
                                            logger.info(f"JSON copiado para historico: {json_dest}")
                                            
                                            # Adicionar info ao resultado
                                            csv_result['historico_json'] = str(json_dest)


                                        else:
                                            logger.error(f"Falha ao atualizar dataset CSV: {csv_result.get('error')}")
                                            
                                    except Exception as e:
                                        logger.error(f"ERRO CRITICO ao criar dataset CSV: {e}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                        combined_results['dataset_csv'] = {"error": str(e)}


                                else:
                                    # Step 11: Criar/atualizar dataset.csv
                                    try:
                                        # Caminho base do dataset (assumindo estrutura katube-novo/dataset)
                                        project_root = Path(__file__).parent.parent
                                        dataset_base = project_root / 'dataset'
                                        
                                        # Criar pasta dataset se nao existir
                                        dataset_base.mkdir(parents=True, exist_ok=True)
                                        
                                        # Caminho do JSON final
                                        final_json_path = Path(filter_result.get('final_json_path'))
                                        
                                        logger.info("Iniciando criacao/atualizacao do dataset.csv...")
                                        
                                        csv_result = self.create_or_update_dataset_csv(
                                            final_json_path=final_json_path,
                                            video_id=video_id,
                                            audio_dataset_base=dataset_base
                                        )
                                        
                                        combined_results['dataset_csv'] = csv_result
                                        
                                        if csv_result.get('success'):
                                            logger.info(f"Dataset CSV atualizado:")
                                            logger.info(f"   - Segmentos adicionados: {csv_result.get('approved_count', 0)}")
                                            logger.info(f"   - Arquivo: {csv_result.get('csv_path')}")
                                            logger.info("Copiando JSON final para historico...")
                                            
                                            historico_dir = dataset_base / 'historico_dataset'
                                            historico_dir.mkdir(parents=True, exist_ok=True)
                                            
                                            # Caminho de destino: apenas {video_id}.json
                                            json_dest = historico_dir / f"{video_id}.json"
                                            
                                            # Validar existencia do JSON origem
                                            if not final_json_path.exists():
                                                error_msg = f"ERRO CRITICO: JSON final nao encontrado para copiar: {final_json_path}"
                                                logger.error(error_msg)
                                                raise FileNotFoundError(error_msg)
                                            
                                            # Se destino ja existe, gerar nome com sufixo
                                            if json_dest.exists():
                                                counter = 2
                                                while json_dest.exists():
                                                    json_dest = historico_dir / f"{video_id}_v{counter}.json"
                                                    counter += 1
                                                logger.warning(f"JSON ja existe, salvando como: {json_dest.name}")
                                            
                                            # Copiar JSON
                                            shutil.copy2(final_json_path, json_dest)
                                            logger.info(f"JSON copiado para historico: {json_dest}")
                                            
                                            # Adicionar info ao resultado
                                            csv_result['historico_json'] = str(json_dest)


                                        else:
                                            logger.error(f"Falha ao atualizar dataset CSV: {csv_result.get('error')}")
                                            
                                    except Exception as e:
                                        logger.error(f"ERRO CRITICO ao criar dataset CSV: {e}")
                                        import traceback
                                        logger.error(traceback.format_exc())
                                        combined_results['dataset_csv'] = {"error": str(e)}
                                        # Nao abortar pipeline, apenas registrar erro
                                    logger.warning(f"Sox normalization failed: {sox_result.get('error')}")
                                    
                            except Exception as e:
                                logger.error(f"Error in Sox normalization: {e}")
                                combined_results['sox_normalization'] = {"error": str(e)}
                    else:
                        logger.warning("Validation output file not found, skipping filter and denoise")
                        
                except Exception as e:
                    logger.error(f"Error in filter and denoise: {e}")
                    combined_results['filter_and_denoise'] = {"error": str(e)}
            return combined_results
            
        except Exception as e:
            logger.error(f"Error in transcription step: {e}")
            return {"error": str(e)}
    
    def process_local_audio(self,  
                           audio_path: Path, 
                           num_speakers: Optional[int] = None,
                           enhance_audio: bool = True,
                           use_intelligent_segmentation: bool = True,
                           session_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete pipeline: process a audio through all steps.
        
        Args:
            audio_path: audio to segment and process
            custom_filename: Custom filename for downloaded audio
            num_speakers: Hint for number of speakers
            enhance_audio: Apply audio enhancement
            use_intelligent_segmentation: Use intelligent vs simple segmentation
            session_name: Custom session name
            
        Returns:
            Dictionary with complete processing results
        """
        start_time = time.time()
        
        logger.info("=== STARTING COMPLETE PIPELINE ===")
        logger.info(f"Processing file: {audio_path}")
        
        #ValidaÃ§Ã£o do Ã¡udio de entrada
        if not audio_path.exists():
            logger.error(f"âŒ Audio directory does not exist: {audio_path}")
            return {'success': False, 'error': f"Audio file does not exist: {audio_path}"}
        
        flac_files = list(audio_path.glob('*.flac'))
        if not flac_files:
            error_msg = f"Nenhum arquivo .flac encontrado no diretÃ³rio: {audio_path}"
            logger.error(f"âŒ {error_msg}")
            raise FileNotFoundError(error_msg)
        source_audio_path = flac_files[0]
        self.source_audio_path = source_audio_path  # Armazenar para uso posterior

        try:
            # Create session
            session_name_resolved = session_name or audio_path.stem
            session_dir = self.create_session(session_name_resolved)
            
            # Step 1: Segment
            segments_with_timestamps = self.segment_audio(source_audio_path, use_intelligent_segmentation)
            
            # Extract paths for processing (filters expect List[Path])
            segments = [seg_path for seg_path, _, _ in segments_with_timestamps]
            
            # Step 2: Apply completeness filter (DISABLED - moved to separate file)
            # Completeness filter is now in src/audio_completeness_filter.py
            # if self.enable_completeness_filter:
            #     completeness_rejected_dir = session_dir / 'audio_descartado_completude'
            #     completeness_result = self.apply_completeness_filter(segments, rejected_dir=completeness_rejected_dir)
            #     segments = completeness_result['complete_segments']
            #     logger.info(f"Completeness filter: {len(segments)} segments passed (filtered {completeness_result['cut_count']} cut segments)")
            
            # Step 3: Apply MOS filter
            if self.enable_mos_filter:
                try:
                    mos_rejected_dir = session_dir / 'audio_descartado_mos'
                    mos_result = self.apply_mos_filter(segments, rejected_dir=mos_rejected_dir)
                    segments = mos_result['filtered_segments']
                    logger.info(f"MOS filter: {len(segments)} segments passed")
                except Exception as e:
                    logger.error(f"MOS filter failed: {e}")
                    return {'success': False, 'error': f"MOS filter failed: {str(e)}"}
            
            # Step 4: Diarization (ANTES do STT)
            diarization_results = self.perform_diarization(segments, num_speakers)
            
            # Step 5: Overlap detection (ANTES do STT)
            clean_segments, overlapping_segments = self.detect_overlaps(segments)
            
            # Step 6: Speaker separation (ANTES do STT)
            separation_results = self.separate_speakers(diarization_results, enhance_audio)
            
            # Step 7: STT preparation (ANTES do STT)
            stt_files = self.prepare_for_stt(separation_results)
            
            # Step 8: Apply STT transcription
            stt_result = {}
            if self.enable_stt:
                try:
                    # Flatten stt_files dictionary to list of paths
                    segment_paths = []
                    if stt_files:
                        for speaker_files in stt_files.values():
                            segment_paths.extend(speaker_files)
                    else:
                        segment_paths = segments

                    stt_result = self.transcribe_audio_segments(segment_paths)
                    logger.info(f"âœ… STT transcription completed: {stt_result.get('whisper_count', 0)} Whisper, {stt_result.get('wav2vec2_count', 0)} WAV2VEC2")
                    
                    # Check if validation and filtering were applied
                    if 'validation' in stt_result and 'filter_and_denoise' in stt_result:
                        validation_info = stt_result['validation']
                        filter_info = stt_result['filter_and_denoise']
                        logger.info(f"ðŸ“Š STT Validation: {validation_info.get('average_similarity', 0):.3f} avg similarity")
                        logger.info(f"ðŸ“Š Filtro 80%: {filter_info.get('validated_count', 0)} validados, {filter_info.get('denoised_count', 0)} denoised")
                    
                except Exception as e:
                    logger.error(f"âŒ STT transcription failed: {e}")
                    # Continue without STT if it fails
                    logger.warning("Continuing pipeline without STT transcription")
            
            # Step 09: Move approved segments to final directory
            try:
                final_segments_dir = session_dir / 'segments_aprovados'
                final_segments_dir.mkdir(exist_ok=True)
                                
                final_segments = []
                import shutil

                # Flatten stt_files dictionary to list if needed
                if stt_files:
                    segments_to_move = []
                    for speaker_files in stt_files.values():
                        segments_to_move.extend(speaker_files)
                else:
                    segments_to_move = segments

                for segment in segments_to_move:
                    if isinstance(segment, Path):
                        final_path = final_segments_dir / segment.name
                        shutil.copy2(segment, final_path)
                        final_segments.append(final_path)
                
                segments = final_segments
                logger.info(f"âœ… {len(segments)} segments moved to final approved directory")
            except Exception as e:
                logger.warning(f"âš ï¸ Could not move segments to final directory: {e}")
            
            # Final results
            processing_time = time.time() - start_time
            
            results = {
                'session_name': self.current_session,
                'session_dir': str(session_dir),
                'processing_time': processing_time,
                'downloaded_audio': str(audio_path),
                'num_segments': len(segments),
                'num_clean_segments': len(clean_segments),
                'num_overlapping_segments': len(overlapping_segments),
                'diarization_results': diarization_results,
                'separation_results': separation_results,
                'stt_ready_files': stt_files,
                'stt_results': stt_result,  # Include STT validation and filtering results
                'statistics': self._generate_statistics(stt_files, separation_results)
            }
            
            # Save results to JSON
            results_file = session_dir / 'pipeline_results.json'
            with open(results_file, 'w', encoding='utf-8') as f:
                # Convert Path objects to strings for JSON serialization
                json_results = self._prepare_for_json(results)
                json.dump(json_results, f, indent=2, ensure_ascii=False)
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            logger.info(f"Processing time: {processing_time:.2f}s")
            logger.info(f"Results saved to: {results_file}")
            
            logger.info("=== LIMPEZA: REMOVENDO PASTA DE SESSAO ===")
            try:
                if session_dir.exists():
                    shutil.rmtree(session_dir)
                    logger.info(f"Pasta de sessao removida com sucesso: {session_dir}")
                else:
                    logger.warning(f"Pasta de sessao nao encontrada: {session_dir}")
            except Exception as e:
                logger.error(f"Erro ao remover pasta de sessao: {e}")
                logger.warning("Continuando mesmo com falha na limpeza...")

            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def _simple_segment_audio(self, audio_path: Path, output_dir: Path) -> List[Path]:
        """Simple time-based segmentation fallback."""
        import librosa
        import soundfile as sf
        
        audio, sr = librosa.load(audio_path, sr=self.segmenter.sample_rate, mono=True)
        duration = len(audio) / sr
        
        segments = []
        segment_duration = (self.segmenter.min_duration + self.segmenter.max_duration) / 2
        
        for i, start in enumerate(range(0, int(duration), int(segment_duration))):
            end = min(start + segment_duration, duration)
            
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            segment_audio = audio[start_sample:end_sample]
            
            filename = f"{audio_path.stem}_segment_{i:04d}.{Config.AUDIO_FORMAT}"
            segment_path = output_dir / filename
            
            sf.write(segment_path, segment_audio, sr)
            segments.append(segment_path)
        
        return segments
    
    # COLAR AQUI 1

    def _generate_statistics(self, stt_files: Dict[str, List[Path]], 
                           separation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate processing statistics."""
        import soundfile as sf
        
        stats = {
            'num_speakers': len(stt_files),
            'total_stt_files': sum(len(files) for files in stt_files.values()),
            'speakers': {}
        }
        
        for speaker, files in stt_files.items():
            total_duration = 0
            for file_path in files:
                try:
                    if isinstance(file_path, Path) and file_path.exists():
                        with sf.SoundFile(file_path) as f:
                            total_duration += len(f) / f.samplerate
                except:
                    pass
            
            stats['speakers'][speaker] = {
                'num_files': len(files),
                'total_duration': total_duration,
                'avg_file_duration': total_duration / len(files) if files else 0
            }
        
        return stats
    
    # Completeness filter method moved to separate file (src/audio_completeness_filter.py)
    
    def validate_stt_transcriptions(self, 
                                   whisper_results: List[Dict], 
                                   wav2vec2_results: List[Dict],
                                   output_dir: Path) -> Dict[str, Any]:
        """
        Step 8: Validate STT transcriptions using Levenshtein distance.
        
        Args:
            whisper_results: List of Whisper transcription results
            wav2vec2_results: List of WAV2VEC2 transcription results
            output_dir: Directory to save validation results
            
        Returns:
            Dictionary with validation results
        """
        logger.info("=== STEP 8: VALIDATING STT TRANSCRIPTIONS ===")
        
        if not whisper_results or not wav2vec2_results:
            logger.warning("âš ï¸ No STT results to validate")
            return {"error": "No STT results to validate"}
        
        try:
            # Create validation output directory
            validation_dir = output_dir / 'validation_results'
            validation_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata files for validation (exactly as validation.py expects)
            whisper_metadata_file = validation_dir / 'metadata_whisper.csv'
            wav2vec2_metadata_file = validation_dir / 'metadata_wav2vec2.csv'
            validation_output_file = validation_dir / 'validation_results.csv'
            
            # Validate that both STT models processed the same segments
            if len(whisper_results) != len(wav2vec2_results):
                logger.warning(f"âš ï¸ Different number of segments: Whisper={len(whisper_results)}, WAV2VEC2={len(wav2vec2_results)}")
                logger.warning("âš ï¸ Skipping validation - both models must process same segments")
                return {"error": "Different number of segments processed by STT models"}
            
            # Write Whisper metadata (format: filename | text - exactly as validation.py expects)
            with open(whisper_metadata_file, 'w', encoding='utf-8') as f:
                for result in whisper_results:
                    filename = Path(result['file']).stem.replace('_whisper', '').strip()
                    text = result['transcription'].strip()
                    logger.debug(f"Whisper metadata: '{filename}' | '{text[:50]}...'")
                    f.write(f"{filename}|{text}\n")
            
            # Write WAV2VEC2 metadata (format: filename | text - exactly as validation.py expects)  
            with open(wav2vec2_metadata_file, 'w', encoding='utf-8') as f:
                for result in wav2vec2_results:
                    filename = Path(result['file']).stem.replace('_wav2vec2', '').strip()
                    text = result['transcription'].strip()
                    logger.debug(f"WAV2VEC2 metadata: '{filename}' | '{text[:50]}...'")
                    f.write(f"{filename}|{text}\n")
                    
            logger.info(f"ðŸ“ Created metadata files:")
            logger.info(f"   - Whisper: {len(whisper_results)} entries")
            logger.info(f"   - WAV2VEC2: {len(wav2vec2_results)} entries")
            
            # Run validation using the professor's validator (exactly as validation.py expects)
            logger.info("ðŸ” Running STT validation with Levenshtein distance...")
            logger.info(f"   - Input file 1: {whisper_metadata_file}")
            logger.info(f"   - Input file 2: {wav2vec2_metadata_file}")
            logger.info(f"   - Output file: {validation_output_file}")
            
            # Use Marcos validation instead of the old one
            validation_success = marcos_create_validation_file(
                input_file1=str(whisper_metadata_file),
                input_file2=str(wav2vec2_metadata_file),
                prefix_filepath="",  # Empty prefix as in validation.py
                output_file=str(validation_output_file)
            )
            
            # Initialize variables
            validation_results = []
            avg_similarity = 0
            min_similarity = 0
            max_similarity = 0
            
            if validation_success:
                logger.info(f"âœ… STT validation completed: {validation_output_file}")
                
                # Read validation results (exactly as validation.py produces)
                validation_results = []
                with open(validation_output_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    logger.info(f"ðŸ“„ Validation file has {len(lines)} lines")
                    
                    if len(lines) > 0:
                        logger.info(f"ðŸ“„ Header line: '{lines[0].strip()}'")
                        
                    # Skip header if present: filename|subtitle|transcript|similarity
                    data_lines = lines[1:] if len(lines) > 1 else lines
                    
                    for i, line in enumerate(data_lines):
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                            
                        parts = line.split('|')  # Split by '|' as validation.py uses (no spaces)
                        logger.debug(f"Line {i+1}: '{line}' -> {len(parts)} parts")
                        
                        if len(parts) >= 4:
                            try:
                                similarity = float(parts[3].strip())
                                validation_results.append({
                                    'filename': parts[0].strip(),
                                    'whisper_text': parts[1].strip(),
                                    'wav2vec2_text': parts[2].strip(),
                                    'similarity': similarity
                                })
                                logger.debug(f"âœ… Added validation result: {parts[0].strip()} -> {similarity}")
                            except ValueError as e:
                                logger.warning(f"âš ï¸ Could not parse similarity '{parts[3]}': {e}")
                        else:
                            logger.warning(f"âš ï¸ Invalid line format (expected 4 parts, got {len(parts)}): '{line}'")
                
                # Calculate statistics
                similarities = [r['similarity'] for r in validation_results]
                avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                min_similarity = min(similarities) if similarities else 0
                max_similarity = max(similarities) if similarities else 0
                
                logger.info(f"ðŸ“Š Validation statistics:")
                logger.info(f"   - Total segments validated: {len(validation_results)}")
                logger.info(f"   - Average similarity: {avg_similarity:.3f}")
                logger.info(f"   - Min similarity: {min_similarity:.3f}")
                logger.info(f"   - Max similarity: {max_similarity:.3f}")
                
            return {
                    'success': True,
                    'validation_file': str(validation_output_file),
                    'total_segments': len(validation_results),
                    'average_similarity': avg_similarity,
                    'min_similarity': min_similarity,
                    'max_similarity': max_similarity,
                    'validation_results': validation_results
                }
            
            # If validation failed
            if not validation_success:
                logger.error("âŒ STT validation failed")
                return {"error": "STT validation failed"}
                
        except Exception as e:
            logger.error(f"Error in STT validation: {e}")
            return {"error": str(e)}
    
    def _reconstruct_segments_from_original(self,
                                           final_json: Dict,
                                           source_audio_path: Path,
                                           session_dir: Path,
                                           mos_range: Tuple[float, float] = (2.5, 3.0)) -> Dict[str, Any]:
        """
        Step 8.5: Reconstroi segmentos aprovados extraindo diretamente do audio original.
        Preserva qualidade maxima do audio source.
        
        Args:
            final_json: JSON completo com todos metadados dos segmentos
            source_audio_path: Caminho do audio original de alta qualidade
            session_dir: Diretorio da sessao
            mos_range: Tupla (min, max) para decidir se aplica denoiser
            
        Returns:
            Dicionario com resultados da reconstrucao
        """
        import soundfile as sf
        import librosa
        import numpy as np
        
        logger.info("=== STEP 8.5: RECONSTRUCTING SEGMENTS FROM ORIGINAL ===")
        logger.info(f"Source audio: {source_audio_path}")
        
        try:
            # Validar audio original existe
            if not source_audio_path.exists():
                error_msg = f"Audio original nao encontrado: {source_audio_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Carregar audio original completo
            logger.info("Carregando audio original...")
            source_audio, source_sr = sf.read(str(source_audio_path))
            source_duration = len(source_audio) / source_sr
            
            logger.info(f"Audio original carregado:")
            logger.info(f"  Sample rate: {source_sr} Hz")
            logger.info(f"  Duracao: {source_duration:.2f}s")
            logger.info(f"  Shape: {source_audio.shape}")
            
            # Criar pastas intermediarias
            audios_reconstruidos = session_dir / 'audios_reconstruidos'
            audios_para_denoiser = session_dir / 'audios_para_denoiser'
            audios_finais = session_dir / 'audios_finais_aprovados'
            
            audios_reconstruidos.mkdir(parents=True, exist_ok=True)
            audios_para_denoiser.mkdir(parents=True, exist_ok=True)
            audios_finais.mkdir(parents=True, exist_ok=True)
            
            logger.info("Pastas intermediarias criadas:")
            logger.info(f"  - {audios_reconstruidos}")
            logger.info(f"  - {audios_para_denoiser}")
            logger.info(f"  - {audios_finais}")
            
            # Filtrar apenas segmentos aprovados
            segments = final_json.get('segments', {})
            approved_segments = {
                seg_id: seg_data 
                for seg_id, seg_data in segments.items() 
                if seg_data.get('status') == 'approved'
            }
            
            if not approved_segments:
                logger.warning("Nenhum segmento aprovado para reconstruir")
                return {
                    "success": True,
                    "reconstructed_count": 0,
                    "message": "Nenhum segmento aprovado"
                }
            
            logger.info(f"Segmentos aprovados para reconstruir: {len(approved_segments)}")
            
            # Contadores
            reconstructed_count = 0
            for_denoiser_count = 0
            direct_final_count = 0
            skipped_count = 0
            
            # Processar cada segmento aprovado
            for segment_id, seg_data in approved_segments.items():
                try:
                    # Validar MOS score
                    mos_score = seg_data.get('mos_score')
                    if mos_score is None:
                        logger.warning(f"Segmento {segment_id}: MOS = None, rejeitando")
                        skipped_count += 1
                        continue
                    
                    # Extrair timestamps absolutos
                    absolute_start = seg_data.get('absolute_start')
                    absolute_end = seg_data.get('absolute_end')
                    
                    if absolute_start is None or absolute_end is None:
                        logger.warning(f"Segmento {segment_id}: timestamps ausentes, pulando")
                        skipped_count += 1
                        continue
                    
                    # Validar timestamps
                    if absolute_start < 0 or absolute_end > source_duration:
                        logger.warning(f"Segmento {segment_id}: timestamps invalidos "
                                     f"({absolute_start:.2f}-{absolute_end:.2f}s vs {source_duration:.2f}s)")
                        skipped_count += 1
                        continue
                    
                    if absolute_start >= absolute_end:
                        logger.warning(f"Segmento {segment_id}: start >= end ({absolute_start:.2f} >= {absolute_end:.2f})")
                        skipped_count += 1
                        continue
                    
                    # Converter timestamps para indices de samples
                    start_sample = int(absolute_start * source_sr)
                    end_sample = int(absolute_end * source_sr)
                    
                    # Extrair segmento do audio original
                    segment_audio = source_audio[start_sample:end_sample]
                    
                    if len(segment_audio) == 0:
                        logger.warning(f"Segmento {segment_id}: extracao resultou em audio vazio")
                        skipped_count += 1
                        continue
                    
                    # Salvar segmento reconstruido (preservando sample rate original)
                    recon_path = audios_reconstruidos / f"{segment_id}.flac"
                    sf.write(recon_path, segment_audio, source_sr)
                    reconstructed_count += 1
                    
                    logger.debug(f"Reconstruido: {segment_id} ({absolute_start:.2f}-{absolute_end:.2f}s, "
                               f"{len(segment_audio)/source_sr:.2f}s, MOS={mos_score:.2f})")
                    
                    # Decidir roteamento por MOS
                    if mos_range[0] <= mos_score <= mos_range[1]:
                        # Precisa denoising
                        # Upsample para 48kHz se necessario (DeepFilterNet3 requer 48kHz)
                        if source_sr != 48000:
                            logger.debug(f"  Upsampling {segment_id}: {source_sr}Hz -> 48000Hz")
                            segment_audio_48k = librosa.resample(
                                segment_audio, 
                                orig_sr=source_sr, 
                                target_sr=48000
                            )
                        else:
                            segment_audio_48k = segment_audio
                        
                        # Salvar para denoiser
                        para_denoiser_path = audios_para_denoiser / f"{segment_id}.flac"
                        sf.write(para_denoiser_path, segment_audio_48k, 48000)
                        for_denoiser_count += 1
                        
                        logger.debug(f"  -> Para denoiser (MOS {mos_score:.2f} in [{mos_range[0]}, {mos_range[1]}])")
                    
                    else:  # MOS > mos_range[1]
                        # Audio excelente, copia direto para finais (preservando SR original)
                        final_path = audios_finais / f"{segment_id}.flac"
                        shutil.copy2(recon_path, final_path)
                        direct_final_count += 1
                        
                        logger.debug(f"  -> Direto para finais (MOS {mos_score:.2f} > {mos_range[1]})")
                
                except Exception as e:
                    logger.error(f"Erro ao reconstruir {segment_id}: {e}")
                    skipped_count += 1
                    continue
            
            # Log de resultados
            logger.info(f"\n{'='*60}")
            logger.info(f"RECONSTRUCTION COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total aprovados: {len(approved_segments)}")
            logger.info(f"Reconstruidos: {reconstructed_count}")
            logger.info(f"  - Para denoiser (MOS {mos_range[0]}-{mos_range[1]}): {for_denoiser_count}")
            logger.info(f"  - Direto finais (MOS > {mos_range[1]}): {direct_final_count}")
            logger.info(f"Pulados (erros): {skipped_count}")
            logger.info(f"{'='*60}\n")
            
            return {
                "success": True,
                "reconstructed_count": reconstructed_count,
                "for_denoiser_count": for_denoiser_count,
                "direct_final_count": direct_final_count,
                "skipped_count": skipped_count,
                "audios_reconstruidos_dir": str(audios_reconstruidos),
                "audios_para_denoiser_dir": str(audios_para_denoiser),
                "audios_finais_dir": str(audios_finais)
            }
            
        except Exception as e:
            logger.error(f"Erro critico na reconstrucao: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def _extract_timestamps_from_flac_name(self, flac_filename: str) -> Dict[str, Any]:
        """
        Extrai timestamps e informacoes do nome do arquivo FLAC.
        
        Args:
            flac_filename: Nome do arquivo (ex: EhzSC3LWez4_segment_000_SPEAKER_00_1.43_24.41.flac)
            
        Returns:
            Dictionary com segment_id, speaker, absolute_start, absolute_end, duration
        """
        try:
            # Remover extensao
            stem = flac_filename.replace('.flac', '')
            
            # Split por underscore
            parts = stem.split('_')
            
            # Padrao esperado: {video_id}_{segment}_{num}_SPEAKER_{num}_{start}_{end}
            # Ex: ['EhzSC3LWez4', 'segment', '000', 'SPEAKER', '00', '1.43', '24.41']
            
            if len(parts) < 7:
                logger.warning(f"Nome de arquivo invalido (poucos parts): {flac_filename}")
                return None
            
            # Extrair informacoes
            video_id = parts[0]
            segment_id = f"{parts[1]}_{parts[2]}"  # segment_000
            speaker = f"{parts[3]}_{parts[4]}"      # SPEAKER_00
            absolute_start = float(parts[-2])       # Penultimo elemento
            absolute_end = float(parts[-1])         # Ultimo elemento
            duration = absolute_end - absolute_start
            
            return {
                'video_id': video_id,
                'segment_id': segment_id,
                'speaker': speaker,
                'absolute_start': absolute_start,
                'absolute_end': absolute_end,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Erro ao extrair timestamps de {flac_filename}: {e}")
            return None


    def filter_and_denoise_segments(self, 
                                   validation_json_path: str,
                                   output_dir: Path,
                                   source_audio_path: Path,
                                   similarity_threshold: float = 0.80,
                                   mos_range: Tuple[float, float] = (2.5, 3.0)) -> Dict[str, Any]:
        """
        Step 9: Filter segments by similarity AND MOS, apply denoising when needed.
        Creates final JSON with all segment data + utilizou_denoiser status.
        
        Args:
            validation_json_path: Path to validation JSON file
            source_audio_path: Path to original high-quality audio
            output_dir: Session directory
            similarity_threshold: Minimum similarity to accept (default 0.80)
            mos_range: MOS range for denoising (default (2.5, 3.0))
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"=== STEP 9: FILTERING AND DENOISING SEGMENTS ===")
        logger.info(f"Similarity threshold: >= {similarity_threshold}")
        logger.info(f"MOS range for denoising: [{mos_range[0]}, {mos_range[1]}]")
        
        try:
            import json
            import shutil
            from pathlib import Path as PathLib
            
            # Carregar JSON de validacao
            validation_path = PathLib(validation_json_path)
            
            if not validation_path.exists():
                error_msg = f"Validation JSON not found: {validation_json_path}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            with open(validation_path, 'r', encoding='utf-8') as f:
                validation_data = json.load(f)
            
            video_id = validation_data.get('video_id', 'unknown')
            normalized_pairs = validation_data.get('normalized_pairs', {})
            
            if not normalized_pairs:
                error_msg = "No normalized pairs found in validation JSON"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
         

            logger.info(f"Processing {len(normalized_pairs)} segments for video: {video_id}")
            
            # Criar pasta de saida (todos os aprovados ficam aqui)
            denoiser_dir = output_dir / 'audios_denoiser'
            denoiser_dir.mkdir(parents=True, exist_ok=True)
            
            # Criar pastas de rejeitados
            rejected_similarity_dir = output_dir / 'audio_rejeitado_validacao'
            rejected_mos_dir = output_dir / 'audio_rejeitado_mos_range'
            rejected_similarity_dir.mkdir(parents=True, exist_ok=True)
            rejected_mos_dir.mkdir(parents=True, exist_ok=True)
            
            # Contadores para estatisticas
            approved_with_denoise = 0
            approved_without_denoise = 0
            rejected_by_similarity = 0
            rejected_by_mos = 0
            denoised_success = 0
            
            # JSON final que sera salvo
            final_json = {
                "video_id": video_id,
                "total_segments": len(normalized_pairs),
                "approved_count": 0,
                "rejected_count": 0,
                "segments": {}
            }
            
            # Processar cada segmento
            for segment_id, pair_data in normalized_pairs.items():
                similarity = pair_data.get('levenshtein_similarity', 0.0)
                mos_score = pair_data.get('mos_score')
                flac_file = pair_data.get('flac_file')
                
                logger.info(f"\nProcessing: {segment_id}")
                logger.info(f"  Similarity: {similarity:.3f}, MOS: {mos_score}")
                
                # Inicializar campos novos
                utilizou_denoiser = False
                status = None
                
                # Buscar arquivo FLAC original
                audio_file = None
                stt_ready_dir = output_dir / 'stt_ready'
                
                if stt_ready_dir.exists():
                    for speaker_dir in stt_ready_dir.iterdir():
                        if speaker_dir.is_dir():
                            for flac_path in speaker_dir.glob("*.flac"):
                                # Extrair prefixo do segment_id (antes de _stt_)
                                prefix = segment_id.split('_stt_')[0] if '_stt_' in segment_id else segment_id
                                if prefix in flac_path.stem or (flac_file and flac_file in flac_path.name):
                                    audio_file = flac_path
                                    break
                        if audio_file:
                            break
                
                if not audio_file or not audio_file.exists():
                    logger.warning(f"  Audio file not found, skipping")
                    continue
                
                # FILTRO 1: Similaridade
                if similarity < similarity_threshold:
                    status = "rejected_similarity"
                    utilizou_denoiser = False
                    rejected_by_similarity += 1
                    
                    logger.info(f"  REJECTED by similarity ({similarity:.3f} < {similarity_threshold})")
                    
                    # Copiar para pasta de rejeitados
                    rejected_path = rejected_similarity_dir / f"{segment_id}.flac"
                    shutil.copy2(audio_file, rejected_path)
                
                # FILTRO 2: MOS (se passou pelo filtro de similaridade)
                elif mos_score is None:
                    logger.warning(f"  No MOS score available, skipping")
                    continue
                
                elif mos_score < mos_range[0]:
                    status = "rejected_mos"
                    utilizou_denoiser = False
                    rejected_by_mos += 1
                    
                    logger.info(f"  REJECTED by MOS ({mos_score} < {mos_range[0]})")
                    
                    # Copiar para pasta de rejeitados por MOS
                    rejected_path = rejected_mos_dir / f"{segment_id}.flac"
                    shutil.copy2(audio_file, rejected_path)
                
                # APROVADO
                else:
                    status = "approved"
                    
                    # Decidir se aplica denoiser
                    if mos_range[0] <= mos_score <= mos_range[1]:
                        # Precisa denoising
                        utilizou_denoiser = True
                        approved_with_denoise += 1
                        
                        logger.info(f"  APPROVED - will apply DENOISER (MOS in [{mos_range[0]}, {mos_range[1]}])")
                        
                        # Aplicar denoising
                        denoised_path = denoiser_dir / f"{segment_id}.flac"
                        
                        try:
                            logger.info(f"  Applying DeepFilterNet3...")
                            self.denoiser.process_file(
                                str(audio_file),
                                str(denoised_path)
                            )
                            denoised_success += 1
                            logger.info(f"  Denoised successfully")
                        except Exception as e:
                            logger.error(f"  Error during denoising: {e}")
                            # Se falhar, copiar original
                            shutil.copy2(audio_file, denoised_path)
                    
                    else:
                        # MOS > 3.0 - audio excelente, nao precisa denoising
                        utilizou_denoiser = False
                        approved_without_denoise += 1
                        
                        logger.info(f"  APPROVED - NO denoising needed (MOS {mos_score} > {mos_range[1]})")
                        
                        # Copiar original para pasta final
                        approved_path = denoiser_dir / f"{segment_id}.flac"
                        shutil.copy2(audio_file, approved_path)
                        logger.info(f"  Copied original to audios_denoiser/")
                
                # Adicionar ao JSON final (TODOS os segmentos, aprovados e rejeitados)
                # Extrair timestamps do nome do arquivo FLAC
                timestamps_info = self._extract_timestamps_from_flac_name(flac_file) if flac_file else None
                
                # Adicionar ao JSON final (TODOS os segmentos, aprovados e rejeitados)
                final_json["segments"][segment_id] = {
                    "txt_whisper": pair_data.get('txt_whisper'),
                    "txt_wav2vec2": pair_data.get('txt_wav2vec2'),
                    "flac_file": flac_file,
                    "whisper_original": pair_data.get('whisper_original'),
                    "whisper_normalized": pair_data.get('whisper_normalized'),
                    "wav2vec2_original": pair_data.get('wav2vec2_original'),
                    "wav2vec2_normalized": pair_data.get('wav2vec2_normalized'),
                    "levenshtein_similarity": similarity,
                    "mos_score": mos_score,
                    "utilizou_denoiser": utilizou_denoiser,
                    "status": status
                }
                
                # Adicionar timestamps se disponiveis
                if timestamps_info:
                    final_json["segments"][segment_id]["absolute_start"] = timestamps_info['absolute_start']
                    final_json["segments"][segment_id]["absolute_end"] = timestamps_info['absolute_end']
                    final_json["segments"][segment_id]["duration"] = timestamps_info['duration']
                    final_json["segments"][segment_id]["speaker"] = timestamps_info['speaker']
                    final_json["segments"][segment_id]["original_segment"] = timestamps_info['segment_id']
            
            # Calcular estatisticas finais
            approved_total = approved_with_denoise + approved_without_denoise
            rejected_total = rejected_by_similarity + rejected_by_mos
            
            final_json["approved_count"] = approved_total
            final_json["rejected_count"] = rejected_total
            
            # Salvar JSON final
            final_json_path = denoiser_dir / f"{video_id}_final_audio_dataset.json"
            
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JSON final salvo: {final_json_path}")
            
            # Step 8.5: Reconstruir segmentos aprovados do audio original
            logger.info("\n" + "="*60)
            logger.info("INICIANDO RECONSTRUCAO DE SEGMENTOS DO AUDIO ORIGINAL")
            logger.info("="*60)
            
            reconstruction_result = self._reconstruct_segments_from_original(
                final_json=final_json,
                source_audio_path=source_audio_path,
                session_dir=output_dir,
                mos_range=mos_range
            )
            
            if not reconstruction_result.get('success'):
                logger.error(f"Reconstrucao falhou: {reconstruction_result.get('error')}")
                return {
                    'success': False,
                    'error': f"Reconstrucao falhou: {reconstruction_result.get('error')}"
                }
            
            logger.info("Reconstrucao concluida com sucesso!")
            logger.info(f"  - Reconstruidos: {reconstruction_result.get('reconstructed_count')}")
            logger.info(f"  - Para denoiser: {reconstruction_result.get('for_denoiser_count')}")
            logger.info(f"  - Direto finais: {reconstruction_result.get('direct_final_count')}")
            
            # Step 8.6: Aplicar denoiser apenas nos audios de audios_para_denoiser/
            audios_para_denoiser_dir = Path(reconstruction_result.get('audios_para_denoiser_dir'))
            audios_finais_dir = Path(reconstruction_result.get('audios_finais_dir'))
            
            if reconstruction_result.get('for_denoiser_count', 0) > 0:
                logger.info("\n" + "="*60)
                logger.info("STEP 8.6: APPLYING DENOISER TO RECONSTRUCTED SEGMENTS")
                logger.info("="*60)
                
                denoised_success_count = 0
                denoised_failed_count = 0
                
                audios_para_denoiser = list(audios_para_denoiser_dir.glob("*.flac"))
                logger.info(f"Processando {len(audios_para_denoiser)} audios com denoiser...")
                
                for i, audio_file in enumerate(audios_para_denoiser, 1):
                    try:
                        logger.info(f"[{i}/{len(audios_para_denoiser)}] Denoising: {audio_file.name}")
                        
                        # Output temporario do denoiser
                        denoised_temp_path = audios_para_denoiser_dir / f"denoised_{audio_file.name}"
                        
                        # Aplicar DeepFilterNet3
                        self.denoiser.process_file(
                            str(audio_file),
                            str(denoised_temp_path)
                        )
                        
                        # Mover denoised para audios_finais (sobrescrevendo temp)
                        final_path = audios_finais_dir / audio_file.name
                        shutil.move(str(denoised_temp_path), str(final_path))
                        
                        denoised_success_count += 1
                        logger.info(f"  SUCCESS -> {final_path.name}")
                        
                    except Exception as e:
                        logger.error(f"  FAILED: {e}")
                        denoised_failed_count += 1
                        
                        # Fallback: copiar original sem denoising
                        try:
                            fallback_path = audios_finais_dir / audio_file.name
                            shutil.copy2(audio_file, fallback_path)
                            logger.warning(f"  Fallback: copiado sem denoising para {fallback_path.name}")
                        except Exception as e2:
                            logger.error(f"  Fallback tambem falhou: {e2}")
                
                logger.info(f"\n{'='*60}")
                logger.info(f"DENOISING COMPLETED")
                logger.info(f"{'='*60}")
                logger.info(f"Total processados: {len(audios_para_denoiser)}")
                logger.info(f"  - Success: {denoised_success_count}")
                logger.info(f"  - Failed: {denoised_failed_count}")
                logger.info(f"{'='*60}\n")
            else:
                logger.info("Nenhum audio para denoising (todos MOS > 3.0)")
            
            # Step 8.7: Validar audios_finais_aprovados/
            audios_finais = list(audios_finais_dir.glob("*.flac"))
            logger.info(f"Total de audios em audios_finais_aprovados: {len(audios_finais)}")
            
            if len(audios_finais) != reconstruction_result.get('reconstructed_count'):
                logger.warning(f"AVISO: Esperado {reconstruction_result.get('reconstructed_count')} audios finais, "
                             f"mas encontrado {len(audios_finais)}")
            
            logger.info(f"\n{'='*60}")
            logger.info(f"FILTER AND DENOISE COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total segments processed: {len(normalized_pairs)}")
            logger.info(f"\nAPPROVED: {approved_total}")
            logger.info(f"  - With denoising: {approved_with_denoise} (successfully denoised: {denoised_success})")
            logger.info(f"  - Without denoising (MOS > {mos_range[1]}): {approved_without_denoise}")
            logger.info(f"\nREJECTED: {rejected_total}")
            logger.info(f"  - By similarity: {rejected_by_similarity}")
            logger.info(f"  - By MOS: {rejected_by_mos}")
            logger.info(f"\nFinal JSON saved: {final_json_path}")
            logger.info(f"All approved audio files in: {audios_finais_dir}")
            logger.info(f"{'='*60}\n")
            
            return {
                'success': True,
                'video_id': video_id,
                'total_segments': len(normalized_pairs),
                'approved_count': approved_total,
                'approved_with_denoise': approved_with_denoise,
                'approved_without_denoise': approved_without_denoise,
                'rejected_count': rejected_total,
                'rejected_by_similarity': rejected_by_similarity,
                'rejected_by_mos': rejected_by_mos,
                'denoised_success': denoised_success,
                'final_json_path': str(final_json_path),
                'denoiser_dir': str(denoiser_dir),
                'reconstruction_result': reconstruction_result,
                'audios_finais_dir': str(audios_finais_dir)
            }
            
        except Exception as e:
            logger.error(f"Error in filter_and_denoise_segments: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}

    def sox_normalize_approved_audios(self, 
                                      audios_finais_dir: Path,
                                      video_id: str) -> Dict[str, Any]:
        """
        Step 10: Normalize approved audio files with Sox.
        Processa audios reconstruidos de alta qualidade de audios_finais_aprovados/.
        Saves to: /katube-novo/dataset/audio_dataset/{video_id}/
        
        Args:
            audios_finais_dir: Directory with reconstructed approved audio files (audios_finais_aprovados/)
            video_id: Video ID for folder name
            
        Returns:
            Dictionary with normalization results
        """
        logger.info(f"=== STEP 10: SOX NORMALIZATION ===")
        logger.info(f"Input directory: {audios_finais_dir}")
        
        try:
            # Caminho base do dataset
            project_root = Path(__file__).parent.parent
            dataset_base = project_root / 'dataset' / 'audio_dataset'
            output_dir = dataset_base / video_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Dataset base: {dataset_base}")
            logger.info(f"Output directory: {output_dir}")
            
            # Buscar todos os audios aprovados (exceto JSON)
            audio_files = [f for f in audios_finais_dir.glob("*.flac")]
            
            if not audio_files:
                logger.warning("No audio files found in audios_finais_aprovados directory")
                return {"success": False, "error": "No audio files to normalize"}
            
            logger.info(f"Found {len(audio_files)} audio files to normalize")
            
            # Contadores
            success_count = 0
            failed_count = 0
            normalized_files = []
            
            # Normalizar cada audio
            for i, audio_file in enumerate(audio_files, 1):
                logger.info(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
                
                # Output mantÃ©m nome original
                output_path = output_dir / audio_file.name
                
                # Normalizar com Sox
                result = self.sox_normalizer.normalize_audio(audio_file, output_path)
                
                if result['success']:
                    success_count += 1
                    normalized_files.append(str(output_path))
                    logger.info(f"  SUCCESS -> {output_path.name}")
                else:
                    failed_count += 1
                    logger.error(f"  FAILED: {result.get('error')}")
            
            # Log final
            logger.info(f"\n{'='*60}")
            logger.info(f"SOX NORMALIZATION COMPLETED")
            logger.info(f"{'='*60}")
            logger.info(f"Total files: {len(audio_files)}")
            logger.info(f"  Success: {success_count}")
            logger.info(f"  Failed: {failed_count}")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"{'='*60}\n")
            
            return {
                'success': True,
                'video_id': video_id,
                'total_files': len(audio_files),
                'success_count': success_count,
                'failed_count': failed_count,
                'output_dir': str(output_dir),
                'normalized_files': normalized_files
            }
            
        except Exception as e:
            logger.error(f"Error in Sox normalization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    def create_or_update_dataset_csv(self, 
                                     final_json_path: Path,
                                     video_id: str,
                                     audio_dataset_base: Path) -> Dict[str, Any]:
        """
        Cria ou atualiza o dataset.csv com informacoes dos segmentos aprovados.
        Garante sincronizacao 100% entre CSV e audios armazenados.
        
        Args:
            final_json_path: Caminho para o JSON final ({video_id}_final_audio_dataset.json)
            video_id: ID do video do YouTube
            audio_dataset_base: Caminho base para audio_dataset/ (ex: /path/to/katube-novo/dataset)
            
        Returns:
            Dictionary com resultados da operacao
        """
        logger.info("=== STEP 11: CREATING/UPDATING DATASET.CSV ===")
        
        try:

            # Validar JSON final
            if not final_json_path.exists():
                error_msg = f"JSON final nao encontrado: {final_json_path}"
                logger.error(f"ERRO CRITICO: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Carregar JSON final
            with open(final_json_path, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
            
            segments = final_data.get('segments', {})
            
            # Filtrar apenas segmentos aprovados
            approved_segments = {
                seg_id: seg_data 
                for seg_id, seg_data in segments.items() 
                if seg_data.get('status') == 'approved'
            }
            
            if not approved_segments:
                logger.warning("Nenhum segmento aprovado encontrado no JSON")
                return {
                    'success': True,
                    'approved_count': 0,
                    'message': 'Nenhum segmento aprovado para adicionar ao CSV'
                }
            
            logger.info(f"Segmentos aprovados encontrados: {len(approved_segments)}")
            
            # Definir caminhos
            csv_path = audio_dataset_base / 'dataset.csv'
            audio_dir = audio_dataset_base / 'audio_dataset' / video_id
            
            # Validar que pasta de audios existe
            if not audio_dir.exists():
                error_msg = f"Pasta de audios nao encontrada: {audio_dir}"
                logger.error(f"ERRO CRITICO: {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Preparar linhas do CSV
            csv_rows = []
            
            for segment_id, seg_data in approved_segments.items():
                # Nome do arquivo (sem extensao)
                arquivo_nome = segment_id
                
                # Caminho do audio (deve existir fisicamente)
                audio_file = audio_dir / f"{segment_id}.flac"
                
                # VALIDACAO CRITICA: Audio deve existir
                if not audio_file.exists():
                    error_msg = f"ERRO CRITICO: Audio aprovado nao encontrado: {audio_file}"
                    logger.error(error_msg)
                    raise FileNotFoundError(error_msg)
                
                # Caminho relativo (a partir de dataset/)
                caminho_relativo = f"audio_dataset/{video_id}/{segment_id}.flac"
                
                # Extrair dados do JSON
                duration = seg_data.get('duration')
                absolute_start = seg_data.get('absolute_start')
                absolute_end = seg_data.get('absolute_end')
                mos_score = seg_data.get('mos_score')
                whisper_text = seg_data.get('whisper_original', '')
                wav2vec2_text = seg_data.get('wav2vec2_original', '')
                levenshtein_sim = seg_data.get('levenshtein_similarity')
                utilizou_denoiser = seg_data.get('utilizou_denoiser')
                
                # Criar linha do CSV
                csv_row = {
                    'id': video_id,
                    'arquivo_nome': arquivo_nome,
                    'caminho': caminho_relativo,
                    'tamanho_segmento': duration,
                    'start_split': absolute_start,
                    'end_split': absolute_end,
                    'mos_score': mos_score,
                    'whisper': whisper_text,
                    'wav2vec2': wav2vec2_text,
                    'levenshtein_similarity': levenshtein_sim,
                    'utilizou_denoiser': utilizou_denoiser
                }
                
                csv_rows.append(csv_row)
                logger.info(f"Validado: {segment_id} - Audio existe em {audio_file}")
            
            # Verificar se CSV ja existe
            csv_exists = csv_path.exists()
            
            # Escrever no CSV (criar ou append)
            with open(csv_path, 'a' if csv_exists else 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'id', 'arquivo_nome', 'caminho', 'tamanho_segmento',
                    'start_split', 'end_split', 'mos_score', 'whisper',
                    'wav2vec2', 'levenshtein_similarity', 'utilizou_denoiser'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='|')

                # Escrever header apenas se arquivo for novo
                if not csv_exists:
                    writer.writeheader()
                    logger.info(f"CSV criado: {csv_path}")
                else:
                    logger.info(f"CSV existente, adicionando dados: {csv_path}")
                
                # Escrever linhas
                writer.writerows(csv_rows)
            
            logger.info(f"Dataset CSV atualizado com sucesso!")
            logger.info(f"  Total de segmentos adicionados: {len(csv_rows)}")
            logger.info(f"  Arquivo CSV: {csv_path}")
            
            return {
                'success': True,
                'csv_path': str(csv_path),
                'approved_count': len(csv_rows),
                'video_id': video_id
            }
            
        except FileNotFoundError as e:
            logger.error(f"ERRO CRITICO: {e}")
            raise
        except Exception as e:
            logger.error(f"Erro ao criar/atualizar dataset CSV: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise


    def _extract_base_name_for_validation(self, filename: str) -> str:
        """
        Extract base name from validation filename for matching with actual audio files.
        
        Examples:
        - segment_000_stt_001 -> segment_000
        - chunk_00_stt_007 -> chunk_00
        
        Args:
            filename: Validation filename with _stt_XXX suffix
            
        Returns:
            Base name without _stt_XXX suffix
        """
        # Remove _stt_XXX pattern from the end
        import re
        base_name = re.sub(r'_stt_\d+$', '', filename)
        logger.debug(f"ðŸ” Extracted base name: '{filename}' -> '{base_name}'")
        return base_name
    
    def _prepare_for_json(self, obj):
            """Recursively convert Path objects and Annotation objects to strings for JSON serialization."""
            from pyannote.core import Annotation
            import pandas as pd
            
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, Annotation):
                # Convert Annotation to string representation or skip it
                return str(obj)
            elif isinstance(obj, pd.DataFrame):
                # Converte DataFrame para lista de dicionarios
                return obj.to_dict(orient='records')
            elif isinstance(obj, dict):
                return {k: self._prepare_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._prepare_for_json(item) for item in obj]
            else:
                return obj
    
    def create_final_dataset(self, denoised_audio_paths: List[Path], stt_results_dir: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Cria o dataset final com normalizaÃ§Ã£o Sox e transcriÃ§Ãµes organizadas.
        
        Args:
            denoised_audio_paths: Lista de caminhos dos Ã¡udios denoised
            stt_results_dir: DiretÃ³rio com resultados STT
            output_dir: DiretÃ³rio de saÃ­da para o dataset final
            
        Returns:
            DicionÃ¡rio com resultados da criaÃ§Ã£o do dataset
        """
        logger.info("ðŸŽ¯ Criando dataset final com normalizaÃ§Ã£o Sox...")
        
        # Criar diretÃ³rios para o dataset final
        final_audio_dir = output_dir / "audios_final"
        final_transcriptions_dir = output_dir / "transcricoes_final"
        final_audio_dir.mkdir(parents=True, exist_ok=True)
        final_transcriptions_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'successful_normalizations': [],
            'failed_normalizations': [],
            'transcription_pairs': [],
            'total_processed': len(denoised_audio_paths),
            'success_count': 0,
            'failure_count': 0
        }
        
        logger.info(f"ðŸ“ DiretÃ³rios criados:")
        logger.info(f"   - Ãudios finais: {final_audio_dir}")
        logger.info(f"   - TranscriÃ§Ãµes finais: {final_transcriptions_dir}")
        
        # Processar cada Ã¡udio denoised
        for i, denoised_path in enumerate(denoised_audio_paths):
            try:
                logger.info(f"ðŸ”„ Processando {i+1}/{len(denoised_audio_paths)}: {denoised_path.name}")
                
                # Extrair nome base para nomenclatura final
                from .naming_utils import extract_base_name, generate_standard_name
                base_name = extract_base_name(denoised_path)
                
                # Remove "_denoised" suffix para nomenclatura limpa
                if base_name.endswith("_denoised"):
                    base_name = base_name[:-9]  # Remove "_denoised"
                
                # SEMPRE EXECUTAR SOX - Normalizar Ã¡udio (24kHz â†’ 48kHz)
                final_audio_name = generate_standard_name(base_name, "final", i+1)
                final_audio_path = final_audio_dir / f"{final_audio_name}.flac"
                
                logger.info(f"ðŸŽµ EXECUTANDO SOX: {denoised_path.name} â†’ {final_audio_path.name}")
                print(f"ðŸŽµ SOX NORMALIZATION: {denoised_path} â†’ {final_audio_path}")
                
                normalization_result = self.sox_normalizer.normalize_audio(
                    input_path=denoised_path,
                    output_path=final_audio_path
                )
                
                if normalization_result['success']:
                    results['successful_normalizations'].append(normalization_result)
                    results['success_count'] += 1
                    logger.info(f"âœ… SOX CONCLUÃDO: {final_audio_path.name}")
                    print(f"âœ… SOX SUCCESS: {final_audio_path}")
                    
                    # BUSCAR E COPIAR TRANSCRIÃ‡Ã•ES STT (Whisper + WAV2VEC2)
                    logger.info(f"ðŸ“ Buscando transcriÃ§Ãµes STT para: {base_name}")
                    transcription_files = self._find_transcription_files(base_name, stt_results_dir)
                    
                    if transcription_files:
                        # Copiar transcriÃ§Ãµes para pasta final
                        final_transcriptions = self._copy_transcriptions_to_final(
                            transcription_files, 
                            final_transcriptions_dir, 
                            final_audio_name
                        )
                        
                        results['transcription_pairs'].append({
                            'audio_file': str(final_audio_path),
                            'transcriptions': final_transcriptions,
                            'base_name': base_name
                        })
                        
                        logger.info(f"âœ… {final_audio_path.name} + {len(final_transcriptions)} transcriÃ§Ãµes copiadas")
                        print(f"ðŸ“ TRANSCRIPTIONS COPIED: {len(final_transcriptions)} files for {final_audio_path.name}")
                    else:
                        logger.warning(f"âš ï¸ Nenhuma transcriÃ§Ã£o encontrada para {base_name}")
                        print(f"âš ï¸ NO TRANSCRIPTIONS FOUND for {base_name}")
                        
                else:
                    results['failed_normalizations'].append({
                        'input_path': str(denoised_path),
                        'error': normalization_result['error']
                    })
                    results['failure_count'] += 1
                    logger.error(f"âŒ SOX FALHOU: {normalization_result['error']}")
                    print(f"âŒ SOX FAILED: {normalization_result['error']}")
                    
            except Exception as e:
                error_msg = f"Erro no processamento de {denoised_path.name}: {str(e)}"
                results['failed_normalizations'].append({
                    'input_path': str(denoised_path),
                    'error': error_msg
                })
                results['failure_count'] += 1
                logger.error(f"âŒ {error_msg}")
        
        # EstatÃ­sticas finais
        logger.info(f"ðŸŽ¯ Dataset final criado:")
        logger.info(f"   âœ… Sucessos: {results['success_count']}")
        logger.info(f"   âŒ Falhas: {results['failure_count']}")
        logger.info(f"   ðŸ“ Pares Ã¡udio-transcriÃ§Ã£o: {len(results['transcription_pairs'])}")
        logger.info(f"   ðŸ“ LocalizaÃ§Ã£o: {output_dir}")
        
        return results
    
    def _find_transcription_files(self, base_name: str, stt_results_dir: Path) -> List[Path]:
        """
        Busca arquivos de transcriÃ§Ã£o correspondentes a um Ã¡udio.
        
        Args:
            base_name: Nome base do arquivo de Ã¡udio (ex: segment_000_stt_001)
            stt_results_dir: DiretÃ³rio com resultados STT
            
        Returns:
            Lista de caminhos dos arquivos de transcriÃ§Ã£o encontrados
        """
        transcription_files = []
        
        # Buscar em subdiretÃ³rios de STT (whisper e wav2vec2) - caminhos corretos
        stt_directories = [
            stt_results_dir / "STT-whisper",
            stt_results_dir / "STT-wav2vec2"
        ]
        
        logger.info(f"ðŸ” Buscando transcriÃ§Ãµes para base_name: {base_name}")
        
        for stt_dir in stt_directories:
            logger.info(f"ðŸ“ Verificando diretÃ³rio: {stt_dir}")
            
            if stt_dir.exists():
                # Listar todos os arquivos .txt no diretÃ³rio
                txt_files = list(stt_dir.glob("*.txt"))
                logger.info(f"   Encontrados {len(txt_files)} arquivos .txt")
                
                for txt_file in txt_files:
                    logger.debug(f"   Verificando arquivo: {txt_file.name}")
                    
                    # Verificar se o nome base estÃ¡ no nome do arquivo
                    if base_name in txt_file.stem or txt_file.stem.startswith(base_name):
                        transcription_files.append(txt_file)
                        logger.info(f"   âœ… MATCH: {txt_file.name}")
                    else:
                        logger.debug(f"   âŒ No match: {txt_file.stem} != {base_name}")
            else:
                logger.warning(f"   âŒ DiretÃ³rio nÃ£o existe: {stt_dir}")
        
        logger.info(f"ðŸ“ Total de transcriÃ§Ãµes encontradas: {len(transcription_files)}")
        for tf in transcription_files:
            logger.info(f"   - {tf}")
        
        return transcription_files
    
    def _copy_transcriptions_to_final(self, transcription_files: List[Path], final_transcriptions_dir: Path, final_audio_name: str) -> List[Dict[str, str]]:
        """
        Copia arquivos de transcriÃ§Ã£o para o diretÃ³rio final com nomenclatura padronizada.
        
        Args:
            transcription_files: Lista de arquivos de transcriÃ§Ã£o
            final_transcriptions_dir: DiretÃ³rio final para transcriÃ§Ãµes
            final_audio_name: Nome do Ã¡udio final (sem extensÃ£o)
            
        Returns:
            Lista de dicionÃ¡rios com informaÃ§Ãµes das transcriÃ§Ãµes copiadas
        """
        final_transcriptions = []
        
        logger.info(f"ðŸ“„ Copiando {len(transcription_files)} transcriÃ§Ãµes para: {final_transcriptions_dir}")
        
        for i, transcription_file in enumerate(transcription_files):
            try:
                # Determinar tipo de STT pelo nome do arquivo ou diretÃ³rio pai
                if 'whisper' in transcription_file.parent.name.lower():
                    stt_type = 'whisper'
                elif 'wav2vec2' in transcription_file.parent.name.lower():
                    stt_type = 'wav2vec2'
                else:
                    stt_type = 'unknown'
                
                # Nome padronizado para transcriÃ§Ã£o final
                final_transcription_name = f"{final_audio_name}_{stt_type}.txt"
                final_transcription_path = final_transcriptions_dir / final_transcription_name
                
                logger.info(f"ðŸ“„ Copiando {stt_type}: {transcription_file.name} â†’ {final_transcription_name}")
                print(f"ðŸ“„ COPYING TRANSCRIPTION: {transcription_file} â†’ {final_transcription_path}")
                
                # Copiar arquivo
                import shutil
                shutil.copy2(transcription_file, final_transcription_path)
                
                final_transcriptions.append({
                    'type': stt_type,
                    'original_path': str(transcription_file),
                    'final_path': str(final_transcription_path),
                    'filename': final_transcription_name
                })
                
                logger.info(f"âœ… TranscriÃ§Ã£o {stt_type} copiada: {final_transcription_name}")
                
            except Exception as e:
                logger.error(f"âŒ Erro ao copiar transcriÃ§Ã£o {transcription_file}: {e}")
                print(f"âŒ TRANSCRIPTION COPY FAILED: {transcription_file} - {e}")
        
        logger.info(f"ðŸ“„ Total de transcriÃ§Ãµes copiadas: {len(final_transcriptions)}")
        
        return final_transcriptions


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    pipeline = AudioProcessingPipeline()
    
    # Example: process a audio file
    results = pipeline.process_local_audio(
         r"C:\Igor\BIA\Alcateia\Katube_2025_new\katube-novo\BZ-QBv4Vc5k_chunk_00.flac"
    )
    print(f"Pipeline results: {results['statistics']}")