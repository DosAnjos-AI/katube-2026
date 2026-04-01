"""
MOS-Bench/SHEET Audio Quality Filter
Filters audio segments based on speech quality assessment (MOS scores 1-5)
"""
import torch
import torchaudio
import numpy as np
import logging
import shutil
import json  
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings
logger = logging.getLogger(__name__)

class MOSQualityFilter:
    """
    Audio quality filter using MOS-Bench/SHEET for speech quality assessment.
    Filters out low-quality audio segments (MOS < 2.0).
    """
    
    def __init__(self, 
                 mos_threshold: float = 2.5,
                 sample_rate: int = 24000,
                 use_cuda: bool = False):
        """
        Initialize MOS quality filter.
        
        Args:
            mos_threshold: Minimum MOS score to accept (default: 2.0)
            sample_rate: Target sample rate for audio processing
            use_cuda: Whether to use GPU acceleration
        """
        self.mos_threshold = mos_threshold
        self.sample_rate = sample_rate
        self.use_cuda = use_cuda
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained MOS predictor from SHEET
        self.predictor = None
        self._load_mos_predictor()
        
    def _load_mos_predictor(self):
        """Load pre-trained MOS predictor from SHEET - OBRIGATÓRIO."""
        try:
            logger.info("🔄 Loading MOS predictor from SHEET (OBRIGATÓRIO)...")
            
            # PyTorch é obrigatório para o filtro MOS
            if not self._check_torch_availability():
                raise RuntimeError("❌ PyTorch é OBRIGATÓRIO para o filtro MOS. Instale: pip install torch torchaudio")
            
            # Load pre-trained model using torch.hub
            self.predictor = torch.hub.load(
                "unilight/sheet:v0.1.0", 
                "default", 
                trust_repo=True, 
                force_reload=False
            )
            
            if self.use_cuda and torch.cuda.is_available():
                self.predictor.model.cuda()
                logger.info("✅ MOS predictor loaded on GPU")
            else:
                logger.info("✅ MOS predictor loaded on CPU")
                
        except Exception as e:
            logger.error(f"❌ ERRO CRÍTICO: Não foi possível carregar o filtro MOS: {e}")
            raise RuntimeError(f"Filtro MOS é OBRIGATÓRIO e falhou ao carregar: {e}")
    
    def _check_torch_availability(self) -> bool:
        """Check if PyTorch is available and working."""
        try:
            import torch
            import torchaudio
            # Test basic functionality
            test_tensor = torch.randn(1000)
            return True
        except ImportError:
            logger.warning("⚠️ PyTorch/torchaudio not installed")
            return False
        except Exception as e:
            logger.warning(f"⚠️ PyTorch error: {e}")
            return False
    
    def predict_mos_score(self, audio_path: Path) -> float:
        try:
            if self.predictor is None:
                raise RuntimeError("❌ Filtro MOS não foi inicializado corretamente")
            
            if not audio_path.exists():
                logger.error(f"❌ Arquivo não encontrado: {audio_path}")
                return 1.0
            
            # 🔥 FIX: Carregar áudio, converter para mono, squeeze e mover para GPU
            import torchaudio
            waveform, sr = torchaudio.load(str(audio_path))
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Remove channel dimension and move to GPU
            waveform = waveform.squeeze(0).to(self.device)
            
            # SEMPRE usar SHEET predictor (obrigatório) - formato correto: 1D tensor na GPU
            return self.predictor.predict(wav=waveform)
                
        except Exception as e:
            logger.error(f"❌ ERRO ao predizer MOS para {audio_path.name}: {e}")
            logger.error("🔧 Verifique se PyTorch está instalado: pip install torch torchaudio")
            return 1.0
    
    def _simple_quality_assessment(self, audio_path: Path) -> float:
        """
        Simple quality assessment based on audio characteristics.
        Fallback when SHEET predictor is not available.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Estimated MOS score (1.0-5.0)
        """
        try:
            # Try to load audio with different methods
            audio, sr = self._load_audio_flexible(audio_path)
            
            if audio is None:
                logger.warning(f"⚠️ Could not load audio: {audio_path.name}")
                return 1.0
            
            # Resample if needed
            if sr != self.sample_rate:
                audio = self._resample_audio(audio, sr, self.sample_rate)
            
            # Convert to mono
            if len(audio.shape) > 1 and audio.shape[0] > 1:
                audio = np.mean(audio, axis=0)
            
            # Calculate quality metrics
            snr_estimate = self._estimate_snr(audio)
            dynamic_range = self._calculate_dynamic_range(audio)
            spectral_centroid = self._calculate_spectral_centroid(audio)
            silence_ratio = self._calculate_silence_ratio(audio)
            
            # Simple scoring based on metrics
            score = 1.0
            
            # SNR contribution (0-2 points)
            if snr_estimate > 20:
                score += 2.0
            elif snr_estimate > 15:
                score += 1.5
            elif snr_estimate > 10:
                score += 1.0
            elif snr_estimate > 5:
                score += 0.5
            
            # Dynamic range contribution (0-1 point)
            if dynamic_range > 40:
                score += 1.0
            elif dynamic_range > 25:
                score += 0.5
            
            # Spectral centroid contribution (0-1 point)
            if 1000 < spectral_centroid < 3000:  # Good speech range
                score += 1.0
            elif 500 < spectral_centroid < 4000:
                score += 0.5
            
            # Silence ratio penalty (0-1 point)
            if silence_ratio > 0.5:  # Too much silence
                score -= 0.5
            elif silence_ratio > 0.3:
                score -= 0.2
            
            return max(min(score, 5.0), 1.0)  # Clamp between 1.0 and 5.0
            
        except Exception as e:
            logger.warning(f"⚠️ Error in simple quality assessment: {e}")
            return 1.0
    
    def _load_audio_flexible(self, audio_path: Path) -> Tuple[Optional[np.ndarray], int]:
        """Load audio using multiple methods as fallback."""
        # Check if file exists first
        if not audio_path.exists():
            logger.error(f"❌ Arquivo não encontrado: {audio_path}")
            return None, 0
        
        # Check if file is empty or too small
        try:
            file_size = audio_path.stat().st_size
            if file_size < 1024:  # Less than 1KB
                logger.error(f"❌ Arquivo muito pequeno ou vazio: {audio_path} ({file_size} bytes)")
                return None, 0
        except Exception as e:
            logger.error(f"❌ Erro ao verificar tamanho do arquivo {audio_path}: {e}")
            return None, 0
            
        try:
            # Try torchaudio first
            if torch is not None:
                waveform, sr = torchaudio.load(str(audio_path))
                return waveform.squeeze().numpy(), sr
        except Exception as e:
            logger.debug(f"Torchaudio failed for {audio_path.name}: {e}")
        
        try:
            # Try librosa as fallback
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=None)
            return audio, sr
        except Exception as e:
            logger.debug(f"Librosa failed for {audio_path.name}: {e}")
        
        try:
            # Try soundfile as last resort
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            return audio, sr
        except Exception as e:
            logger.debug(f"Soundfile failed for {audio_path.name}: {e}")
        
        return None, 0
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            if torch is not None:
                # Use torchaudio resampler
                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                audio_tensor = torch.from_numpy(audio).float()
                resampled = resampler(audio_tensor)
                return resampled.numpy()
            else:
                return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            logger.warning(f"⚠️ Resampling failed: {e}")
            return audio
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            # Use RMS as signal estimate
            signal_power = np.mean(audio ** 2)
            
            # Estimate noise from quiet parts
            quiet_threshold = np.percentile(np.abs(audio), 10)
            quiet_samples = audio[np.abs(audio) < quiet_threshold]
            
            if len(quiet_samples) > 0:
                noise_power = np.mean(quiet_samples ** 2)
                snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                return max(snr, 0)
            else:
                return 20.0  # Assume good SNR if no quiet parts
                
        except:
            return 10.0  # Default moderate SNR
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB."""
        try:
            max_val = np.max(np.abs(audio))
            min_val = np.min(np.abs(audio[np.abs(audio) > 0]))
            dynamic_range = 20 * np.log10(max_val / (min_val + 1e-10))
            return dynamic_range
        except:
            return 20.0  # Default moderate dynamic range
    
    def _calculate_silence_ratio(self, audio: np.ndarray) -> float:
        """Calculate ratio of silence in audio."""
        try:
            # Simple silence detection based on RMS
            frame_length = 1024
            hop_length = frame_length // 4
            
            # Calculate RMS for each frame
            rms_values = []
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                rms = np.sqrt(np.mean(frame ** 2))
                rms_values.append(rms)
            
            if not rms_values:
                return 0.0
            
            rms_values = np.array(rms_values)
            
            # Define silence threshold (adaptive)
            silence_threshold = np.percentile(rms_values, 20)  # Bottom 20% as silence
            
            # Count silent frames
            silent_frames = np.sum(rms_values < silence_threshold)
            total_frames = len(rms_values)
            
            return silent_frames / total_frames if total_frames > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculating silence ratio: {e}")
            return 0.0
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid (brightness)."""
        try:
            # Simple FFT-based spectral centroid
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
            
            # Only positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Calculate centroid
            centroid = np.sum(positive_freqs * positive_fft) / np.sum(positive_fft)
            return centroid
        except:
            return 2000.0  # Default speech-like centroid
    
    def filter_audio_segments(self, segment_paths: List[Path], output_dir: Optional[Path] = None, video_id: Optional[str] = None) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Filter audio segments based on MOS quality scores into three categories.
        Salva scores em JSON para uso posterior.
        
        Args:
            segment_paths: List of audio segment paths
            output_dir: Base directory to save categorized segments (optional)
            video_id: ID do video para nomear o JSON (optional)
            
        Returns:
            Tuple of (approved_segments, intermediate_segments, rejected_segments)
        """
        approved_segments = []
        intermediate_segments = []
        rejected_segments = []
        
        # Dicionario para guardar scores MOS
        mos_scores_dict = {}
        
        # Create output directories
        if output_dir:
            approved_dir = output_dir / "audios_acima_3,0_MOS"
            intermediate_dir = output_dir / "audios_entre_2,5_e_3,0_MOS"
            rejected_dir = output_dir / "audios_abaixo_2,5_MOS"
            
            approved_dir.mkdir(parents=True, exist_ok=True)
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            rejected_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Filtering {len(segment_paths)} audio segments with 3-tier MOS classification")
        
        for i, segment_path in enumerate(segment_paths):
            try:
                # Predict MOS score
                mos_score = self.predict_mos_score(segment_path)
                
                logger.info(f"{segment_path.name}: MOS = {mos_score:.2f}")
                
                # Import naming utilities
                from naming_utils import extract_base_name, generate_standard_name
                
                # Extract base name and create standardized filename
                base_name = extract_base_name(segment_path)
                
                # Guardar score MOS com nome original do segmento
                segment_original_name = segment_path.stem
                mos_scores_dict[segment_original_name] = round(mos_score, 2)
                
                if mos_score >= 3.0:
                    # Aprovados - acima de 3,0
                    approved_segments.append(segment_path)
                    logger.info(f"Approved: {segment_path.name} (MOS: {mos_score:.2f})")
                    
                    if output_dir:
                        try:
                            # Create filename with MOS score: [base_name]_mos_3,5_approved_001.flac
                            mos_str = f"{mos_score:.1f}".replace('.', ',')
                            standard_name = generate_standard_name(base_name, "mos_approved", i+1)
                            approved_filename = f"{standard_name}_{mos_str}.flac"
                            approved_path = approved_dir / approved_filename
                            
                            # Copy the approved file
                            shutil.copy2(segment_path, approved_path)
                            logger.debug(f"Saved approved segment: {approved_filename}")
                        except Exception as e:
                            logger.warning(f"Could not save approved segment {segment_path.name}: {e}")
                            
                elif mos_score >= 2.5:
                    # Intermediarios - de 2,5 ate 3,0
                    intermediate_segments.append(segment_path)
                    logger.info(f"Intermediate: {segment_path.name} (MOS: {mos_score:.2f})")
                    
                    if output_dir:
                        try:
                            # Create filename with MOS score: [base_name]_mos_2,8_intermediate_001.flac
                            mos_str = f"{mos_score:.1f}".replace('.', ',')
                            standard_name = generate_standard_name(base_name, "mos_intermediate", i+1)
                            intermediate_filename = f"{standard_name}_{mos_str}.flac"
                            intermediate_path = intermediate_dir / intermediate_filename
                            
                            # Copy the intermediate file
                            shutil.copy2(segment_path, intermediate_path)
                            logger.debug(f"Saved intermediate segment: {intermediate_filename}")
                        except Exception as e:
                            logger.warning(f"Could not save intermediate segment {segment_path.name}: {e}")
                else:
                    # Ruins - abaixo de 2,5
                    rejected_segments.append(segment_path)
                    logger.warning(f"Rejected: {segment_path.name} (MOS: {mos_score:.2f} < 2.5)")
                    
                    if output_dir:
                        try:
                            # Create filename with MOS score: [base_name]_mos_2,1_rejected_001.flac
                            mos_str = f"{mos_score:.1f}".replace('.', ',')
                            standard_name = generate_standard_name(base_name, "mos_rejected", i+1)
                            rejected_filename = f"{standard_name}_{mos_str}.flac"
                            rejected_path = rejected_dir / rejected_filename
                            
                            # Copy the rejected file
                            shutil.copy2(segment_path, rejected_path)
                            logger.debug(f"Saved rejected segment: {rejected_filename}")
                        except Exception as e:
                            logger.warning(f"Could not save rejected segment {segment_path.name}: {e}")
                
                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(segment_paths)} segments...")
                    
            except Exception as e:
                logger.error(f"Error processing {segment_path.name}: {e}")
                rejected_segments.append(segment_path)
                
                # Guardar score 0.0 para segmentos com erro
                segment_original_name = segment_path.stem
                mos_scores_dict[segment_original_name] = 0.0
                
                # Save error segment to rejected directory
                if output_dir:
                    try:
                        rejected_dir.mkdir(parents=True, exist_ok=True)
                        error_filename = f"mos_error_{segment_path.name}"
                        error_path = rejected_dir / error_filename
                        shutil.copy2(segment_path, error_path)
                    except Exception as e:
                        logger.warning(f"Could not save MOS error segment {segment_path.name}: {e}")
        
        logger.info(f"3-tier filtering complete:")
        logger.info(f"   Approved (>=3.0): {len(approved_segments)}")
        logger.info(f"   Intermediate (2.5-3.0): {len(intermediate_segments)}")
        logger.info(f"   Rejected (<2.5): {len(rejected_segments)}")
        
        # Salvar JSON com scores MOS (formato simplificado)
        if output_dir and mos_scores_dict:
            # Extrair video_id do primeiro segmento se nao foi fornecido
            if not video_id and segment_paths:
                first_segment = segment_paths[0].stem
                video_id = first_segment.split('_segment_')[0] if '_segment_' in first_segment else 'unknown'
            
            # Converter numpy.float32 para float Python
            mos_scores_python = {
                segment: float(score) for segment, score in mos_scores_dict.items()
            }
            
            # Salvar na pasta segments
            segments_dir = output_dir / "segments"
            segments_dir.mkdir(parents=True, exist_ok=True)
            
            mos_json_path = segments_dir / f"{video_id}_mos_scores.json"
            
            try:
                with open(mos_json_path, 'w', encoding='utf-8') as f:
                    json.dump(mos_scores_python, f, indent=2, ensure_ascii=False)
                logger.info(f"MOS scores saved to: {mos_json_path}")
            except Exception as e:
                logger.error(f"Failed to save MOS scores JSON: {e}")
        return approved_segments, intermediate_segments, rejected_segments
    
    def get_quality_report(self, segment_paths: List[Path]) -> dict:
        """
        Generate quality report for audio segments.
        
        Args:
            segment_paths: List of audio segment paths
            
        Returns:
            Dictionary with quality statistics
        """
        scores = []
        
        for segment_path in segment_paths:
            try:
                score = self.predict_mos_score(segment_path)
                scores.append(score)
            except:
                scores.append(1.0)
        
        if not scores:
            return {
                'total_segments': 0,
                'average_mos': 0.0,
                'min_mos': 0.0,
                'max_mos': 0.0,
                'accepted_count': 0,
                'rejected_count': 0,
                'acceptance_rate': 0.0
            }
        
        scores = np.array(scores)
        accepted_count = np.sum(scores >= self.mos_threshold)
        
        return {
            'total_segments': len(scores),
            'average_mos': float(np.mean(scores)),
            'min_mos': float(np.min(scores)),
            'max_mos': float(np.max(scores)),
            'accepted_count': int(accepted_count),
            'rejected_count': int(len(scores) - accepted_count),
            'acceptance_rate': float(accepted_count / len(scores))
        }


if __name__ == "__main__":
    # Test the MOS filter
    logging.basicConfig(level=logging.INFO)
    
    filter_instance = MOSQualityFilter()
    
    # Test with sample audio
    test_path = Path("test_audio.flac")
    if test_path.exists():
        mos_score = filter_instance.predict_mos_score(test_path)
        print(f"MOS Score: {mos_score:.2f}")
    else:
        print("No test audio found")
