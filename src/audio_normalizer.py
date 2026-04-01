"""
Audio Normalizer using FFmpeg
Normalizes audio to FLAC format, 24kHz, Mono after download
"""
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from naming_utils import extract_base_name, generate_standard_name

logger = logging.getLogger(__name__)

class AudioNormalizer:
    """
    Audio normalizer using FFmpeg to standardize audio format.
    Converts to FLAC, 24kHz, Mono.
    """
    
    def __init__(self, 
                 target_sample_rate: int = 24000,
                 target_format: str = "flac",
                 target_channels: int = 1):
        """
        Initialize audio normalizer.
        
        Args:
            target_sample_rate: Target sample rate (default: 24000 Hz)
            target_format: Target audio format (default: flac)
            target_channels: Target number of channels (default: 1 = mono)
        """
        self.target_sample_rate = target_sample_rate
        self.target_format = target_format
        self.target_channels = target_channels
        
        # Check if FFmpeg is available
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Check if FFmpeg is available in the system."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                logger.info("✅ FFmpeg is available")
            else:
                raise RuntimeError("FFmpeg not working properly")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
            logger.error(f"❌ FFmpeg not found or not working: {e}")
            raise RuntimeError("FFmpeg is required for audio normalization. Please install FFmpeg.")
    
    def normalize_audio(self, 
                       input_path: Path, 
                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Normalize audio file to standard format.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for normalized output (if None, replaces input)
            
        Returns:
            Dictionary with normalization results
        """
        try:
            # Generate output path if not provided
            if output_path is None:
                # Use standardized naming
                base_name = extract_base_name(input_path)
                standard_name = generate_standard_name(base_name, "normalized")
                output_path = input_path.parent / f"{standard_name}.{self.target_format}"
            
            logger.info(f"🔄 Normalizing audio: {input_path.name}")
            logger.info(f"   Target: {self.target_format}, {self.target_sample_rate}Hz, {self.target_channels} channel(s)")
            
            # FFmpeg command for normalization
            cmd = [
                'ffmpeg',
                '-i', str(input_path),                    # Input file
                '-ar', str(self.target_sample_rate),     # Sample rate
                '-ac', str(self.target_channels),        # Channels (mono)
                '-acodec', 'flac',                       # Codec
                '-compression_level', '5',               # FLAC compression level
                '-y',                                    # Overwrite output
                str(output_path)                         # Output file
            ]
            
            # Execute FFmpeg command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                # Get file info
                input_size = input_path.stat().st_size
                output_size = output_path.stat().st_size
                
                logger.info(f"✅ Audio normalized successfully")
                logger.info(f"   Input: {input_size / (1024*1024):.1f} MB")
                logger.info(f"   Output: {output_size / (1024*1024):.1f} MB")
                logger.info(f"   Saved to: {output_path}")
                
                return {
                    'success': True,
                    'input_path': str(input_path),
                    'output_path': str(output_path),
                    'input_size': input_size,
                    'output_size': output_size,
                    'format': self.target_format,
                    'sample_rate': self.target_sample_rate,
                    'channels': self.target_channels
                }
            else:
                logger.error(f"❌ FFmpeg normalization failed")
                logger.error(f"   Error: {result.stderr}")
                return {
                    'success': False,
                    'error': f"FFmpeg failed: {result.stderr}",
                    'input_path': str(input_path)
                }
                
        except subprocess.TimeoutExpired:
            logger.error(f"❌ FFmpeg normalization timeout")
            return {
                'success': False,
                'error': "FFmpeg normalization timeout (5 minutes)",
                'input_path': str(input_path)
            }
        except Exception as e:
            logger.error(f"❌ Audio normalization error: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_path': str(input_path)
            }
    
    def normalize_and_replace(self, audio_path: Path) -> Dict[str, Any]:
        """
        Normalize audio file and replace the original.
        
        Args:
            audio_path: Path to audio file to normalize
            
        Returns:
            Dictionary with normalization results
        """
        try:
            # Create temporary output path
            temp_path = audio_path.with_suffix(f'.temp.{self.target_format}')
            
            # Normalize to temporary file
            result = self.normalize_audio(audio_path, temp_path)
            
            if result['success']:
                # Replace original with normalized version
                audio_path.unlink()  # Delete original
                temp_path.rename(audio_path)  # Rename temp to original
                
                logger.info(f"✅ Original file replaced with normalized version")
                
                return {
                    'success': True,
                    'path': str(audio_path),
                    'format': self.target_format,
                    'sample_rate': self.target_sample_rate,
                    'channels': self.target_channels,
                    'size': result['output_size']
                }
            else:
                # Clean up temp file if normalization failed
                if temp_path.exists():
                    temp_path.unlink()
                return result
                
        except Exception as e:
            logger.error(f"❌ Error replacing normalized file: {e}")
            return {
                'success': False,
                'error': str(e),
                'path': str(audio_path)
            }
    
    def get_audio_info(self, audio_path: Path) -> Dict[str, Any]:
        """
        Get audio file information using FFmpeg.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                info = json.loads(result.stdout)
                
                # Extract audio stream info
                audio_stream = None
                for stream in info.get('streams', []):
                    if stream.get('codec_type') == 'audio':
                        audio_stream = stream
                        break
                
                if audio_stream:
                    return {
                        'success': True,
                        'format': audio_stream.get('codec_name', 'unknown'),
                        'sample_rate': int(audio_stream.get('sample_rate', 0)),
                        'channels': int(audio_stream.get('channels', 0)),
                        'duration': float(audio_stream.get('duration', 0)),
                        'bit_rate': int(audio_stream.get('bit_rate', 0))
                    }
                else:
                    return {'success': False, 'error': 'No audio stream found'}
            else:
                return {'success': False, 'error': f'FFprobe failed: {result.stderr}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
