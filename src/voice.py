"""
Gnani Voice Pipeline & Wrappers for ZentroMart VoiceOps 2026.

Integrates:
- Gnani Speech-to-Text (STT): GnaniSTTClient with language_code='en-IN' (or 'hi-IN')
- Gnani Text-to-Speech (TTS): GnaniTTSClient with model='timbre-v2.5' and voice='Kaveri'
- Uses GNANI_API_KEY from environment variables (.env)
"""
import io
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

logger = logging.getLogger("VoiceService")

try:
    from gnani.stt import GnaniSTTClient
    from gnani.tts import GnaniTTSClient
    GNANI_AVAILABLE = True
except ImportError:
    GNANI_AVAILABLE = False
    logger.warning("gnani-vachana library not installed; running in mock fallback mode.")


class VoiceService:
    """
    Speech interface wrapper managing Gnani STT and TTS services.
    Configured for Indian English ('en-IN') STT and 'Kaveri' ('timbre-v2.5') TTS.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_language: str = "en-IN",
        default_voice: str = "Kaveri",
        default_model: str = "timbre-v2.5",
    ) -> None:
        self.api_key = api_key or os.getenv("GNANI_API_KEY", "")
        self.default_language = default_language
        self.default_voice = default_voice
        self.default_model = default_model

        self.stt_client: Optional[Any] = None
        self.tts_client: Optional[Any] = None

        if GNANI_AVAILABLE and self.api_key:
            try:
                self.stt_client = GnaniSTTClient(api_key=self.api_key)
                self.tts_client = GnaniTTSClient(api_key=self.api_key)
                logger.info("Gnani Voice clients initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gnani clients: {e}")

    # =========================================================================
    # Speech-to-Text (STT)
    # =========================================================================

    def speech_to_text(
        self,
        audio_input: Union[str, Path, bytes],
        language_code: Optional[str] = None,
    ) -> str:
        """
        Transcribe speech audio into text using Gnani STT.
        
        Args:
            audio_input: Path to audio file or raw audio bytes (WAV/MP3).
            language_code: Language code (defaults to 'en-IN').
            
        Returns:
            Transcribed text string.
        """
        lang = language_code or self.default_language

        if not self.stt_client:
            logger.warning("Gnani STT client not active; returning empty transcript.")
            return ""

        try:
            if isinstance(audio_input, (str, Path)):
                path = Path(audio_input)
                if not path.exists():
                    raise FileNotFoundError(f"Audio file not found: {path}")
                res = self.stt_client.transcribe(str(path), language_code=lang, format="transcribe")
            elif isinstance(audio_input, bytes):
                res = self.stt_client.transcribe_bytes(
                    audio_input, filename="speech.wav", language_code=lang, format="transcribe"
                )
            else:
                raise ValueError("audio_input must be a file path or bytes")

            transcript = res.get("transcript", "")
            return transcript.strip()
        except Exception as e:
            logger.error(f"Error during Gnani STT transcription: {e}")
            return ""

    # =========================================================================
    # Text-to-Speech (TTS)
    # =========================================================================

    def text_to_speech(
        self,
        text: str,
        voice: Optional[str] = None,
        model: Optional[str] = None,
        output_file: Optional[Union[str, Path]] = None,
    ) -> bytes:
        """
        Synthesize text into speech audio bytes using Gnani TTS.
        
        Args:
            text: Text to synthesize.
            voice: Voice name (defaults to 'Kaveri').
            model: TTS model (defaults to 'timbre-v2.5').
            output_file: Optional path to save the generated audio WAV.
            
        Returns:
            Synthesized audio bytes.
        """
        chosen_voice = voice or self.default_voice
        chosen_model = model or self.default_model

        if not text or not text.strip():
            return b""

        if not self.tts_client:
            logger.warning("Gnani TTS client not active; returning dummy audio bytes.")
            return b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00"

        try:
            out_path_str = str(output_file) if output_file else None
            audio_bytes = self.tts_client.synthesize(
                text=text.strip(),
                voice=chosen_voice,
                model=chosen_model,
                output_file=out_path_str,
            )

            # Ensure file is saved if requested and not written automatically
            if output_file and audio_bytes and not Path(output_file).exists():
                p = Path(output_file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(audio_bytes)

            return audio_bytes
        except Exception as e:
            logger.error(f"Error during Gnani TTS synthesis: {e}")
            return b""
