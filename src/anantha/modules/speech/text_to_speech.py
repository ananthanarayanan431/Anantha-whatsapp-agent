
import os 
from typing import Optional

from anantha.core.exceptions import TextToSpeechError
from anantha.settings import settings
from elevenlabs import ElevenLabs, Voice, VoiceSettings

class TextToSpeech:
    """A class to handle text-to-speech conversion using ElevenLabs."""

    REQUIRED_ENV_VARS = ["ELEVENLABS_API_KEY", "ELEVENLABS_VOICE_ID"]
    _client: Optional[ElevenLabs] = None

    @classmethod
    def _validate_env_vars(cls) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    @classmethod
    def client(cls) -> ElevenLabs:
        """Get or create ElevenLabs client instance using singleton pattern."""
        if cls._client is None:
            cls._validate_env_vars()
            cls._client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)
        return cls._client
    
    @classmethod
    async def synthesize(cls, text: str) -> bytes:
        """Convert text to speech using ElevenLabs.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio data

        Raises:
            ValueError: If the input text is empty or too long
            TextToSpeechError: If the text-to-speech conversion fails
        """
        if not text.strip():
            raise ValueError("Input text cannot be empty")

        if len(text) > 5000:
            raise ValueError("Input text exceeds maximum length of 5000 characters")

        try:
            audio_generator = cls.client().generate(
                text=text,
                voice=Voice(
                    voice_id=settings.ELEVENLABS_VOICE_ID,
                    settings=VoiceSettings(stability=0.5, similarity_boost=0.5),
                ),
                model=settings.TTS_MODEL_NAME,
            )

            audio_bytes = b"".join(audio_generator)
            if not audio_bytes:
                raise TextToSpeechError("Generated audio is empty")

            return audio_bytes

        except Exception as e:
            raise TextToSpeechError(f"Text-to-speech conversion failed: {str(e)}") from e