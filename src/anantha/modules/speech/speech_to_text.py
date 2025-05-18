
import os
import tempfile
from typing import Optional

from anantha.core.exceptions import SpeechToTextError
from anantha.settings import settings
from groq import Groq


class SpeechToText:
    """A class to handle speech-to-text conversion using Groq's Whisper model."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]
    _client: Optional[Groq] = None

    @classmethod
    def _validate_env_vars(cls) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @classmethod
    def client(cls) -> Groq:
        """Get or create Groq client instance using singleton pattern."""
        if cls._client is None:
            cls._validate_env_vars()
            cls._client = Groq(api_key=settings.GROQ_API_KEY)
        return cls._client

    @classmethod
    async def transcribe(cls, audio_data: bytes) -> str:
        """Convert speech to text using Groq's Whisper model.

        Args:
            audio_data: Binary audio data

        Returns:
            str: Transcribed text

        Raises:
            ValueError: If the audio file is empty or invalid
            RuntimeError: If the transcription fails
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name

            try:
                with open(temp_file_path, "rb") as audio_file:
                    transcription = cls.client().audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3-turbo",
                        language="en",
                        response_format="text",
                    )

                if not transcription:
                    raise SpeechToTextError("Transcription result is empty")

                return transcription

            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            raise SpeechToTextError(f"Speech-to-text conversion failed: {str(e)}") from e
