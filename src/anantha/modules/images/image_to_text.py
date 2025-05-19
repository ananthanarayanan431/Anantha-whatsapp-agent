
import base64
import logging
import os
from typing import Optional, Union

from anantha.core.exceptions import ImageToTextError
from anantha.settings import settings
from groq import Groq

class ImageToText:
    """Class to handle image-to-text conversion using Groq API."""


    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]
    _client: Optional[Groq] = None
    logger = logging.getLogger(__name__)

    @classmethod
    def _validate_env_vars(cls):
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
    async def analyze_image(cls, image_data: Union[str, bytes], prompt:str ="") -> str:
        """Analyze an image and return the text description.
        
        Args:
            image_data (Union[str, bytes]): The image data to analyze. Can be a base64 string or bytes.
            prompt (str): Optional prompt to guide the analysis.

        Returns:
            str: The text description of the image.
        
        Raises:
            ImageToTextError: If there is an error during image analysis.
            ValueError: If the image data is invalid.  
        """
        try:
            if isinstance(image_data, str):
                if not os.path.exists(image_data):
                    raise ValueError(f"Invalid image path: {image_data}")
                
                with open(image_data, "rb") as image_file:
                    image_bytes = image_file.read()
            
            else: 
                image_bytes = image_data

            if not image_bytes:
                raise ValueError("Image data is empty or invalid.")
            
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            if not prompt:
                prompt = "Please describe what you see in this image in detail."

            # Create a message for the Vision API
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ]

            # API call!
            response = cls.client().chat.completions.create(
                model=settings.ITT_MODEL_NAME,
                messages=messages,
                max_tokens=2000,
            )

            if not response.choices:
                raise ImageToTextError("No response from the image analysis API.")
            
            
            description = response.choices[0].message.content
            cls.logger.info(f"Generated image description: {description}")

            return description
        
        except Exception as e:
            raise ImageToTextError(f"Error during image analysis: {str(e)}") from e