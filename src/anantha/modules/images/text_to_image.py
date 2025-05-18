

import base64
import logging
import os 
 
from typing import Optional, Union

from anantha.core.exceptions import TextToImageError
from anantha.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from anantha.modules.images.schema import ScenarioPrompt, EnhancedPrompt
from anantha.settings import settings

from langchain_core.prompts import PromptTemplate
from langchain_groq.chat_models import ChatGroq

from together import Together

class TextToImage:
    """Class to handle text-to-image conversion using Groq API."""
    
    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TOGETHER_API_KEY"]
    _together_client: Optional[Together] = None
    logger = logging.getLogger(__name__)

    @classmethod
    def _validate_env_vars(cls):
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
    @classmethod
    def together_client(cls) -> Together:
        """Get or create Together client instance using singleton pattern."""
        if cls._together_client is None:
            cls._validate_env_vars()
            cls._together_client = Together(api_key=settings.TOGETHER_API_KEY)

        return cls._together_client
    

    @classmethod
    async def generate_image(cls, prompt: str, output_path: str) -> str:
        """Generate an image from a text prompt and save it to a file.
        
        Args:
            prompt (str): The text prompt to generate the image.
            output_path (str): The path where the generated image will be saved.
        
        Returns:
            str: The path to the saved image file.
        
        Raises:
            TextToImageError: If there is an error during image generation.
        """
        
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        
        try:
            cls.logger.info(f"Generating image for prompt: {prompt}")

            response = cls.together_client().images.generate(
                prompt=prompt,
                model=settings.ITT_MODEL_NAME,
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json",
            )

            image_data = base64.b64decode(response.data[0].b64_json)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(image_data)
                cls.logger.info(f"Image saved to {output_path}")

            return output_path

        except Exception as e:
            raise TextToImageError(f"Error generating image: {e}") from e
        

    @classmethod
    async def create_scenario(cls, chat_history: list = None) -> ScenarioPrompt:
        """Creates a first-person narrative scenario and corresponding image prompt based on chat history.
        Args:
            chat_history (list): The recent conversation context.

        Returns:
            ScenarioPrompt: A structured object containing the narrative and image prompt.
        """
        try:
            formatted_history = "\n".join([f"{msg.type.title()}: {msg.content}" for msg in chat_history[-5:]])
            cls.logger.info(f"Creating scenario with chat history: {formatted_history}")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.6,
                max_retries=2,
            )

            structured_llm = llm.with_structured_output(ScenarioPrompt)
            chain = (
                PromptTemplate(
                    input_variables=["chat_history"],
                    template=IMAGE_SCENARIO_PROMPT,
                )
                | structured_llm
            )
            scenario = chain.invoke({"chat_history": formatted_history})
            cls.logger.info(f"Generated scenario: {scenario}")

            return scenario
        except Exception as e:
            raise TextToImageError(f"Error creating scenario: {e}") from e
        
        
    @classmethod
    async def enhance_prompt(cls, prompt: str) -> EnhancedPrompt:
        """Enhances the given prompt using best practices in prompt engineering."""

        try:
            cls.logger.info(f"Enhancing prompt: {prompt}")

            llm = ChatGroq(
                model=settings.TEXT_MODEL_NAME,
                api_key=settings.GROQ_API_KEY,
                temperature=0.25,
                max_retries=2
            )

            structured_llm = llm.with_structured_output(EnhancedPrompt)

            chain = (
                PromptTemplate(
                    input_variables=["prompt"],
                    template=IMAGE_ENHANCEMENT_PROMPT,
                )
                | structured_llm
            )

            enhanced_prompt = chain.invoke({"prompt": prompt}).content 
            cls.logger.info(f"Enhanced prompt: {enhanced_prompt}")

            return enhanced_prompt
        
        except Exception as e:
            raise TextToImageError(f"Error enhancing prompt: {e}") from e