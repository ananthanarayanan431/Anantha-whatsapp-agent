
import re 


from anantha.settings import settings
from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq


def remover_asterisk_content(text: str)->str:
    """Remove content between asterisks in the text."""

    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    """Custom output parser to remove content between asterisks."""

    def parse(self, text: str) -> str:
        """Parse the text and remove content between asterisks."""
        return remover_asterisk_content(super().parse(text))



def get_chat_model(temperature: float = 0.6) -> ChatGroq:
    """Get the chat model with the specified temperature.
    
    Args:
        temperature (float): The temperature for the chat model. Default is 0.6.
    Returns:
        ChatGroq: The chat model instance.
    """

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model_name=settings.TEXT_MODEL_NAME,
        temperature=temperature,
    )


def get_text_to_speech_module():
    pass 

def get_text_to_image_module():
    pass

def get_image_to_text_module():
    pass
