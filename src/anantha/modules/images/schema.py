

from pydantic import BaseModel, Field

class ScenarioPrompt(BaseModel):
    """Class to represent a scenario prompt."""

    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")


class EnhancedPrompt(BaseModel):
    """Class to represent an enhanced prompt."""

    content: str = Field(..., description="The enhanced text prompt to generate an image")