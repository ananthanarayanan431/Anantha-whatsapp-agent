
from pydantic import BaseModel, Field
from typing import Literal, List

class RouterResponse(BaseModel):
    response_type: Literal["conversation", "image", "audio"] = Field(
        description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'"
    )