
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from anantha.core.prompts import (
    CHARACTER_CARD_PROMPT,
    ROUTER_PROMPT
)

from anantha.graph.utils.helpers import AsteriskRemovalParser, get_chat_model
from anantha.graph.utils.schema import RouterResponse

def get_router_chain():
    """Get the router chain for the Anantha application."""

    model = get_chat_model(temperature=0.4).with_structured_output(RouterResponse)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', ROUTER_PROMPT), 
            MessagesPlaceholder(variable_name="messages"),
        ],
    )
    return prompt | model 

def get_anantha_response_chain(summary:str = ""):
    """Get the Anantha response chain for the Anantha application."""

    model = get_chat_model()
    system_message = CHARACTER_CARD_PROMPT

    if summary:
        system_message += f"\n\nSummary of conversation earlier between Anantha and the user: {summary}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', system_message),
            MessagesPlaceholder(variable_name="messages"),
        ],
    )
    return prompt | model | AsteriskRemovalParser()


