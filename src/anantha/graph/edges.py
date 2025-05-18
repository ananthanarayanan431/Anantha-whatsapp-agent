
from langgraph.graph import START, END
from typing import Literal

from anantha.graph.state import AIAnanthaState
from anantha.settings import settings

def should_summarize_conversation(
        state: AIAnanthaState
    ) -> Literal['summarize_conversation_node', '__end__']:

    """Should summarize conversation node for the Anantha application."""

    messages = state['messages']
    if len(messages) > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        return 'summarize_conversation_node'
    
    return END


def select_workflow(
        state: AIAnanthaState
    ) -> Literal["conversation_node", "image_node", "audio_node"]:

    """Select workflow node for the Anantha application."""

    workflow = state['workflow']

    if workflow == "image":
        return "image_node"

    elif workflow == "audio":
        return "audio_node"

    else:
        return "conversation_node"
