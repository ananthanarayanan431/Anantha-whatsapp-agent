
from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from anantha.graph.edges import (
    should_summarize_conversation,
    select_workflow
)

from anantha.graph.nodes import (
    audio_node,
    context_injection_node,
    conversation_node,
    image_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node
)

from anantha.graph.state import AIAnanthaState

@lru_cache(maxsize=1)
def create_workflow_graph() -> StateGraph:

    builder = StateGraph(AIAnanthaState)

    builder.add_node("memory_extraction_node", memory_extraction_node)
    builder.add_node("router_node", router_node)
    builder.add_node("context_injection_node", context_injection_node)
    builder.add_node("memory_injection_node", memory_injection_node)
    builder.add_node("conversation_node", conversation_node)
    builder.add_node("image_node", image_node)
    builder.add_node("audio_node", audio_node)
    builder.add_node("summarize_conversation_node", summarize_conversation_node)

    builder.add_edge(START, "memory_extraction_node")
    builder.add_edge("memory_extraction_node", "router_node") # response_type = conversation, image or audio

    builder.add_edge("router_node", "context_injection_node") 
    builder.add_edge("context_injection_node", "memory_injection_node")

    builder.add_conditional_edges("memory_injection_node",select_workflow)

    builder.add_conditional_edges("conversation_node", should_summarize_conversation)
    builder.add_conditional_edges("image_node", should_summarize_conversation)
    builder.add_conditional_edges("audio_node", should_summarize_conversation)
    builder.add_edge("summarize_conversation_node", END)

    return builder

# graph = create_workflow_graph().compile()