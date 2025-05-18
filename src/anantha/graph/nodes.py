
import os 
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig

from anantha.graph.state import AIAnanthaState
from anantha.graph.utils.chains import (
    get_anantha_response_chain,
    get_router_chain,
)
from anantha.graph.utils.helpers import (
    get_chat_model,
    get_text_to_speech_module,
    get_text_to_image_module
)

from anantha.modules.memory.long_term.memory_manager import get_memory_manager
from anantha.modules.schedules.context_generation import ScheduleContextGenerator
from anantha.settings import settings

async def router_node(state: AIAnanthaState) -> AIAnanthaState:
    """Router node for the Anantha application."""

    chain = get_router_chain()
    response = await chain.ainvoke(
        {
            'messages': state['messages'][-settings.ROUTER_MESSAGES_TO_ANALYZE :]
        }
    )
    state['workflow'] = response.response_type
    return state 


def context_injection_node(state: AIAnanthaState) -> AIAnanthaState:
    """Context injection node for the Anantha application."""

    schedule_context = ScheduleContextGenerator.get_current_activity()
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False

    state['apply_activity'] = apply_activity
    state['current_activity'] = schedule_context
    return state


async def conversation_node(state: AIAnanthaState, config: RunnableConfig):
    """Conversation node for the Anantha application."""

    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context =  state.get("memory_context", "")

    chain = get_anantha_response_chain(state.get("summary", ""))

    response = await chain.ainvoke(
        {
            'messages': state['messages'],
            'current_activity': current_activity,
            'memory_context': memory_context
        },
        config,
    )
    state['messages'] = AIMessage(content=response)
    return state 


async def image_node(state: AIAnanthaState, config: RunnableConfig):
    """Image node for the Anantha application."""

    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_anantha_response_chain(state.get("summary", ""))
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state["messages"][-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    scenario_message = HumanMessage(content=f"<image attached by Anantha generated from prompt: {scenario.image_prompt}>")
    updated_messages = state["messages"] + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    state['messages'] = AIMessage(content=response)
    state['image_path'] = img_path

    return state 


async def audio_node(state: AIAnanthaState, config: RunnableConfig) -> AIAnanthaState:
    """Audio node for the Anantha application."""

    current_activity = ScheduleContextGenerator.get_current_activity()
    memory_context = state.get("memory_context", "")

    chain = get_anantha_response_chain(state.get("summary", ""))
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state["messages"],
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )

    output_audio = await text_to_speech_module.synthesize(response)
    state['messages'] = AIMessage(content=response)
    state['audio_buffer'] = output_audio

    return state

async def summarize_conversation_node(state: AIAnanthaState) -> AIAnanthaState:
    """Summarize conversation node for the Anantha application."""

    model = get_chat_model()
    summary = state.get("summary", "")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Anantha and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Anantha and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Anantha and the user:"
        )

    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    state['summary'] = response.content
    state['messages'] = delete_messages

    return state


async def memory_extraction_node(state: AIAnanthaState) -> AIAnanthaState:
    """Memory extraction node for the Anantha application."""

    if not state["messages"]:
        return {}
    
    memory_manager = get_memory_manager()
    await memory_manager.extract_and_store_memory(state["messages"][-1])
    return {}

def memory_injection_node(state: AIAnanthaState) -> AIAnanthaState:
    """Memory injection node for the Anantha application."""
    
    memory_manager = get_memory_manager()
    recent_context = " ".join([m.content for m in state["messages"][-3:]])
    memories = memory_manager.get_relevant_memories(recent_context)
    memory_context = memory_manager.format_memories_for_prompt(memories)

    state['memory_context'] = memory_context
    return state
