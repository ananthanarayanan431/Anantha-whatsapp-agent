

from langgraph.graph import MessagesState

class AIAnanthaState(MessagesState):
    """
    AIAnnathaState is a subclass of MessagesState that represents the state of the AI Anantha system.
    It is used to manage the messages exchanged between the user and the AI Anantha system.

    Attributes:
        1. Last_message: (AnyMessage) - The last message exchanged between the user and the AI Anantha system.
                                        Langchain Message type (HumanMessage, AIMessage, SystemMessage)
        2. summary: (str) - A summary of the conversation.
        3. workflow: (str) - The current workflow being executed. (Can be "conversation", "image", or "audio".)
        4. audio_buffer: (bytes) - The audio buffer to be used for speech-to-text conversion.
        5. image_path: (str) - The path to the image to be used for image processing.
        6. current_activity: (str) - The current activity being performed by the AI Anantha system.
        7. apply_activity: (str) - The activity to be applied to the current activity.
        8. memory_context: (str) - The context of the memory being used by the AI Anantha system. 
                                   (injected into the character)
    """

    summary: str 
    workflow: str
    audio_buffer: str
    image_path: str
    current_activity: str
    apply_activity: str
    memory_context: str

