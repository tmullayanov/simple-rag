from typing import Optional
from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

prompts = {
    'default': 'Make summary of a given text in a few short sentences.'
}

class UnknownPromptError(ValueError): pass

def summary(llm: BaseChatModel, content: str, prompt_id: Optional[str] = None, custom_prompt: Optional[str] = None) -> str:
    if prompt_id is None and custom_prompt is None:
        raise ValueError("Either prompt_id or custom_prompt must be provided.")
    
    if prompt_id not in prompts:
        raise UnknownPromptError()
    
    prompt = prompts[prompt_id] if prompt_id else custom_prompt

    response = llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=content)
    ])

    return response.content
    
