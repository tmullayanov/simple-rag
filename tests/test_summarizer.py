import pytest
from unittest.mock import Mock

from simple_rag.models.summarizer import summary
from simple_rag.models.summarizer.model import prompts, UnknownPromptError

from langchain.chat_models.base import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

llm = Mock(BaseChatModel)

class MockReply:
    def __init__(self, content):
        self.content = content

def test_summarizer_raises_when_no_prompt_nor_id_is_given():
    with pytest.raises(ValueError) as e:
        summary(llm=llm, content='hello')


def test_summarizer_raises_on_unknown_prompt_id():
    with pytest.raises(UnknownPromptError) as e:
        summary(llm=llm, prompt_id='unknown', content='hello')


def test_summarizer_calls_llm_with_correct_args():
    llm.invoke.return_value = MockReply('summary')
    _id = 'default'
    summary(llm=llm, prompt_id=_id, content='hello')

    assert llm.invoke.call_count == 1
    llm.invoke.assert_called_with([
        SystemMessage(content=prompts[_id]),
        HumanMessage(content='hello')
    ])