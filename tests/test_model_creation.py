import pytest
from unittest.mock import MagicMock

from simple_rag.chats.chat import ChatModel
from simple_rag.models import ModelCreator
from simple_rag.models.stub_model import StubModel


@pytest.fixture
def model_creator():
    llm = MagicMock(),
    config = MagicMock()
    creator = ModelCreator(llm, config)
    yield creator


def test_model_creation():
    creator = ModelCreator(MagicMock(), MagicMock())


def test_create_model_by_name(model_creator):
    name = "stub_model"

    model = model_creator.build(name=name)
    assert isinstance(model, ChatModel)


def test_throws_when_model_not_found(model_creator):
    name = "does_not_exist"

    with pytest.raises(ValueError) as e_info:
        model = model_creator.build(name)

def test_models_returns_names():
    creator = ModelCreator(MagicMock(), MagicMock())

    names = creator.models()
    assert len(names) > 0
    assert 'stub_model' in names
    assert 'rag_question_vector' in names

    