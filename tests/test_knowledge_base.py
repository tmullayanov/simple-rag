from typing import Optional
from unittest.mock import MagicMock
import pytest

from simple_rag.knowledge_base.base import KnowledgeBaseModel
from simple_rag.knowledge_base.manager import KnowledgeBaseManager, NoSuchModelError

class TestKnowledgeBaseModel(KnowledgeBaseModel):
    def query(self, query: str) -> Optional[str]:
        return "test_response"



@pytest.fixture
def knowledge_base_mgr():
    llm = MagicMock()
    embeddings = MagicMock()

    yield KnowledgeBaseManager(llm=llm, embeddings=embeddings)

def test_create_kb(knowledge_base_mgr):
    pass


def test_register_model_to_mgr():
    KnowledgeBaseManager.register_model('some_key', KnowledgeBaseModel)

def test_registered_model_can_be_accessed(knowledge_base_mgr):
    KnowledgeBaseManager.register_model('knowledge_base', TestKnowledgeBaseModel)

    model = knowledge_base_mgr.get_model('knowledge_base')
    assert isinstance(model, KnowledgeBaseModel)

def test_registered_model_is_the_same_model_across_accesses(knowledge_base_mgr):
    KnowledgeBaseManager.register_model('knowledge_base', TestKnowledgeBaseModel)
    model_1 = knowledge_base_mgr.get_model('knowledge_base')
    model_2 = knowledge_base_mgr.get_model('knowledge_base')
    assert model_1 == model_2

def test_can_register_multiple_models(knowledge_base_mgr):
    KnowledgeBaseManager.register_model('model_1', TestKnowledgeBaseModel)
    KnowledgeBaseManager.register_model('model_2', TestKnowledgeBaseModel)
    model_1 = knowledge_base_mgr.get_model('model_1')
    model_2 = knowledge_base_mgr.get_model('model_2')
    assert model_1 != model_2

def test_cannot_create_unknown_model(knowledge_base_mgr):
    with pytest.raises(NoSuchModelError):
        model = knowledge_base_mgr.get_model('unknown_model')