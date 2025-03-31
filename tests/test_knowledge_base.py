import abc
from typing import Callable, Optional
import pytest


class KnowledgeBaseModel(abc.ABC):
    pass

class KnowledgeBaseMgr():

    model: Optional[KnowledgeBaseModel] = None
    models: dict[str, KnowledgeBaseModel] = {}
    builders: dict[str, Callable[[], KnowledgeBaseModel]] = {}
    
    @staticmethod
    def register_model(key: str, model: Callable[[], KnowledgeBaseModel]):
        KnowledgeBaseMgr.builders.update({key: model})

    def get_model(self, key: str):
        if key not in self.models:
            self.models[key] = self.builders[key]()
        
        return self.models[key]
        


@pytest.fixture
def knowledge_base_mgr():
    yield KnowledgeBaseMgr()

def test_create_kb(knowledge_base_mgr):
    pass


def test_register_model_to_mgr():
    KnowledgeBaseMgr.register_model('some_key', KnowledgeBaseModel)

def test_registered_model_can_be_accessed(knowledge_base_mgr):
    KnowledgeBaseMgr.register_model('knowledge_base', KnowledgeBaseModel)

    model = knowledge_base_mgr.get_model('knowledge_base')
    assert isinstance(model, KnowledgeBaseModel)

def test_registered_model_is_the_same_model_across_accesses(knowledge_base_mgr):
    KnowledgeBaseMgr.register_model('knowledge_base', KnowledgeBaseModel)
    model_1 = knowledge_base_mgr.get_model('knowledge_base')
    model_2 = knowledge_base_mgr.get_model('knowledge_base')
    assert model_1 == model_2

def test_can_register_multiple_models(knowledge_base_mgr):
    class Model1(KnowledgeBaseModel): pass
    class Model2(KnowledgeBaseModel): pass

    KnowledgeBaseMgr.register_model('model_1', Model1)
    KnowledgeBaseMgr.register_model('model_2', Model2)
    model_1 = knowledge_base_mgr.get_model('model_1')
    model_2 = knowledge_base_mgr.get_model('model_2')
    assert model_1 != model_2