from typing import Callable

from .base import KnowledgeBaseModel


class KnowledgeBaseManager():
    '''
    Provides access to registered knowledge base models
    '''

    models: dict[str, KnowledgeBaseModel] = {}
    builders: dict[str, Callable[[], KnowledgeBaseModel]] = {}
    
    @staticmethod
    def register_model(key: str, model: Callable[[], KnowledgeBaseModel]):
        KnowledgeBaseManager.builders.update({key: model})

    def get_model(self, key: str):
        if key not in self.models:
            self.models[key] = self.builders[key]()
        
        return self.models[key]
        