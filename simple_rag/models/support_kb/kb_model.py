from simple_rag.knowledge_base.base import KnowledgeBaseModel


class IaaSSupportKBModel(KnowledgeBaseModel):
    def query(self, query: str) -> list[str]:
        raise NotImplementedError
