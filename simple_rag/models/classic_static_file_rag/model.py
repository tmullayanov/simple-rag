import uuid

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from simple_rag.chats import ChatModel

from .prompt import default_rag_prompt
from .state import RagState


class ClassicRagModel(ChatModel):

    llm: BaseChatModel
    store: VectorStore
    prompt: PromptTemplate = default_rag_prompt
    checkpointer: MemorySaver
    graph: CompiledStateGraph

    def __init__(self, llm, store: VectorStore):
        self.llm = llm
        self.store = store
        self.checkpointer = MemorySaver()
        self._build_graph()

    def _retrieve(self, state: RagState):
        retrieved_docs = self.store.similarity_search(state['question'])

        return {'context': retrieved_docs}

    def _generate(self, state: RagState):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)

        return {"answer": response.content}

    def _build_graph(self):
        graph_builder = StateGraph(RagState).add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        graph = graph_builder.compile()

        self.graph = graph

    def send(self, conversation_id: uuid.UUID, message: str) -> str:
        config = {"configurable": {"thread_id": str(id)}}
        answer = self.graph.invoke({"question": message}, config=config)

        return answer["answer"]

    def update(self, new_cfg: dict[str, str]):
        logger.debug("QnaStaticFileService::update()")

        if "prompt" in new_cfg:
            self.change_prompt(new_cfg["prompt"])
            logger.debug("Updated prompt for QnaStaticFileService")

    def change_prompt(self, new_prompt: str):
        self.prompt = PromptTemplate.from_template(new_prompt)
