from json import tool
from simple_rag.qna_rag.memory import get_checkpointer
from simple_rag.qna_rag.state import RagState
from simple_rag.qna_rag.store import SimpleVectorStore
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel


from .prompts import rag_prompt

class RagDynamicPromptEngine:

    llm: BaseChatModel
    vector_store: SimpleVectorStore

    prompt_template: PromptTemplate = rag_prompt

    def __init__(self, llm: BaseChatModel, vector_store: SimpleVectorStore):
        self.llm = llm
        self.vector_store = vector_store

    def change_prompt(self, new_prompt: str):
        self.prompt_template = PromptTemplate.from_template(new_prompt)
    
    def manual_retrieve(self, state: RagState):
        questions = self.vector_store.similarity_search(state["raw_input"])
        return {"questions": questions}
    
    def get_answers(self, state: RagState):
        answers = [
            i
            for d in state["questions"]
            for i in self.vector_store.lookup_answers(d.page_content)
        ]

        return {"qna_context": answers}
    
    def answer_based_on_context(self, state: RagState):
        context = "\n\n".join(doc for doc in state["qna_context"])
        messages = self.prompt_template.invoke(
            {
                "questions": state["questions"],
                "answers": context,
                "raw_input": state["raw_input"],
            }
        )
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def build_graph(self) -> CompiledStateGraph:
        
        graph_builder = StateGraph(RagState).add_sequence(
            [self.manual_retrieve, self.get_answers, self.answer_based_on_context]
        )
        graph_builder.add_edge(START, "manual_retrieve")
        graph = graph_builder.compile(checkpointer=get_checkpointer())

        return graph
    
    def update(self, new_cfg: dict[str, str]):
        if "prompt" in new_cfg:
            self.change_prompt(new_cfg["prompt"])