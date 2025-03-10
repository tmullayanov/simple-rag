from json import tool
from simple_rag.models.qna_rag.memory import get_checkpointer
from simple_rag.models.qna_rag.state import RagState
from simple_rag.models.qna_rag.store import SimpleVectorStore
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_core.prompts import PromptTemplate
from langchain.chat_models.base import BaseChatModel
from langchain.docstore.document import Document

from langgraph.checkpoint.memory import MemorySaver

from .prompts import rag_prompt


class RagEngineDynamicPrompt:
    """
    RAG engine for processing queries based on knowledge base data.

    For semantic search, vectorization is performed.
    Only questions are vectorized.
    Answers to contextually relevant questions are obtained separately from the vector_store object (SimpleVectorStore class).

    The class also allows you to change the prompt during operation.

    Usage example:

    vector_store = SimpleVectorStore(...)
    llm = make_llm(...) # any ChatModel goes.

    engine = RagEngineDynamicPrompt(llm, vector_store)
    rag = engine.build_rag()
    response = rag.invoke({"raw_input": "question goes here"})

    engine.change_prompt("new prompt goes here")
    """

    llm: BaseChatModel
    vector_store: SimpleVectorStore
    checkpointer: MemorySaver

    prompt_template: PromptTemplate = rag_prompt

    def __init__(
        self,
        llm: BaseChatModel,
        vector_store: SimpleVectorStore,
        checkpointer=get_checkpointer(),
    ):
        """
        Class initialization.

        Args:
            llm (BaseChatModel): Language model.
            vector_store (SimpleVectorStore): Vector storage.
            checkpointer (MemorySaver): Object for saving messages.
        """
        self.llm = llm
        self.vector_store = vector_store
        self.checkpointer = checkpointer

    def manual_retrieve(self, state: RagState):
        """
        Returns questions related to input data.

        Args:
            state (RagState): System state.

        Returns:
            dict: Dictionary with the key "questions" and a list of questions.
        """
        questions = self.vector_store.similarity_search(state["raw_input"])
        return {"questions": questions}

    def get_answers(self, state: RagState):
        """
        Returns answers to questions.

        Args:
            state (RagState): System state.

        Returns:
            dict: Dictionary with the key "qna_context" and a list of answers.
        """
        answers = [
            i
            for d in state["questions"]
            for i in self.vector_store.lookup_answers(d.page_content)
        ]

        return {"qna_context": answers}

    def answer_based_on_context(self, state: RagState):
        """
        Returns an answer based on context.

        Args:
            state (RagState): System state.

        Returns:
            dict: Dictionary with the key "answer" and the answer.
        """
        context = "\n\n".join(doc for doc in state["qna_context"])
        llm = self.prompt_template | self.llm

        response = llm.invoke(
            {
                "questions": get_questions_texts(state["questions"]),
                "answers": context,
                "raw_input": state["raw_input"],
            }
        )

        return {"answer": response.content}

    def build_rag(self) -> CompiledStateGraph:
        """
        Creates a state graph.

        Returns:
            CompiledStateGraph: State graph.
        """
        graph_builder = StateGraph(RagState).add_sequence(
            [self.manual_retrieve, self.get_answers, self.answer_based_on_context]
        )
        graph_builder.add_edge(START, "manual_retrieve")
        graph = graph_builder.compile(checkpointer=self.checkpointer)

        return graph

    def update(self, new_cfg: dict[str, str]):
        """
        Updates the model configuration.

        Args:
            new_cfg (dict[str, str]): New configuration.
        """
        if "prompt" in new_cfg:
            self.change_prompt(new_cfg["prompt"])

    def change_prompt(self, new_prompt: str):
        """
        Changes the prompt.

        Args:
            new_prompt (str): New prompt.
        """
        self.prompt_template = PromptTemplate.from_template(new_prompt)


def get_questions_texts(questions: list[Document]):
    return [q.page_content for q in questions]
