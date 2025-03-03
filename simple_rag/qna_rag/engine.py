from simple_rag.qna_rag.state import RagState
from simple_rag.qna_rag.store import SimpleVectorStore
from langgraph.graph import START, StateGraph
from .prompts import rag_prompt

from langchain.chat_models.base import BaseChatModel


def build_question_retriever(vector_store: SimpleVectorStore):
    def manual_retrieve(state: RagState):
        questions = vector_store.similarity_search(state["questions"])
        return {"questions": questions}

    return manual_retrieve


def build_answers_retriever(vector_store: SimpleVectorStore):
    def get_answers(state: RagState):
        answers = [
            i
            for d in state["questions"]
            for i in vector_store.lookup_answers(d.page_content)
        ]

        return {"qna_context": answers}

    return get_answers


def build_summarizer(llm: BaseChatModel):
    def answer_based_on_context(state: RagState):
        context = "\n\n".join(doc for doc in state["qna_context"])
        messages = rag_prompt.invoke(
            {
                "questions": state["questions"],
                "answers": context,
                "raw_input": state["raw_input"],
            }
        )
        response = llm.invoke(messages)
        return {"answer": response.content}

    return answer_based_on_context


def build_rag_graph(
    llm: BaseChatModel,
    vector_store: SimpleVectorStore,
):
    manual_retrieve = build_question_retriever(vector_store)
    get_answers = build_answers_retriever(vector_store)
    answer_based_on_context = build_summarizer(llm)

    graph_builder = StateGraph(RagState).add_sequence(
        [manual_retrieve, get_answers, answer_based_on_context]
    )
    graph_builder.add_edge(START, "manual_retrieve")
    graph = graph_builder.compile()

    return graph
