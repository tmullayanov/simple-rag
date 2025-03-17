from langchain_core.prompts import PromptTemplate

default_rag_prompt_template = ''''
You are an assistant. You have the following information:
DOCUMENT:
{context}
--
QUESTION:
{question}
--
INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION, say it.
'''

default_rag_prompt = PromptTemplate.from_template(default_rag_prompt_template)