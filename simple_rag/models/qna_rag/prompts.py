from langchain_core.prompts import PromptTemplate

template = """Используй приложенный контекст для ответа на вопрос от пользователя.
Контекст - это наша база знаний: близкие по смыслу вопросы и ответы из него.
Если ты не знаешь ответ или он не вычисляется из контекста, скажи, что не знаешь, не придумывай сам.
Используй максимум три приложения, и будь краток, лаконичен и точен.
Всегда добавляй "Спасибо за вопрос!" в конце.

Вопросы: {questions}
===
Контекст: 
{answers}

===
Изначальный вопрос от пользователя: {raw_input}

Ответ:"""

rag_prompt = PromptTemplate.from_template(template)
