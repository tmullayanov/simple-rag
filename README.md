# Simple RAG Application

A simple RAG (Retrieval-Augmented Generation) application built with FastAPI and Groq.

Ipynb shows the general idea of RAG approach

## Requirements

- Python 3.8+
- uv package manager

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd simple-rag
```

2. Install dependencies using uv:

```bash
uv sync
```

## env

### Groq settings
GROQ_MODEL_NAME=<your-groq-model-name>

### Server settings (optional)
HOST=127.0.0.1 # default
PORT=8000 # default

### QnA file settings
QNA_FILE_PATH=<path-to-your-qna-file>
QNA_DELIMITER=; # default

### Logging settings (optional)
CONSOLE_LOG_LEVEL=info # default
FILE_LOG=/path/to/log/file # optional
FILE_LOG_LEVEL=info # default, only used if FILE_LOG is set


## Running the application

```bash
uv run --env-file=.env main.py  
```


The application will be available at `http://127.0.0.1:8000` (or the configured HOST:PORT).

### Environment Variables

- `GROQ_MODEL_NAME`: Required. The name of the Groq model to use for generation
- `HOST`: Optional. Server host address (default: 127.0.0.1)
- `PORT`: Optional. Server port (default: 8000)
- `QNA_FILE_PATH`: Required. Path to your QnA dataset file
- `QNA_DELIMITER`: Optional. Delimiter used in the QnA file (default: ;)
- `CONSOLE_LOG_LEVEL`: Optional. Logging level for console output (default: info)
- `FILE_LOG`: Optional. Path to log file. If not set, file logging is disabled
- `FILE_LOG_LEVEL`: Optional. Logging level for file output (default: info)


## Usage

0. (Optional) Check out an OpenApi schema at `http://HOST:PORT/docs` or `http://HOST:PORT/openapi.json`

1. Create a new chat

```bash
curl -X POST http://localhost:8000/chat/create \
--header 'Content-Type: application/json' \
--data '{
    "model": "rag_question_vector"
}'
```

The response will be a JSON object with the chat id.

2. Send a message to the chat

```bash
curl -X POST http://HOST:PORT/chat/message \
-/-header 'Content-Type: application/json' \
--data '{
    "chat_id": "c73e5e67-6f3a-4518-856b-8561a0ec7832",
    "message": "Чем отличается глубокое обучение от машинного?"
}'
```

The response will be a JSON object with the answer.

3. Delete a chat

Inactive chats are automatically deleted after 5 minutes of inactivity.
At this time this is not configurable.

User can delete a chat manually as well by it's id:

```bash
curl -X DELETE "http://HOST:PORT/chat/550e8400-e29b-41d4-a716-446655440000"
```

If the deletion is successful, the following response is generated:
```json
{
  "status": "success",
  "message": "Chat deleted"
}
```

If the chat with id provided is not found, `404` with `Chat not found` is returned.

4. Update model.

Every chat has it's own instance of model. It can be configurable with `/update` method.
Currently only the `prompt` attribute is supported, and each model can treat this attribute differently.
Please refer to the model documentation or the model developers for further information.

```bash
curl -X POST 'http://HOST:PORT/chat/update_model' \
-header 'Content-Type: application/json' \
--data '{
    "chat_id": "chat_id goes here",
    "prompt": "new prompt goes here"   
}'

## Models

Currently there is only one model: 'rag_question_vector'

### `rag_question_vector`

This model is used to parse static csv with questions and answers.
Then only the questions are used to create an in-memory vector database.

So the additional step is needed to retrieve the answers from the original file.

The name of the model must be provided when creating a new chat.
