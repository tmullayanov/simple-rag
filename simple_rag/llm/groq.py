import getpass
import os

# in case of any interactive mode
# if not os.environ.get("GROQ_API_KEY"):
#   os.environ["GROQ_API_KEY"] = getpass.getpass("Enter API key for Groq: ")

from langchain.chat_models import init_chat_model


# FIXME: This is a hack to get around the fact that the Groq API key is not present in the environment
assert os.environ["GROQ_API_KEY"] is not None, (
    "Please set GROQ_API_KEY environment variable"
)


def make_llm(model_name: str = "llama3-8b-8192"):
    return init_chat_model(model_name, model_provider="groq")


llm = make_llm()
