from langgraph.checkpoint.memory import MemorySaver

# FIXME: a placeholder for now
memory = MemorySaver()


def get_checkpointer():
    return memory
