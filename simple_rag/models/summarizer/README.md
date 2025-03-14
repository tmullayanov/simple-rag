# Summarizer model

This model might be implemented via ChatModel mechanics.
But it doesn't need to since it doesn't have conversation history or any kind of state.
And implementing ChatModel interface would result in a clunky and awkward api calls.

Because of that, more straight-forward approach is used.

