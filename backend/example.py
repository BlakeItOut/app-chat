import uuid

from arcade_rocket_approval.main import create_rm_assistant
from langchain_openai import ChatOpenAI


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The passenger_id is used in our flight tools to
        # fetch the user's flight information
        "user_id": "3442 587242",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
        "rm_loan_id": "3442 587242",
    }
}


def run_chat_interface():
    """Run a terminal-based chat interface with the assistant."""
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "user_id": "3442 587242",
            "thread_id": thread_id,
            "rm_loan_id": "3442 587242",
        }
    }

    _printed = set()
    print(
        "Welcome to the Mortgage Assistant Chat! Type 'exit' or 'quit' to end the conversation.\n"
    )

    llm = ChatOpenAI(model="gpt-4o")
    graph = create_rm_assistant(llm)
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            events = graph.stream(
                {"messages": ("user", user_input)}, config, stream_mode="values"
            )
            print("\nAssistant: ", end="")
            for event in events:
                _print_event(event, _printed)

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    run_chat_interface()
