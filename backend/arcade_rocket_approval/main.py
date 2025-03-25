from typing import Literal

from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.types import Command

from arcade_rocket_approval.assistants import (
    ToApproveMortgage,
    create_entry_node,
    get_mortgage_assistant,
    pop_dialog_state,
    route_approve_mortgage,
)
from arcade_rocket_approval.base import (
    Assistant,
    State,
    create_tool_node_with_fallback,
)
from arcade_rocket_approval.prompts import PRIMARY_ASSISTANT_PROMPT


def check_application_state(config: RunnableConfig) -> Command:
    """Check if the user has started an application.

    Returns:
        A Command to update the state with the application state.
    """
    configuration = config.get("configurable", {})
    rm_loan_id = configuration.get("rm_loan_id", None)
    if not rm_loan_id:
        raise ValueError("No rm_loan_id configured.")
    # response = get_application_state(rm_loan_id)
    return Command(update={"current_step": "start_application"})


def route_primary_assistant(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    if tool_calls:
        if tool_calls[0]["name"] == "ToApproveMortgage":
            return "enter_approve_mortgage"
        return "primary_assistant_tools"
    raise ValueError("Invalid route")


def create_rm_assistant(
    llm: BaseLanguageModel = ChatOpenAI(model="gpt-4o"),
) -> Runnable:
    primary_assistant_tools = []
    assistant_runnable = PRIMARY_ASSISTANT_PROMPT | llm.bind_tools(
        primary_assistant_tools
        + [
            ToApproveMortgage,
        ]
    )

    approve_mortgage_runnable, approve_mortgage_safe_tools = get_mortgage_assistant(llm)

    # Each delegated workflow can directly respond to the user
    # When the user responds, we want to return to the currently active workflow
    def route_to_workflow(
        state: State,
    ) -> Literal[
        "primary_assistant",
        "approve_mortgage",
    ]:
        """If we are in a delegated state, route directly to the appropriate assistant."""
        dialog_state = state.get("dialog_state")
        if not dialog_state:
            return "primary_assistant"
        return dialog_state[-1]

    builder = StateGraph(State)
    builder.add_node("leave_skill", pop_dialog_state)
    builder.add_edge("leave_skill", "primary_assistant")

    builder.add_node(
        "enter_approve_mortgage",
        create_entry_node("Mortgage Assistant", "approve_mortgage"),
    )
    builder.add_node("approve_mortgage", Assistant(approve_mortgage_runnable))
    builder.add_edge("enter_approve_mortgage", "approve_mortgage")
    builder.add_node(
        "approve_mortgage_safe_tools",
        create_tool_node_with_fallback(approve_mortgage_safe_tools),
    )

    builder.add_conditional_edges(
        "approve_mortgage",
        route_approve_mortgage,
        [
            "approve_mortgage_safe_tools",
            "leave_skill",
            END,
        ],
    )
    # Primary assistant
    builder.add_node("primary_assistant", Assistant(assistant_runnable))
    builder.add_node(
        "primary_assistant_tools",
        create_tool_node_with_fallback(primary_assistant_tools),
    )
    builder.add_edge(START, "primary_assistant")
    # The assistant can route to one of the delegated assistants,
    # directly use a tool, or directly respond to the user
    builder.add_conditional_edges(
        "primary_assistant",
        route_primary_assistant,
        [
            "enter_approve_mortgage",
            "primary_assistant_tools",
            END,
        ],
    )
    builder.add_edge("primary_assistant_tools", "primary_assistant")

    # Compile graph
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        # Let the user approve or deny the use of sensitive tools
        interrupt_before=[],
    )
    return graph


def make_graph():
    llm = ChatOpenAI(model="gpt-4o")
    return create_rm_assistant(llm)
