import logging
from typing import Callable, Optional

from arcadepy import Arcade
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import StructuredTool
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from arcade_rocket_approval.base import CompleteOrEscalate, State
from arcade_rocket_approval.prompts import APPROVE_MORTGAGE_PROMPT
from arcade_rocket_approval.tool_utils import (
    create_tool_function,
    tool_definition_to_pydantic_model,
)

logger = logging.getLogger(__name__)


def create_mortgage_tool(
    tool_name: str,
):
    """Factory function to create mortgage application tool functions.

    Args:
        tool_name: Name of the tool

    Returns:
        A decorated tool function
    """

    client = Arcade()
    tool_def = client.tools.get(tool_name)
    args_schema = tool_definition_to_pydantic_model(tool_def)
    tool_name = tool_name.replace(".", "_")
    tool_function = create_tool_function(
        client=client,
        tool_name=tool_name,
        tool_def=tool_def,
        args_schema=args_schema,
        langgraph=True,
    )
    return StructuredTool.from_function(
        tool_function,
        name=tool_name,
        args_schema=args_schema,
        description=tool_def.description,
    )


def application_node_tools() -> dict:
    wrapped = []
    all_tools = [
        "RocketApproval.StartMortgageApplication",
        "RocketApproval.SetNewHomeDetails",
        # ... add more steps here
    ]
    for tool_name in all_tools:
        wrapped.append(create_mortgage_tool(tool_name))
    return wrapped


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        # Initialize default values for mortgage assistant fields if they're missing
        current_rm_loan_id = state.get("current_rm_loan_id", "Not set yet")
        current_step = state.get("current_step", "start_application")

        return {
            "messages": [
                ToolMessage(
                    content=f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant.",
                    tool_call_id=tool_call_id,
                )
            ],
            "dialog_state": new_dialog_state,
            "current_rm_loan_id": current_rm_loan_id,
            "current_step": current_step,
        }

    return entry_node


# This node will be shared for exiting all specialized assistants
def pop_dialog_state(state: State) -> dict:
    """Pop the dialog stack and return to the main assistant.

    This lets the full graph explicitly track the dialog flow and delegate control
    to specific sub-graphs.
    """
    messages = []
    if state["messages"][-1].tool_calls:
        # Note: Doesn't currently handle the edge case where the llm performs parallel tool calls
        messages.append(
            ToolMessage(
                content="Resuming dialog with the host assistant. Please reflect on the past conversation and assist the user as needed.",
                tool_call_id=state["messages"][-1].tool_calls[0]["id"],
            )
        )
    return {
        "dialog_state": "pop",
        "messages": messages,
    }


class ToApproveMortgage(BaseModel):
    """Transfers work to a specialized assistant to handle mortgage approvals."""

    request: str = Field(
        description="Any additional information or requests from the user regarding the mortgage."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request": "I need to approve a mortgage.",
            },
        }


def route_approve_mortgage(
    state: State,
):
    route = tools_condition(state)
    if route == END:
        return END
    tool_calls = state["messages"][-1].tool_calls
    did_cancel = any(tc["name"] == "CompleteOrEscalate" for tc in tool_calls)
    if did_cancel:
        return "leave_skill"
    return "approve_mortgage_safe_tools"


def get_mortgage_assistant(llm: BaseLanguageModel) -> Runnable:
    # Tools for the mortgage approval process
    approve_mortgage_safe_tools = application_node_tools()

    # Create a mapping of all registered property detail tools
    safe_mortgage_tools = [t for t in approve_mortgage_safe_tools]
    approve_mortgage_runnable = APPROVE_MORTGAGE_PROMPT | llm.bind_tools(
        safe_mortgage_tools + [CompleteOrEscalate]
    )
    return approve_mortgage_runnable, approve_mortgage_safe_tools
