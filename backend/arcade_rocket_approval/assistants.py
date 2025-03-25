import logging
from typing import Annotated, Callable, Dict, Optional

from arcadepy import NOT_GIVEN, Arcade
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field

from arcade_rocket_approval.base import CompleteOrEscalate, State
from arcade_rocket_approval.prompts import APPROVE_MORTGAGE_PROMPT
from arcade_rocket_approval.tool_utils import (
    create_tool_function,
    tool_definition_to_pydantic_model,
)

logger = logging.getLogger(__name__)


def create_mortgage_tool_node(
    tool_name: str,
    next_step: str,
) -> Callable:
    """Create a node function for a mortgage tool.

    Args:
        tool_name: Name of the tool to execute
        next_step: The next step to transition to after this tool completes

    Returns:
        A node function that executes the tool and updates state
    """
    client = Arcade()

    def tool_node(state: State, config: RunnableConfig) -> Dict:
        """Node function that executes a mortgage tool.

        The function extracts tool arguments from the tool call in the state,
        executes the tool, and updates the state with the result.
        """
        # Get the last tool call from the state
        tool_calls = state["messages"][-1].tool_calls
        tool_call_id = tool_calls[0]["id"]
        tool_args = tool_calls[0]["args"]

        user_id = config.get("configurable", {}).get("user_id") if config else None

        params = {
            "rm_loan_id": state.get("current_rm_loan_id", None),
            "session_token": state.get("current_session_token", None),
        }
        if tool_name.startswith("RocketApproval") and not tool_name.endswith(
            "StartMortgageApplication"
        ):
            tool_args.update(params)

        print(f"tool_args: {tool_args}")
        # Execute the tool
        execute_response = client.tools.execute(
            tool_name=tool_name,
            input=tool_args,
            user_id=user_id if user_id is not None else NOT_GIVEN,
        )

        if not execute_response.success or not execute_response.output:
            # Handle error case
            error_message = "Error executing tool"
            if execute_response.output and execute_response.output.error:
                error_message = execute_response.output.error.message

            return {
                "messages": [
                    ToolMessage(
                        content=f"Error: {error_message}\nPlease try again with valid inputs.",
                        tool_call_id=tool_call_id,
                    )
                ],
                # "current_step": tool_name + "_node",
            }

        # Tool executed successfully
        result = execute_response.output.value if execute_response.output else {}

        # Extract rm_loan_id if available
        rm_loan_id = result.get("rmLoanId", None)
        if rm_loan_id:
            current_rm_loan_id = rm_loan_id
        else:
            current_rm_loan_id = state.get("current_rm_loan_id", "Not set yet")

        # Extract session_token if available
        session_token = result.get("sessionToken", None)
        if session_token:
            current_session_token = session_token
        else:
            current_session_token = state.get("current_session_token", None)

        # Prepare response message
        success_message = f"Successfully completed {tool_name}"
        if "message" in result:
            success_message = result["message"]

        return {
            "messages": [
                ToolMessage(
                    content=success_message,
                    tool_call_id=tool_call_id,
                )
            ],
            "current_rm_loan_id": current_rm_loan_id,
            "current_session_token": current_session_token,
            "current_step": next_step,
        }

    return tool_node


def create_mortgage_tool(
    tool_name: str,
    next_step: str,
) -> StructuredTool:
    """Create a structured tool for mortgage operations.

    Args:
        tool_name: Name of the tool
        next_step: The next step to transition to after this tool completes

    Returns:
        A structured tool that can be used by LangChain
    """
    client = Arcade()
    tool_def = client.tools.get(tool_name)
    args_schema = tool_definition_to_pydantic_model(tool_def)
    tool_name_clean = tool_name.replace(".", "_")

    # Create the tool function but don't worry about Command objects
    # as we'll handle state updates in the node
    tool_function = create_tool_function(
        client=client,
        tool_name=tool_name,
        tool_def=tool_def,
        args_schema=args_schema,
        langgraph=False,  # Don't need langgraph mode since we handle state in nodes
    )

    return StructuredTool.from_function(
        tool_function,
        name=tool_name_clean,
        args_schema=args_schema,
        description=tool_def.description,
    )


def application_node_tools() -> list[StructuredTool]:
    """Create a list of structured tools for the mortgage application.

    Returns:
        A list of structured tools for the mortgage application
    """
    wrapped = []
    # Define the mortgage application tools
    for tool_name, next_step in ALL_TOOLS.items():
        wrapped.append(create_mortgage_tool(tool_name, next_step))
    return wrapped


def application_nodes() -> Dict[str, Callable]:
    """Create a dictionary of node functions for the mortgage application.

    Returns:
        A dictionary mapping node names to node functions
    """
    nodes = {}
    # Define the mortgage application flow with steps and transitions
    for tool_name, next_step in ALL_TOOLS.items():
        node_name = tool_name.replace(".", "_") + "_node"
        nodes[node_name] = create_mortgage_tool_node(tool_name, next_step)
    return nodes


def create_entry_node(assistant_name: str, new_dialog_state: str) -> Callable:
    def entry_node(state: State) -> dict:
        tool_call_id = state["messages"][-1].tool_calls[0]["id"]
        # Initialize default values for mortgage assistant fields if they're missing
        current_rm_loan_id = state.get("current_rm_loan_id", "Not set yet")
        current_step = state.get(
            "current_step", "RocketApproval.StartMortgageApplication"
        )

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


ALL_TOOLS = {
    "RocketApproval.StartMortgageApplication": "RocketApproval.SetNewHomeDetails",
    "RocketApproval.SetNewHomeDetails": "RocketApproval.SetHomePrice",
}

APPLICATION_NODES = {}
if not APPLICATION_NODES:
    APPLICATION_NODES = application_nodes()

APPLICATION_TOOLS = []
if not APPLICATION_TOOLS:
    APPLICATION_TOOLS = application_node_tools()


def route_approve_mortgage(
    state: State,
):
    """Route to the appropriate mortgage tool node or leave the skill.

    This function checks the tool calls in the state and routes to the
    appropriate tool node based on the tool name, or to leave_skill
    if the user wants to cancel.
    """
    route = tools_condition(state)
    if route == END:
        return END

    tool_calls = state["messages"][-1].tool_calls
    if not tool_calls:
        return "approve_mortgage"  # No tool calls, back to LLM

    # Check if user wants to cancel
    did_cancel = any(tc["name"] == "CompleteOrEscalate" for tc in tool_calls)
    if did_cancel:
        return "leave_skill"

    # Find the appropriate tool node based on the tool name
    tool_name = tool_calls[0]["name"]
    # Convert to the node name format
    tool_node_name = tool_name + "_node"

    # If the node exists, route to it
    if tool_node_name in APPLICATION_NODES:
        return tool_node_name

    # Otherwise route to the LLM with tools
    return "approve_mortgage_safe_tools"


def get_mortgage_assistant(
    llm: BaseLanguageModel,
) -> tuple[Runnable, list[StructuredTool], Dict[str, Callable]]:
    """Get the mortgage assistant and associated tools.

    Args:
        llm: The language model to use

    Returns:
        A tuple containing:
        - The mortgage assistant runnable
        - The list of structured tools for the mortgage assistant
        - A dictionary of tool nodes
    """
    # Tools for the mortgage approval process
    approve_mortgage_safe_tools = APPLICATION_TOOLS

    # Create a mapping of all tool nodes
    mortgage_nodes = APPLICATION_NODES

    # Create the mortgage assistant runnable
    safe_mortgage_tools = [t for t in approve_mortgage_safe_tools]
    approve_mortgage_runnable = APPROVE_MORTGAGE_PROMPT | llm.bind_tools(
        safe_mortgage_tools + [CompleteOrEscalate]
    )

    return approve_mortgage_runnable, approve_mortgage_safe_tools, mortgage_nodes
