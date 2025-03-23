import datetime
import logging
import uuid
from typing import Annotated, Any, Callable, Dict, List, Literal, Optional

from langchain.tools import BaseTool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langchain_core.tools import StructuredTool, tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from arcade_rocket_approval.api import (
    Address,
    ContactInfo,
    CurrentLivingSituation,
    HomeDetails,
    HomePurchase,
    Location,
    PersonalInfo,
    PhoneNumber,
    PrimaryAssets,
    RealEstateAgent,
    create_account,
    do_soft_credit_pull,
    set_contact_info,
    set_funds,
    set_home_details,
    set_home_price,
    set_income,
    set_living_situation,
    set_marital_status,
    set_military_status,
    set_personal_info,
    set_real_estate_agent,
    start_application,
)

# The top-level assistant performs general Q&A and delegates specialized tasks to other assistants.
# The task delegation is a simple form of semantic routing / does simple intent detection
# llm = ChatAnthropic(model="claude-3-haiku-20240307")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

logger = logging.getLogger(__name__)

###########
# Mortgage approval tools
###########


@tool
def start_mortgage_application(
    tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig
) -> Command:
    """Start a new mortgage application. First step in the application process always"""
    response = start_application()
    print(response.raw_response)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"New mortgage application started successfully. Loan ID: {response.data['rmLoanId']}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_rm_loan_id": response.data["rmLoanId"],
                "current_step": "home_details",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to start mortgage application: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_property_details(
    city: str,
    state: str,
    zip_code: str,
    property_type: Literal["Single Family", "Multi Family", "Condo", "Townhouse"],
    occupancy_type: Literal["Primary Residence", "Investment Property", "Second Home"],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set property details for a mortgage application."""
    print(
        f"Setting property details for loan ID: {config.get('configurable', {}).get('rm_loan_id', None)}"
    )
    print(f"City: {city}, State: {state}, Zip Code: {zip_code}")
    print(f"Property Type: {property_type}, Occupancy Type: {occupancy_type}")
    print(f"Tool Call ID: {tool_call_id}")
    print(f"Config: {config}")
    location = Location(city=city, state=state, zip_code=zip_code)
    home_details = HomeDetails(
        location=location,
        property_type=property_type,
        occupancy_type=occupancy_type,
    )
    print(f"Home Details: {home_details}")
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_home_details(current_rm_loan_id, home_details)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Home details updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "home_price",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update home details: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_home_price(
    has_budget: bool,
    desired_price: int,
    minimum_price: Optional[int],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the desired home price information for a mortgage application."""
    print(
        f"Setting home price for loan ID: {config.get('configurable', {}).get('rm_loan_id', None)}"
    )
    print(
        f"Has budget: {has_budget}, Desired price: {desired_price}, Minimum price: {minimum_price}"
    )
    print(f"Tool Call ID: {tool_call_id}")
    print(f"Config: {config}")
    purchase = HomePurchase(
        has_budget=has_budget,
        desired_price=desired_price,
        minimum_price=minimum_price,
    )
    print(f"Purchase: {purchase}")
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )
    response = set_home_price(current_rm_loan_id, purchase)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Home price information updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_rm_loan_id": current_rm_loan_id,
                "current_step": "real_estate_agent",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update home price information: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_real_estate_agent(
    has_agent: bool,
    agent_first_name: Optional[str],
    agent_last_name: Optional[str],
    agent_email_address: Optional[str],
    agent_phone_area_code: Optional[str],
    agent_phone_prefix: Optional[str],
    agent_phone_line: Optional[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the real estate agent information for a mortgage application."""
    work_phone = None
    if has_agent and agent_phone_area_code and agent_phone_prefix and agent_phone_line:
        work_phone = PhoneNumber(
            area_code=agent_phone_area_code,
            prefix=agent_phone_prefix,
            line=agent_phone_line,
        )

    agent = RealEstateAgent(
        has_agent=has_agent,
        first_name=agent_first_name or "",
        last_name=agent_last_name or "",
        email_address=agent_email_address or "",
        work_phone=work_phone,
    )

    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_real_estate_agent(current_rm_loan_id, agent)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Real estate agent information updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "living_situation",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update real estate agent information: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_living_situation(
    rent_or_own: Literal["Rent", "Own"],
    street: str,
    street2: Optional[str],
    city: str,
    state: str,
    zip_code: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the living situation information for a mortgage application."""
    address = Address(
        street=street,
        street2=street2,
        city=city,
        state=state,
        zip_code=zip_code,
    )

    living_situation = CurrentLivingSituation(rent_or_own=rent_or_own, address=address)

    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_living_situation(current_rm_loan_id, living_situation)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Living situation updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "personal_info",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update living situation: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_marital_status(
    marital_status: Literal["Married", "Single"],
    is_spouse_on_loan: Optional[bool],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the marital status information for a mortgage application."""
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_marital_status(current_rm_loan_id, marital_status, is_spouse_on_loan)

    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Marital status updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "military_status",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update marital status: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_military_status(
    military_status: Literal["Active Duty", "Reserve", "None"],
    military_branch: Optional[
        Literal["Army", "Navy", "Air Force", "Marine Corps", "None"]
    ],
    service_type: Optional[Literal["Regular", "Reserve", "None"]],
    expiration_date: Optional[str],
    eligible_for_va: bool,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the military status information for a mortgage application."""
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_military_status(
        current_rm_loan_id,
        military_status,
        military_branch,
        service_type,
        expiration_date,
        eligible_for_va,
    )

    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Military status updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "contact_info",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update military status: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_personal_info(
    first_name: str,
    last_name: str,
    date_of_birth: str,
    marital_status: Literal["Married", "Single"],
    is_spouse_on_loan: Optional[bool],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the personal information for a mortgage application."""
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_personal_info(
        current_rm_loan_id,
        first_name,
        last_name,
        date_of_birth,
        marital_status,
        is_spouse_on_loan,
    )

    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Personal information updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "marital_status",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update personal information: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_contact_info(
    email: str,
    area_code: str,
    prefix: str,
    line: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the contact information for a mortgage application."""
    phone_number = PhoneNumber(area_code=area_code, prefix=prefix, line=line)

    contact_info = ContactInfo(email=email, phone_number=phone_number)

    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_contact_info(current_rm_loan_id, contact_info=contact_info)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Contact information updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "income",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update contact information: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def set_user_income(
    annual_income: int,
    income_type: Literal[
        "Employment",
        "Self-Employed",
        "Unemployment",
        "Social Security",
        "Pension",
        "Other",
    ],
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Set the income information for a mortgage application."""
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = set_income(
        current_rm_loan_id,
        annual_income,
        income_type,
    )

    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Income information updated successfully for loan ID: {current_rm_loan_id}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "credit_check",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to update income information: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def do_user_credit_pull(
    birthdate: str,
    ssn_last4: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Perform a soft credit pull for a mortgage application."""
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = do_soft_credit_pull(current_rm_loan_id, birthdate, ssn_last4)
    if response.success:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Credit check completed successfully for loan ID: {current_rm_loan_id}. You're eligible for a mortgage!",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "create_account",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to complete credit check: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


@tool
def create_user_account(
    first_name: str,
    last_name: str,
    username: str,
    password: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    config: RunnableConfig,
) -> Command:
    """Create a Rocket Mortgage account after completing the application."""
    # Get the loan ID from config
    current_rm_loan_id = config.get("configurable", {}).get("rm_loan_id", None)
    if not current_rm_loan_id:
        current_rm_loan_id = (
            "default_loan_id"  # Or handle this error case appropriately
        )

    response = create_account(
        current_rm_loan_id,
        first_name,
        last_name,
        username,
        password,
    )

    if response.success:
        account_id = response.data.get("rocketAccountId", "Unknown")
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Mortgage account created successfully! Your Rocket Account ID is: {account_id}. Your mortgage application is now complete.",
                        tool_call_id=tool_call_id,
                    )
                ],
                "current_step": "completed",
            }
        )
    else:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Failed to create mortgage account: {response.message}",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_pydantic_tool(model_cls):
    """Create a tool function from a Pydantic model class that has a __call__ method."""
    # Get the schema from the Pydantic model
    schema = model_cls.model_json_schema(by_alias=False)

    # Process the schema to ensure it's compatible with the tool decorator
    properties = schema.get("properties", {})
    schema_properties = {k: v for k, v in properties.items()}

    # Define the tool function without specific parameter annotations
    def model_tool(
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig,
        **kwargs,
    ):
        """Dynamic tool function that instantiates and calls the Pydantic model."""
        try:
            # Create an instance of the model with the remaining kwargs
            instance = model_cls(**kwargs)

            # Call the instance's __call__ method with the required parameters
            return instance(tool_call_id=tool_call_id, config=config)
        except Exception as e:
            logger.exception("Tool Execution Failed: %s", e)
            raise RuntimeError(f"Tool Execution Failed: {e}") from e

    # Create a properly decorated tool with schema information
    decorated_tool = StructuredTool.from_function(
        model_tool,
        description=model_cls.__doc__,
        args_schema=schema,  # Include the full schema for better LLM understanding
        name=model_cls.__name__,  # Use the class name as the tool name
    )

    return decorated_tool


def create_tool_node_with_fallback(tools: list) -> dict:
    """Create a ToolNode with proper handling of Pydantic model tools."""
    processed_tools = []
    for t in tools:
        if isinstance(t, type) and issubclass(t, BaseModel):
            # For Pydantic model classes, create a proper tool
            processed_tools.append(create_pydantic_tool(t))
        else:
            # For regular functions or already instantiated tools
            processed_tools.append(t)

    return ToolNode(processed_tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


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


def update_dialog_stack(left: list[str], right: Optional[str]) -> list[str]:
    """Push or pop the state."""
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    dialog_state: Annotated[
        list[
            Literal[
                "assistant",
                "approve_mortgage",
            ]
        ],
        update_dialog_stack,
    ]
    current_rm_loan_id: Optional[str]
    current_step: Optional[str]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed and/or to escalate control of the dialog to the main assistant,
    who can re-route the dialog based on the user's needs."""

    cancel: bool = True
    reason: str

    class Config:
        json_schema_extra = {
            "example": {
                "cancel": True,
                "reason": "User changed their mind about the current task.",
            },
            "example 2": {
                "cancel": True,
                "reason": "I have fully completed the task.",
            },
            "example 3": {
                "cancel": False,
                "reason": "I need to search the user's emails or calendar for more information.",
            },
        }

    def __call__(
        self, tool_call_id: Annotated[str, InjectedToolCallId], config: RunnableConfig
    ) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=f"Task status: {'Completed' if self.cancel else 'In progress'} - {self.reason}",
                        tool_call_id=tool_call_id,
                    )
                ],
                "dialog_state": "pop" if self.cancel else None,
            }
        )


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


# Mortgage Assistant
approve_mortgage_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a specialized assistant for handling mortgage approvals. "
            "The primary assistant delegates work to you whenever the user needs help approving a mortgage. "
            "Walk the user through each step of the mortgage application process in order, starting with home details and ending with account creation. "
            "For each step, collect all required information, then use the appropriate tool to submit it. "
            "The steps should be followed in this order: "
            "1. Start application (if no loan ID exists) "
            "2. Personal information "
            "3. Contact information "
            "4. Home details "
            "5. Home price "
            "6. Real estate agent information "
            "7. Current living situation "
            "8. Marital status "
            "9. Military status "
            "10. Income information "
            "11. Credit check "
            "12. Create account "
            "If a step fails, help the user correct their information and try again. "
            "You can also search for mortgage policies based on the user's preferences to help them understand their options. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            " Remember that an application isn't completed until all steps have been successfully processed."
            "\nCurrent time: {time}."
            "\nCurrent loan ID: {current_rm_loan_id}"
            "\nCurrent step: {current_step}"
            "\n\nIf the user needs help, and none of your tools are appropriate for it, then "
            '"CompleteOrEscalate" the dialog to the host assistant. Do not waste the user\'s time. Do not make up invalid tools or functions.'
            "\n\nSome examples for which you should CompleteOrEscalate:\n"
            " - 'actually I need something else'\n"
            " - 'nevermind I don't need to approve a mortgage'\n"
            " - 'stop the process'\n"
            " - 'I need to cancel the mortgage application'\n",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now)

# Tools for the mortgage approval process
approve_mortgage_safe_tools = [
    start_mortgage_application,
    set_property_details,
    set_user_home_price,
    set_user_real_estate_agent,
    set_user_living_situation,
    set_user_marital_status,
    set_user_military_status,
    set_user_personal_info,
    set_user_contact_info,
    set_user_income,
]
approve_mortgage_sensitive_tools = [
    do_user_credit_pull,
    create_user_account,
]

# Create a mapping of all registered property detail tools
safe_mortgage_tools = [t for t in approve_mortgage_safe_tools]
sensitive_mortgage_tools = [t for t in approve_mortgage_sensitive_tools]
approve_mortgage_runnable = approve_mortgage_prompt | llm.bind_tools(
    safe_mortgage_tools + sensitive_mortgage_tools + [CompleteOrEscalate]
)


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
    sensitive_toolnames = [t.name for t in approve_mortgage_sensitive_tools]
    if any(tc["name"] in sensitive_toolnames for tc in tool_calls):
        return "approve_mortgage_sensitive_tools"
    return "approve_mortgage_safe_tools"


primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for a mortgage company. "
            "Your primary role is to search for mortgage information and company policies to answer customer queries. "
            "If a customer requests to apply for a mortgage, update an existing application, or cancel a mortgage, "
            "delegate the task to the specialized mortgage assistant by invoking the ToApproveMortgage tool. You are not able to make these types of changes yourself."
            " Only the specialized assistant is given permission to do this for the user."
            "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
            "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            "\nCurrent time: {time}.",
        ),
        ("placeholder", "{messages}"),
    ]
).partial(time=datetime.datetime.now)
primary_assistant_tools = []
assistant_runnable = primary_assistant_prompt | llm.bind_tools(
    primary_assistant_tools
    + [
        ToApproveMortgage,
    ]
)


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
    create_tool_node_with_fallback(safe_mortgage_tools),
)
builder.add_node(
    "approve_mortgage_sensitive_tools",
    create_tool_node_with_fallback(sensitive_mortgage_tools),
)

builder.add_edge("approve_mortgage_sensitive_tools", "approve_mortgage")
builder.add_edge("approve_mortgage_safe_tools", "approve_mortgage")
builder.add_conditional_edges(
    "approve_mortgage",
    route_approve_mortgage,
    [
        "approve_mortgage_safe_tools",
        "approve_mortgage_sensitive_tools",
        "leave_skill",
        END,
    ],
)
# Primary assistant
builder.add_node("primary_assistant", Assistant(assistant_runnable))
builder.add_node(
    "primary_assistant_tools", create_tool_node_with_fallback(primary_assistant_tools)
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
part_4_graph = builder.compile(
    checkpointer=memory,
    # Let the user approve or deny the use of sensitive tools
    interrupt_before=[
        "approve_mortgage_sensitive_tools",
    ],
    debug=True,
)


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

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            events = part_4_graph.stream(
                {"messages": ("user", user_input)}, config, stream_mode="values"
            )
            print("\nAssistant: ", end="")
            for event in events:
                _print_event(event, _printed)

            snapshot = part_4_graph.get_state(config)
            while snapshot.next:
                # We have an interrupt! The agent is trying to use a tool, and the user can approve or deny it
                try:
                    approval = input(
                        "\nDo you approve of the above actions? Type 'y' to continue;"
                        " otherwise, explain your requested changed.\n\nYour response: "
                    )
                except KeyboardInterrupt:
                    print("\nExiting chat...")
                    return

                if approval.strip().lower() == "y":
                    # Just continue
                    result = part_4_graph.invoke(
                        None,
                        config,
                    )
                else:
                    # Satisfy the tool invocation by
                    # providing instructions on the requested changes / change of mind
                    result = part_4_graph.invoke(
                        {
                            "messages": [
                                ToolMessage(
                                    tool_call_id=event["messages"][-1].tool_calls[0][
                                        "id"
                                    ],
                                    content=f"API call denied by user. Reasoning: '{approval}'. Continue assisting, accounting for the user's input.",
                                )
                            ]
                        },
                        config,
                    )
                # Print the result
                for event in result.get("events", []):
                    _print_event(event, _printed)

                snapshot = part_4_graph.get_state(config)

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            logger.exception(e)
            print(f"\nAn error occurred: {str(e)}")


if __name__ == "__main__":
    run_chat_interface()
