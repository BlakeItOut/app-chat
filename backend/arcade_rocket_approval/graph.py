import uuid
from typing import TYPE_CHECKING, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt

from arcade_rocket_approval.api import (
    BankingAsset,
    PhoneNumber,
    PrimaryAssets,
    RocketUserContext,
    do_soft_credit_pull,
    set_contact_info,
    set_funds,
    set_home_details,
    set_home_price,
    set_income,
    set_military_status,
    set_personal_info,
    start_application,
)
from arcade_rocket_approval.defaults import (
    CHECKPOINTER,
    INFO_MODEL,
    MODEL,
    get_cached_tools,
    load_chat_model,
)

if TYPE_CHECKING:
    from arcade_rocket_approval.graph import MortgageState
from arcade_rocket_approval.agent.info_agent import get_user_info_agent


class MortgageState(MessagesState):
    rmLoanId: str | None = None
    user_id: str | None = None
    user_info: RocketUserContext | None = None


def info_gathering_node(
    state: MortgageState,
    config: RunnableConfig,
) -> Command[Literal["verify_user_info_node"]]:
    """
    This node initializes or runs the information gathering agent.
    Once information is collected, it transitions to verification.
    """

    if state.get("error"):
        # Add error message to guide the agent to fix the issue
        state["messages"].append(
            AIMessage(
                content=f"I need to correct some information: {state['error']}. Please help me collect the right information."
            )
        )

    # Run the info gathering agent
    info_agent = get_user_info_agent(
        model=load_chat_model(INFO_MODEL),
        tools=get_cached_tools(),  # TODO cache this
        checkpointer=CHECKPOINTER,
    )

    result = info_agent.invoke(input=state, config=config)

    # Extract the user info from the result
    user_info = result.get("user_info")

    # Move to verification
    return Command(
        goto="verify_user_info_node",
        update={
            "user_info": user_info,
            "messages": result.get("messages", []),
        },
    )


def create_config(user_id: str = None) -> RunnableConfig:
    """Create a RunnableConfig with necessary checkpoint configuration."""
    # TODO replace with langgraphcloud auth (supabase or auth0)
    user_id = user_id or str(uuid.uuid4())

    # Create config with all required checkpoint parameters
    config = RunnableConfig(
        user_id=user_id,
        thread_id=str(uuid.uuid4()),
        checkpoint_ns="mortgage_approval_agent",
        checkpoint_id=str(uuid.uuid4()),
    )

    return config


def verify_user_info_node(
    state: MortgageState,
) -> Command[Literal["start_application_node"]]:
    """
    Presents the collected user information to the user for verification.
    If verified, proceed to start_application_node, otherwise return to info_gathering_node.
    """
    if not state.get("user_info"):
        # If no user info available, we need to gather it
        return Command(goto="info_gathering_node", update={})

    # Format user info for display
    user_info = state["user_info"]
    info_display = user_info.info_display()

    # Interrupt to get user verification
    user_input = interrupt(info_display)
    user_reply = user_input["value"].strip().lower()

    if user_reply.startswith("y"):
        return Command(
            goto="start_application_node",
            update={
                "messages": [
                    AIMessage(content=info_display),
                    HumanMessage(content=user_reply),
                    AIMessage(
                        content="Thank you for confirming. Let's start your application."
                    ),
                ]
            },
        )
    else:
        return Command(
            goto="info_gathering_node",
            update={
                "messages": [
                    AIMessage(content=info_display),
                    HumanMessage(content=user_reply),
                    AIMessage(content="Let's update your information."),
                ]
            },
        )


def start_application_node(
    state: MortgageState,
) -> Command[Literal["home_details_node", "__end__"]]:
    """
    Creates a new mortgage application using the collected user information.
    """
    # Create welcome message
    user_prompt = "Ready to start your Purchase application? (yes/no)"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    # A robust check for yes/no
    user_reply_cleaned = user_input["value"].strip().lower()
    if not user_reply_cleaned.startswith("y"):
        # If user said "no", we end the application process
        new_messages = [
            HumanMessage(content=user_reply_cleaned),
            AIMessage(content="Understood. We won't start the application now."),
        ]
        return Command(goto="__end__", update={"messages": new_messages})

    try:
        rm_loan_id = start_application()
        return Command(
            goto="home_details_node",
            update={"rmLoanId": rm_loan_id, "messages": messages},
        )
    except Exception as e:
        error_messages = messages + [
            AIMessage(
                content=f"I encountered an error starting your application: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def home_details_node(
    state: MortgageState,
) -> Command[Literal["home_price_node", "__end__"]]:
    """
    This node corresponds to 'purchase/home-info/buying-plans/home-details'.
    For example, ask for home value, monthly payment, etc.
    """
    user_prompt = "Share your estimated home value and desired monthly payment."
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    try:
        # Process home details
        # This would normally call an API endpoint
        # set_home_details(rm_loan_id=state["rmLoanId"], details=user_input["value"])

        # Next route is 'purchase/home-info/buying-plans/home-price'
        return Command(goto="home_price_node", update={"messages": messages})
    except Exception as e:
        error_messages = messages + [
            AIMessage(
                content=f"I encountered an error with your home details: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def home_price_node(
    state: MortgageState,
) -> Command[Literal["personal_info_node", "__end__"]]:
    """
    This node corresponds to 'purchase/home-info/buying-plans/home-price'.
    Ask for a bit more info about the purchase price if needed.
    """
    user_prompt = "Do you know your target home price?"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    try:
        # Process home price info
        response = set_home_price(
            rm_loan_id=state["rmLoanId"], price_info=user_input["value"]
        )

        # Next big section is "purchase/personal-info" (the interstitial)
        return Command(goto="personal_info_node", update={"messages": messages})
    except Exception as e:
        error_messages = messages + [
            AIMessage(
                content=f"I encountered an error with your home price information: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def personal_info_node(
    state: MortgageState,
) -> Command[Literal["contact_info_node", "__end__"]]:
    """
    This would correspond to 'purchase/personal-info' before we jump to contact info routes, etc.
    """
    # Extract user info
    user_info = state.get("user_info", {})

    try:
        if (
            state["rmLoanId"]
            and hasattr(user_info, "first_name")
            and hasattr(user_info, "last_name")
        ):
            # Call API to set personal info using collected user data
            response = set_personal_info(
                rm_loan_id=state["rmLoanId"],
                first_name=user_info.first_name,
                last_name=user_info.last_name,
                date_of_birth=getattr(user_info, "date_of_birth", None),
                marital_status=getattr(user_info, "marital_status", None),
                is_spouse_on_loan=getattr(user_info, "is_spouse_on_loan", None),
            )

            if response.status == "error":
                return Command(
                    goto="info_gathering_node",
                    update={
                        "messages": [
                            AIMessage(
                                content=f"I encountered an error with your personal information: {response.message}. Let's gather more information."
                            )
                        ],
                        "error": response.message,
                    },
                )

        # User prompt for additional info if needed
        user_prompt = "Next, let's handle personal info. Enter marital status or any relevant info."
        user_input = interrupt(user_prompt)
        messages = [
            AIMessage(content=user_prompt),
            HumanMessage(content=user_input["value"]),
        ]

        return Command(goto="contact_info_node", update={"messages": messages})
    except Exception as e:
        error_messages = [
            AIMessage(
                content=f"I encountered an error with your personal information: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def contact_info_node(
    state: MortgageState,
) -> Command[Literal["military_status_node", "__end__"]]:
    """
    'purchase/personal-info/contact-info'
    Ask user for phone, email, etc.
    """
    # Extract user info
    user_info = state.get("user_info", {})

    try:
        if state["rmLoanId"] and hasattr(user_info, "email"):
            # Extract phone number components from user info
            phone_str = getattr(user_info, "phone_number", "")

            # Create phone number object
            phone = None
            if phone_str:
                # This is a simplification - in reality you'd want to parse the phone string properly
                phone_parts = (
                    phone_str.replace("(", "")
                    .replace(")", "")
                    .replace("-", "")
                    .replace(" ", "")
                )
                if len(phone_parts) == 10:
                    phone = PhoneNumber(
                        area_code=phone_parts[:3],
                        prefix=phone_parts[3:6],
                        line=phone_parts[6:],
                    )

                # Call API to set contact info
                response = set_contact_info(
                    rm_loan_id=state["rmLoanId"],
                    first_name=user_info.first_name,
                    last_name=user_info.last_name,
                    email=user_info.email,
                    phone_number=phone,
                )

                if response.status == "error":
                    return Command(
                        goto="info_gathering_node",
                        update={
                            "messages": [
                                AIMessage(
                                    content=f"I encountered an error with your contact information: {response.message}. Let's gather more information."
                                )
                            ],
                            "error": response.message,
                        },
                    )

        # User prompt for confirmation or additional info
        user_prompt = "Please confirm your email and phone number: is this correct? If not, please provide them comma separated."
        user_input = interrupt(user_prompt)
        messages = [
            AIMessage(content=user_prompt),
            HumanMessage(content=user_input["value"]),
        ]

        return Command(goto="military_status_node", update={"messages": messages})
    except Exception as e:
        error_messages = [
            AIMessage(
                content=f"I encountered an error with your contact information: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def military_status_node(
    state: MortgageState,
) -> Command[Literal["finances_node", "__end__"]]:
    """
    'purchase/personal-info/military-status'
    Possibly ask if user is military, etc.
    Then we proceed to 'finances'...
    """
    # Extract user info
    user_info = state.get("user_info", {})

    try:
        military_status = getattr(user_info, "military_status", None)
        if state["rmLoanId"] and military_status:
            # Call API to set military status
            response = set_military_status(
                rm_loan_id=state["rmLoanId"], military_status=military_status
            )

            if response.status == "error":
                return Command(
                    goto="info_gathering_node",
                    update={
                        "messages": [
                            AIMessage(
                                content=f"I encountered an error with your military status: {response.message}. Let's gather more information."
                            )
                        ],
                        "error": response.message,
                    },
                )

        # User prompt for confirmation or additional info
        user_prompt = (
            "Are you or a co-borrower associated with the Military? Yes or No?"
        )
        user_input = interrupt(user_prompt)
        messages = [
            AIMessage(content=user_prompt),
            HumanMessage(content=user_input["value"]),
        ]

        return Command(goto="finances_node", update={"messages": messages})
    except Exception as e:
        error_messages = [
            AIMessage(
                content=f"I encountered an error with your military status: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def finances_node(
    state: MortgageState,
) -> Command[Literal["income_node", "__end__"]]:
    """
    'purchase/finances' as an interstitial or direct route
    """
    user_prompt = "Let's talk about finances. Press enter to continue."
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    return Command(goto="income_node", update={"messages": messages})


def income_node(
    state: MortgageState,
) -> Command[Literal["funds_node", "__end__"]]:
    """
    'purchase/finances/income'
    """
    # Extract user info
    user_info = state.get("user_info", {})

    try:
        annual_income = getattr(user_info, "annual_income", None)
        if state["rmLoanId"] and annual_income:
            # Call API to set income
            response = set_income(
                rm_loan_id=state["rmLoanId"],
                annual_income=annual_income,
                income_type="Employment",  # Default to employment
            )

            if response.status == "error":
                return Command(
                    goto="info_gathering_node",
                    update={
                        "messages": [
                            AIMessage(
                                content=f"I encountered an error with your income information: {response.message}. Let's gather more information."
                            )
                        ],
                        "error": response.message,
                    },
                )

        # User prompt for confirmation or additional info
        user_prompt = "What is your annual income?"
        user_input = interrupt(user_prompt)
        messages = [
            AIMessage(content=user_prompt),
            HumanMessage(content=user_input["value"]),
        ]

        return Command(goto="funds_node", update={"messages": messages})
    except Exception as e:
        error_messages = [
            AIMessage(
                content=f"I encountered an error with your income information: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def funds_node(
    state: MortgageState,
) -> Command[Literal["credit_info_node", "__end__"]]:
    """
    'purchase/finances/funds'
    """
    user_prompt = "How will you fund your down payment? (savings, gift, etc.)"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    try:
        if state["rmLoanId"] and user_input["value"].strip():
            # Basic funds processing - in a real implementation you'd parse the user's input
            # and create proper objects based on their response
            if "savings" in user_input["value"].lower():
                # Create a banking asset for savings
                primary_assets = PrimaryAssets(
                    assets=[
                        BankingAsset(
                            bank_amount=50000,  # This would be extracted from user input
                            bank_name="User's Bank",
                            type_code="Savings",
                        )
                    ]
                )

                # Call API to set funds
                response = set_funds(
                    rm_loan_id=state["rmLoanId"], primary_assets=primary_assets
                )

                if response.status == "error":
                    return Command(
                        goto="info_gathering_node",
                        update={
                            "messages": [
                                AIMessage(
                                    content=f"I encountered an error with your funds information: {response.message}. Let's gather more information."
                                )
                            ],
                            "error": response.message,
                        },
                    )

        # Move on to credit info
        return Command(goto="credit_info_node", update={"messages": messages})
    except Exception as e:
        error_messages = messages + [
            AIMessage(
                content=f"I encountered an error processing your funds information: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def credit_info_node(
    state: MortgageState,
) -> Command[Literal["pull_credit_node", "__end__"]]:
    """
    'purchase/credit-info'
    """
    user_prompt = "We'll talk about your credit info next. Press enter to continue."
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    return Command(goto="pull_credit_node", update={"messages": messages})


def pull_credit_node(
    state: MortgageState,
) -> Command[Literal["affordability_amount_node", "__end__"]]:
    """
    'purchase/credit-info/birthdate-SSN'
    Ask for minimal data to do a soft credit pull, if needed.
    """
    user_prompt = "Enter your birthdate and last 4 of SSN, comma separated."
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    try:
        if state["rmLoanId"] and "," in user_input["value"]:
            parts = user_input["value"].split(",", 1)
            birthdate = parts[0].strip()
            ssn_last4 = parts[1].strip()

            # Call API to do soft credit pull
            response = do_soft_credit_pull(
                rm_loan_id=state["rmLoanId"], birthdate=birthdate, ssn_last4=ssn_last4
            )

            if response.status == "error":
                return Command(
                    goto="info_gathering_node",
                    update={
                        "messages": [
                            AIMessage(
                                content=f"I encountered an error with your credit information: \
                                    {response.message}. Let's gather more information."
                            )
                        ],
                        "error": response.message,
                    },
                )

        # Transition to final or to "purchase/chat-with-expert/affordability-amount"
        return Command(goto="affordability_amount_node", update={"messages": messages})
    except Exception as e:
        error_messages = messages + [
            AIMessage(
                content=f"I encountered an error processing your credit information: {str(e)}. Let's gather more information."
            )
        ]
        return Command(
            goto="info_gathering_node",
            update={"messages": error_messages, "error": str(e)},
        )


def affordability_amount_node(state: MortgageState) -> Command[Literal["__end__"]]:
    """
    'purchase/chat-with-expert/affordability-amount'
    Possibly handle final route or chat with expert.
    """
    user_prompt = "Let's see your affordability range. Press enter to finalize."
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
        AIMessage(
            content="Great! Your mortgage application is now complete. A Rocket Mortgage expert will be in touch with you shortly to discuss next steps."
        ),
    ]

    # We can end the flow or loop back. For now, let's just end.
    return Command(goto="__end__", update={"messages": messages})




################################################################################
# Build the entire Workflow
################################################################################


def build_mortgage_workflow(
    model: BaseChatModel, tools: list[BaseTool], checkpointer: MemorySaver, **kwargs
):
    """
    Builds and returns the complete mortgage application workflow.

    Args:
        model: The language model to use for the workflow
        tools: The tools to use for the workflow
        checkpointer: The checkpointer to use for state persistence
        **kwargs: Additional arguments to pass to the graph compiler

    Returns:
        A compiled state graph representing the mortgage application workflow
    """
    builder = StateGraph(MortgageState)

    # Add all nodes
    builder.add_node("info_gathering_node", info_gathering_node)
    builder.add_node("verify_user_info_node", verify_user_info_node)
    builder.add_node("start_application_node", start_application_node)
    builder.add_node("home_details_node", home_details_node)
    builder.add_node("home_price_node", home_price_node)
    builder.add_node("personal_info_node", personal_info_node)
    builder.add_node("contact_info_node", contact_info_node)
    builder.add_node("military_status_node", military_status_node)
    builder.add_node("finances_node", finances_node)
    builder.add_node("income_node", income_node)
    builder.add_node("funds_node", funds_node)
    builder.add_node("credit_info_node", credit_info_node)
    builder.add_node("pull_credit_node", pull_credit_node)
    builder.add_node("affordability_amount_node", affordability_amount_node)

    # Set entry point - start with information gathering
    builder.set_entry_point("info_gathering_node")

    # Compile the graph with persistence and any extra kwargs
    mortgage_workflow = builder.compile(checkpointer=checkpointer, **kwargs)

    return mortgage_workflow


def make_graph(config: RunnableConfig = None) -> CompiledStateGraph:
    """
    Creates a compiled mortgage application workflow graph.

    Args:
        config: Configuration object containing user_id and checkpoint settings

    Returns:
        A compiled workflow graph for mortgage applications
    """

    # Create a config if none is provided
    if not config:
        config = create_config()

    model = load_chat_model(MODEL)
    checkpointer = MemorySaver()

    # Initialize the mortgage workflow with model, tools, and checkpointer
    mortgage_graph = build_mortgage_workflow(
        model=model, tools=get_cached_tools(), checkpointer=checkpointer, debug=True
    )

    return mortgage_graph
