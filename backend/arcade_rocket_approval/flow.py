from typing import Any, Dict, List, Literal

from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# Import the tools from purchase.py
from arcade_rocket_approval.tools.purchase import (
    do_soft_credit_pull,
    get_purchase_application_status,
    set_buying_plans,
    set_contact_info,
    set_funds,
    set_home_details,
    set_home_price,
    set_income,
    set_military_status,
    set_personal_info,
    start_application,
)
from arcade_rocket_approval.utils import load_chat_model

################################################################################
# 1. Define your shared (global) model for all agent calls
################################################################################

model = load_chat_model("openai/o3-mini")

################################################################################
# 2. Define your shared state schema
################################################################################

# We'll store the user's entire state, or a subset, in a dictionary.
# This includes the "context" from schema.json plus additional fields
# that we want to track. For example, picked route, missing fields, etc.

# For brevity, we'll just do a simple dictionary with a "messages" key
# for the conversation history, and a "nav" key for route state, etc.

# Often you'd define a Pydantic model for strongly-typed state; here's a quick example:


class MortgageState(BaseModel):
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    rmLoanId: str = ""
    # You can copy in the entire context structure from schema.json if you prefer:
    # context: Dict[str, Any] = Field(default_factory=dict)
    # ...
    # For demonstration, we keep it minimal.


################################################################################
# 3. Define route node functions (tasks)
################################################################################


@task
def purchase_welcome_node(
    state: MortgageState,
) -> Command[Literal["buying_plans_node", "__end__"]]:
    """
    This node corresponds to 'purchase/welcome'
    We'll check if we have an rmLoanId; if not, we might call start_application.
    Then we either move on to the next route or end.
    """
    # Prompt for missing data if needed, or greet the user.
    # The simplest example: we'll check if user wants to start an application.

    # Insert a SystemMessage or something to the conversation:
    sys_msg = SystemMessage(
        content="You're Rocket Mortgage bot. Welcome the user and ask if they want to begin the application."
    )
    state.messages.append({"role": "system", "content": sys_msg.content})

    # Interrupt to get user input
    user_input = interrupt("Ready to start your Purchase application? (yes/no)")
    state.messages.append({"role": "user", "content": user_input})

    # For demonstration, we assume user said "Yes," so we call start_application
    # A more robust approach would parse user_input to see if they said yes/no
    rm_loan_id = start_application()

    if rm_loan_id:
        state.rmLoanId = rm_loan_id
        state.messages.append(
            {
                "role": "tool",
                "content": f"rmLoanId created: {rm_loan_id}",
                "tool_call_id": "start_application_call",
            }
        )

    # Next step in the nav is usually "purchase/home-info/buying-plans"
    return Command(goto="buying_plans_node", update={})


@task
def buying_plans_node(
    state: MortgageState,
) -> Command[Literal["home_details_node", "__end__"]]:
    """
    This node corresponds to 'purchase/home-info/buying-plans'.
    Suppose the user needs to confirm some fields (e.g. rmLoanId).
    Then we proceed to 'purchase/home-info/buying-plans/home-details'
    """
    # Ask user more questions, gather data, etc.
    user_input = interrupt(
        "What is your plan for buying a home? Or timeframe to purchase?"
    )
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_buying_plans(rm_loan_id=state.rmLoanId, details=user_input)

    # We store it in the state. In real code, you'd parse user_input for relevant data.
    # Next route in the nav is 'purchase/home-info/buying-plans/home-details'
    return Command(goto="home_details_node", update={})


@task
def home_details_node(
    state: MortgageState,
) -> Command[Literal["home_price_node", "__end__"]]:
    """
    This node corresponds to 'purchase/home-info/buying-plans/home-details'.
    For example, ask for home value, monthly payment, etc.
    """
    user_input = interrupt(
        "Share your estimated home value and desired monthly payment."
    )
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_home_details(rm_loan_id=state.rmLoanId, details=user_input)

    # Real logic would parse the user_input and store in the state
    # Next route is 'purchase/home-info/buying-plans/home-price'
    return Command(goto="home_price_node")


@task
def home_price_node(
    state: MortgageState,
) -> Command[Literal["personal_info_node", "__end__"]]:
    """
    This node corresponds to 'purchase/home-info/buying-plans/home-price'.
    Ask for a bit more info about the purchase price if needed.
    """
    user_input = interrupt("Do you know your target home price?")
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_home_price(rm_loan_id=state.rmLoanId, price_info=user_input)

    # Next big section is "purchase/personal-info" (the interstitial)
    return Command(goto="personal_info_node")


@task
def personal_info_node(
    state: MortgageState,
) -> Command[Literal["contact_info_node", "__end__"]]:
    """
    This would correspond to 'purchase/personal-info' before we jump to contact info routes, etc.
    """
    user_input = interrupt(
        "Next, let's handle personal info. Enter marital status or any relevant info."
    )
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_personal_info(rm_loan_id=state.rmLoanId, info=user_input)

    return Command(goto="contact_info_node")


@task
def contact_info_node(
    state: MortgageState,
) -> Command[Literal["military_status_node", "__end__"]]:
    """
    'purchase/personal-info/contact-info'
    Ask user for phone, email, etc.
    """
    user_input = interrupt("Please share your email and phone number, comma separated.")
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and "," in user_input:
        parts = user_input.split(",", 1)
        email = parts[0].strip()
        phone = parts[1].strip()
        set_contact_info(rm_loan_id=state.rmLoanId, email=email, phone=phone)

    # Next route might be 'purchase/personal-info/military-status'
    return Command(goto="military_status_node")


@task
def military_status_node(
    state: MortgageState,
) -> Command[Literal["finances_node", "__end__"]]:
    """
    'purchase/personal-info/military-status'
    Possibly ask if user is military, etc.
    Then we proceed to 'finances'...
    """
    user_input = interrupt(
        "Are you or a co-borrower associated with the Military? Yes or No?"
    )
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_military_status(rm_loan_id=state.rmLoanId, status=user_input)

    # Next route is typically 'purchase/finances'
    return Command(goto="finances_node")


@task
def finances_node(state: MortgageState) -> Command[Literal["income_node", "__end__"]]:
    """
    'purchase/finances' as an interstitial or direct route
    """
    user_input = interrupt("Let's talk about finances. Press enter to continue.")
    state.messages.append({"role": "user", "content": user_input})

    return Command(goto="income_node")


@task
def income_node(state: MortgageState) -> Command[Literal["funds_node", "__end__"]]:
    """
    'purchase/finances/income'
    """
    user_input = interrupt("What is your annual income?")
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_income(rm_loan_id=state.rmLoanId, annual_income=user_input)

    return Command(goto="funds_node")


@task
def funds_node(state: MortgageState) -> Command[Literal["credit_info_node", "__end__"]]:
    """
    'purchase/finances/funds'
    """
    user_input = interrupt("How will you fund your down payment? (savings, gift, etc.)")
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and user_input.strip():
        set_funds(rm_loan_id=state.rmLoanId, downpayment_funds=user_input)

    # Move on to credit info
    return Command(goto="credit_info_node")


@task
def credit_info_node(
    state: MortgageState,
) -> Command[Literal["pull_credit_node", "__end__"]]:
    """
    'purchase/credit-info'
    """
    user_input = interrupt(
        "We'll talk about your credit info next. Press enter to continue."
    )
    state.messages.append({"role": "user", "content": user_input})

    return Command(goto="pull_credit_node")


@task
def pull_credit_node(
    state: MortgageState,
) -> Command[Literal["affordability_amount_node", "__end__"]]:
    """
    'purchase/credit-info/birthdate-SSN'
    Ask for minimal data to do a soft credit pull, if needed.
    """
    user_input = interrupt("Enter your birthdate and last 4 of SSN, comma separated.")
    state.messages.append({"role": "user", "content": user_input})

    if state.rmLoanId and "," in user_input:
        parts = user_input.split(",", 1)
        birthdate = parts[0].strip()
        ssn_last4 = parts[1].strip()
        do_soft_credit_pull(
            rm_loan_id=state.rmLoanId, birthdate=birthdate, ssn_last4=ssn_last4
        )

    if state.rmLoanId:
        status = get_purchase_application_status(rm_loan_id=state.rmLoanId)
        state.messages.append({"role": "tool", "content": status})

    # Transition to final or to "purchase/chat-with-expert/affordability-amount"
    return Command(goto="affordability_amount_node")


@task
def affordability_amount_node(state: MortgageState) -> Command[Literal["__end__"]]:
    """
    'purchase/chat-with-expert/affordability-amount'
    Possibly handle final route or chat with expert.
    """
    user_input = interrupt(
        "Let's see your affordability range. Press enter to finalize."
    )
    state.messages.append({"role": "user", "content": user_input})

    # We can end the flow or loop back. For now, let's just end.
    return Command(goto="__end__")


################################################################################
# 4. Build the entire Workflow
################################################################################

builder = StateGraph(MortgageState)

builder.add_node("purchase_welcome_node", purchase_welcome_node)
builder.add_node("buying_plans_node", buying_plans_node)
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

# We'll define the starting node as "purchase_welcome_node".
builder.set_entry_point("purchase_welcome_node")

# Now link the edges so the graph knows how to transition:
builder.add_edge("purchase_welcome_node", "buying_plans_node")
builder.add_edge("buying_plans_node", "home_details_node")
builder.add_edge("home_details_node", "home_price_node")
builder.add_edge("home_price_node", "personal_info_node")
builder.add_edge("personal_info_node", "contact_info_node")
builder.add_edge("contact_info_node", "military_status_node")
builder.add_edge("military_status_node", "finances_node")
builder.add_edge("finances_node", "income_node")
builder.add_edge("income_node", "funds_node")
builder.add_edge("funds_node", "credit_info_node")
builder.add_edge("credit_info_node", "pull_credit_node")
builder.add_edge("pull_credit_node", "affordability_amount_node")

# __end__ will be the final terminal node (we do not define a node for it, so it auto-happens)
subgraph = builder.compile(checkpointer=MemorySaver())
