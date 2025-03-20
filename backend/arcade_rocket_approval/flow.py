from typing import Any, Dict, List, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import Command, interrupt

# Import the tools from purchase.py
from arcade_rocket_approval.tools.purchase import (
    do_soft_credit_pull,
    get_purchase_application_status,
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


class MortgageState(MessagesState):
    rmLoanId: str | None = None



def purchase_welcome_node(
    state: MortgageState,
) -> Command[Literal["home_details_node", "__end__"]]:
    """
    This node corresponds to 'purchase/welcome'.
    We'll check if we have an rmLoanId; if not, we might call start_application
    after parsing the user input for yes/no.
    Then we either move on to the next route or end.
    """
    # Prompt user
    user_prompt = "Ready to start your Purchase application? (yes/no)"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    # A robust check for yes/no
    user_reply_cleaned = user_input["value"].strip().lower()
    if user_reply_cleaned.startswith("y"):
        rm_loan_id = start_application()
        # Next step in the nav is usually "purchase/home-info/buying-plans"
        return Command(
            goto="home_details_node",
            update={"rmLoanId": rm_loan_id, "messages": messages},
        )
    else:
        # If user said "no", we might just end or do something else
        new_messages = [
            HumanMessage(content=user_reply_cleaned),
            AIMessage(content="Understood. We won't start the application now."),
        ]
        return Command(goto="__end__", update={"messages": new_messages})


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

    if state["rmLoanId"] and user_input.strip():
        status = set_home_details(rm_loan_id=state["rmLoanId"], details=user_input)

    # Real logic would parse the user_input and store in the state
    # Next route is 'purchase/home-info/buying-plans/home-price'
    return Command(goto="home_price_node", update={"messages": messages})


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

    if state["rmLoanId"] and user_input.strip():
        status = set_home_price(rm_loan_id=state["rmLoanId"], price_info=user_input)

    # Next big section is "purchase/personal-info" (the interstitial)
    return Command(goto="personal_info_node", update={"messages": messages})



def personal_info_node(
    state: MortgageState,
) -> Command[Literal["contact_info_node", "__end__"]]:
    """
    This would correspond to 'purchase/personal-info' before we jump to contact info routes, etc.
    """
    user_prompt = (
        "Next, let's handle personal info. Enter marital status or any relevant info."
    )
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    if state["rmLoanId"] and user_input.strip():
        status = set_personal_info(rm_loan_id=state["rmLoanId"], info=user_input)

    return Command(goto="contact_info_node", update={"messages": messages})



def contact_info_node(
    state: MortgageState,
) -> Command[Literal["military_status_node", "__end__"]]:
    """
    'purchase/personal-info/contact-info'
    Ask user for phone, email, etc.
    """
    user_prompt = "Please share your email and phone number, comma separated."
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    if state["rmLoanId"] and "," in user_input:
        parts = user_input.split(",", 1)
        email = parts[0].strip()
        phone = parts[1].strip()
        status = set_contact_info(
            rm_loan_id=state["rmLoanId"], email=email, phone=phone
        )

    # Next route might be 'purchase/personal-info/military-status'
    return Command(goto="military_status_node", update={"messages": messages})


def military_status_node(
    state: MortgageState,
) -> Command[Literal["finances_node", "__end__"]]:
    """
    'purchase/personal-info/military-status'
    Possibly ask if user is military, etc.
    Then we proceed to 'finances'...
    """
    user_prompt = "Are you or a co-borrower associated with the Military? Yes or No?"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    if state["rmLoanId"] and user_input.strip():
        status = set_military_status(rm_loan_id=state["rmLoanId"], status=user_input)

    # Next route is typically 'purchase/finances'
    return Command(goto="finances_node", update={"messages": messages})


def finances_node(state: MortgageState) -> Command[Literal["income_node", "__end__"]]:
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



def income_node(state: MortgageState) -> Command[Literal["funds_node", "__end__"]]:
    """
    'purchase/finances/income'
    """
    user_prompt = "What is your annual income?"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    if state["rmLoanId"] and user_input.strip():
        status = set_income(rm_loan_id=state["rmLoanId"], annual_income=user_input)

    return Command(goto="funds_node", update={"messages": messages})



def funds_node(state: MortgageState) -> Command[Literal["credit_info_node", "__end__"]]:
    """
    'purchase/finances/funds'
    """
    user_prompt = "How will you fund your down payment? (savings, gift, etc.)"
    user_input = interrupt(user_prompt)
    messages = [
        AIMessage(content=user_prompt),
        HumanMessage(content=user_input["value"]),
    ]

    if state["rmLoanId"] and user_input.strip():
        status = set_funds(rm_loan_id=state["rmLoanId"], downpayment_funds=user_input)

    # Move on to credit info
    return Command(goto="credit_info_node", update={"messages": messages})


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

    if state["rmLoanId"] and "," in user_input:
        parts = user_input.split(",", 1)
        birthdate = parts[0].strip()
        ssn_last4 = parts[1].strip()
        status = do_soft_credit_pull(
            rm_loan_id=state["rmLoanId"], birthdate=birthdate, ssn_last4=ssn_last4
        )

    if state["rmLoanId"]:
        status = get_purchase_application_status(rm_loan_id=state["rmLoanId"])

    # Transition to final or to "purchase/chat-with-expert/affordability-amount"
    return Command(goto="affordability_amount_node", update={"messages": messages})


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
    ]

    # We can end the flow or loop back. For now, let's just end.
    return Command(goto="__end__", update={"messages": messages})


################################################################################
# 4. Build the entire Workflow
################################################################################

builder = StateGraph(MortgageState)

builder.add_node("purchase_welcome_node", purchase_welcome_node)
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
builder.add_edge("purchase_welcome_node", "home_details_node")
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

subgraph = builder.compile(checkpointer=MemorySaver())
