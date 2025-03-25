import json
import logging
from typing import Any, List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from arcade_rocket_approval.api import (
    ContactInfo,
    CurrentLivingSituation,
    HomePurchase,
    PersonalInfo,
    PrimaryAssets,
    RocketUserContext,
    SpouseAssets,
)
from arcade_rocket_approval.defaults import (
    INFO_MODEL,
    get_cached_tools,
    load_chat_model,
)
from arcade_rocket_approval.prompts import (
    EXTERNAL_SERVICE_PROMPT,
    QUESTION_INFO_AGENT_PROMPT,
    SUMMARIZER_PROMPT,
)

# Add logger
logger = logging.getLogger(__name__)

DISCOVERY_TURNS = 2

DISCOVERY_QUESTIONS = [
    "Tell me about yourself! What's your name? Address? The more the better",
    "What's your annual income?",
    "What's your current living situation? rent? own?",
    "Have you been in the military? If so, what branch?",
    "What's your current address?",
    "What's your phone number?",
    "What's your email?",
]


def get_question_model(model: str) -> Runnable:
    llm = load_chat_model(model)
    tools = [
        convert_to_openai_tool(
            t
            for t in [
                ContactInfo,
                PersonalInfo,
                CurrentLivingSituation,
                HomePurchase,
                PrimaryAssets,
                SpouseAssets,
            ]
        )
    ]
    tool_names = [tool.name for tool in tools]
    # Create a formatted system message first
    system_message = QUESTION_INFO_AGENT_PROMPT.partial(
        tools="\n".join(tool_names),
        questions="\n".join(DISCOVERY_QUESTIONS),
    )

    # Now use that message with the model
    return system_message | llm.bind_tools(
        tools=tools, tool_choice="any", parallel_tool_calls=False
    )


def get_external_service_model(model: str) -> Runnable:
    tools = get_cached_tools()
    llm = load_chat_model(model)
    return EXTERNAL_SERVICE_PROMPT | llm.bind_tools(
        tools=tools, tool_choice="any", parallel_tool_calls=False
    )

def get_summarizer_model(model: str, context: str) -> Runnable:
    llm = load_chat_model(model)
    if isinstance(context, list):
        context = "\n".join([str(item) for item in context])

    prompt = SUMMARIZER_PROMPT.partial(context=context)
    return prompt | llm


DISCOVERY_TURNS = 7


class MortgageInfoState(MessagesState):
    user_info: RocketUserContext | None = None
    use_external: bool | None = None


def agent_node(
    state: MortgageInfoState,
    config: RunnableConfig,
) -> Command[Literal["tools_call_node", "__end__"]]:
    """
    This node decides whether to use the external service or not.
    """

    use_external = True
    # if state.get("use_external") is None:
    #    use_external = interrupt(
    #        "Can we use an external service to help gather information? (Google)"
    #    )
    # else:
    #    use_external = state.get("use_external")

    if use_external:
        return Command(
            goto="tools_call_node",
            update={"use_external": use_external},
        )
    else:
        return Command(
            goto="__end__",
            update={"messages": state.get("messages", [])},
        )

def tool_call_node(
    state: MortgageInfoState,
    config: RunnableConfig,
) -> Command[Literal["tools"]]:
    """
    This node calls the external service.
    """
    model = get_external_service_model(config.get("model", INFO_MODEL))
    response = model.invoke({"input": state.get("messages", [])}, config=config)

    return Command(
        goto="tools",
        update={"messages": response},
    )


def tool_parse_node(
    state: MortgageInfoState,
    config: RunnableConfig,
) -> Command[Literal["should_continue"]]:
    response = state.get("messages", [])

    # look for ToolMessage
    tool_message = None
    for message in response:
        if isinstance(message, ToolMessage):
            tool_message = message
            break

    if tool_message:
        data = tool_message.content
        if data:
            try:
                data = RocketUserContext.model_validate(data, strict=False)

                return Command(
                    goto="should_continue",
                    update={"messages": response, "user_info": data},
                )
            except Exception as e:
                logger.error(f"Error validating user info: {e}")

    return Command(
        goto="should_continue",
        update={"messages": response},
    )


def gather_info_node(
    state: MortgageInfoState,
    config: RunnableConfig,
) -> Command[Literal["should_continue"]]:
    """
    This node calls the model to generate a response based on the conversation so far.
    """
    messages = state.get("messages", [])

    # Call the standard model to generate conversational response
    model = get_question_model(config.get("model", INFO_MODEL))
    response = model.invoke({"input": messages[0].content}, config=config)

    return Command(
        goto="should_continue",
        update={"messages": response},
    )


def should_continue_node(
    state: MortgageInfoState,
    config: RunnableConfig,
) -> Command[Literal["__end__", "agent_node"]]:
    messages = state.get("messages", [])

    if state.get("user_info") is None:
        return Command(
            goto="agent_node",
            update={"messages": messages},
        )

    return Command(
        goto="__end__",
        update={"messages": messages},
    )

def get_user_info_agent(
    model: BaseChatModel, tools: List[BaseTool], checkpointer
) -> Runnable:
    # Define the graph workflow
    workflow = StateGraph(MortgageInfoState)

    # Add nodes
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("gather_info_node", gather_info_node)
    workflow.add_node("should_continue", should_continue_node)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tools_call_node", tool_call_node)
    workflow.add_node("tool_parse_node", tool_parse_node)

    workflow.add_edge("tools", "tool_parse_node")
    workflow.add_edge("tool_parse_node", "should_continue")
    workflow.set_entry_point("agent_node")

    graph = workflow.compile(checkpointer=checkpointer, debug=True)

    return graph


def extract_tool_message_contents(messages: List[Any]) -> str:
    """
    Extracts content from all ToolMessages in a list of messages and
    combines them into a single string separated by double newlines.
    Args:
        messages: A list of message objects which may include ToolMessages
    Returns:
        A string containing all ToolMessage contents joined by double newlines
    """
    tool_contents = []

    for message in messages:
        if isinstance(message, ToolMessage):
            content = message.content
            # If content is not a string (e.g., it's a dict or other structure), convert it
            if not isinstance(content, str):
                content = str(content)
            tool_contents.append(content)

    # Join all tool message contents with double newlines
    return "\n\n".join(tool_contents)


def get_default_rocket_user_context() -> RocketUserContext:
    return RocketUserContext(
        personal_info=PersonalInfo(
            first_name="",
            last_name="",
            date_of_birth="",
            marital_status="Single",
            is_spouse_on_loan=False,
        ),
        contact_info=ContactInfo(
            first_name="",
            last_name="",
            date_of_birth=None,
            email="",
            phone_number=None,
            has_promotional_sms_consent=False,
        ),
        phone_number=None,
        address=None,
        living_situation=CurrentLivingSituation(
            rent_or_own="Renter",
            address=None,
        ),
        home_purchase=HomePurchase(has_budget=False),
        primary_assets=PrimaryAssets(),
        spouse_assets=SpouseAssets(),
        marital_status="Single",
        income=None,
        military_status=None,
        real_estate_agent=None,
        home_details=None,
        ideal_home_price=None,
        has_promotional_sms_consent=False,
    )
