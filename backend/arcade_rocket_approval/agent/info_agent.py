from typing import Any, Dict, List, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from arcade_rocket_approval.api import RocketUserContext

DISCOVERY_TURNS = 7


class MortgageInfoState(MessagesState):
    user_info: RocketUserContext | None = None


def get_user_info_agent(
    model: BaseChatModel, tools: List[BaseTool], checkpointer: MemorySaver
):
    """
    A react agent that returns a structured output of the information
    about the user that we need to submit a mortgage application.

    This agent has a series of tools that it can use to get the information it needs
    but it can also use the user's message to infer the information it needs.
    """
    # Create a comprehensive system prompt that guides the model
    system_prompt = PromptTemplate.from_template(
        """
        You are a helpful assistant that helps gather information about a user
        that we need to submit a mortgage application.

        IMPORTANT: Always start the conversation by asking if the user would like to use external services
        to speed up the onboarding process. This should be your very first question.

        Your goal is to collect all necessary information and return it in a structured format.
        Ask clarifying questions one at a time. Use any available tools to help gather information.

        The required information includes:
        - Full name (first and last name)
        - Email address
        - Phone number
        - Current address
        - Current living situation (renting or owning)
        - Annual income
        - Military status
        - Marital status
        - Funds available for down payment

        When you have all the required information, provide a complete summary with all collected details.

        You have the following tools available to you:
        {tools}

        when you've used all the tools, you can return the structured output and end the conversation by
        saying "all required information has been collected" or "summary" and a summary of the information.
        """
    ).partial(tools=tools)

    # Force the model to use tools by passing tool_choice="any"
    tool_model = model.bind_tools(tools, tool_choice="any", parallel_tool_calls=False)
    model_with_tools = system_prompt | tool_model

    # summarizer
    summarizer_prompt = PromptTemplate.from_template(
        """
        Based on the collected user information, summarize it in a concise manner.
        The context below will be a ToolMessage with some kind of information about the
        user. Provide a summary of the information in the context keeping in mind
        the fields needed for a RocketUserContext.

        Context:
        {context}

        ALWAYS start your response with "Summary: "

        Summary:
        """
    )
    summarizer_model = summarizer_prompt | model
    # Create a structured output model for the final response
    structured_output_prompt = PromptTemplate.from_template(
        """
        Based on the collected user information, format it as a structured response
        following the RocketUserContext format.

        Return ONLY the structured data with no additional text.
        """
    )
    model_with_structured_output = model.with_structured_output(RocketUserContext)
    structured_output_model = structured_output_prompt | model_with_structured_output

    def gather_info_node(
        state: MortgageInfoState,
        config: RunnableConfig,
    ) -> Command[Literal["process_info_node"]]:
        """
        This node calls the model to generate a response based on the conversation so far.
        If the model calls a tool, we go to the tools node. Otherwise, we go to process_info_node.
        """
        messages = state.get("messages", [])

        # Check for conversation length to prevent infinite loops
        # If we've gone back and forth many times, we likely have enough information
        if len(messages) > DISCOVERY_TURNS:
            return Command(
                goto="process_info_node",
                update={"messages": messages},
            )

        # Call the standard model to generate conversational response
        response = model_with_tools.invoke({"messages": messages}, config=config)

        # Try to extract structured information to check completeness
        try:
            print(
                extract_tool_message_contents(messages + [response]),
                "--------------------------------",
            )
            summary = summarizer_model.invoke(
                {"context": extract_tool_message_contents(messages + [response])},
                config=config,
            )

            # If we have all required fields or the model indicates completion, proceed
            if summary.content.strip().lower().startswith("summary"):
                return Command(
                    goto="process_info_node",
                    update={"messages": summary},
                )
        except Exception as e:
            # If we can't extract structured data yet, that's okay
            print(f"Info extraction not yet complete: {e}")

        # Continue the conversation - the user will need to respond
        return Command(
            goto="tools",
            update={"messages": response},
        )

    def process_info_node(
        state: MortgageInfoState,
        config: RunnableConfig,
    ) -> Command[Literal["__end__"]]:
        """
        Process all information gathered so far and create a structured output.
        """
        messages = state.get("messages", [])

        # Generate structured output from conversation
        user_info = structured_output_model.invoke(
            {"messages": messages[0].content}, config=config
        )
        print(user_info, "--------------------------------")
        return Command(
            goto="__end__",
            update={"user_info": user_info, "messages": messages},
        )

    # Define the graph workflow
    workflow = StateGraph(MortgageInfoState)

    # Add nodes
    workflow.add_node("gather_info_node", gather_info_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("process_info_node", process_info_node)

    # Set entry point
    workflow.set_entry_point("gather_info_node")

    # Compile the graph
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
