import asyncio
import os
import uuid

from dotenv import load_dotenv
from langchain.output_parsers import BooleanOutputParser
from langchain.prompts import PromptTemplate
from langchain_arcade import ToolManager
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent
from pydantic import Field

# Import your subgraph and MortgageState from flow.py
from arcade_rocket_approval.flow import MortgageState, subgraph
from arcade_rocket_approval.utils import load_chat_model

load_dotenv()

arcade_api_key = os.environ.get("ARCADE_API_KEY")
arcade_base_url = os.environ.get("ARCADE_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

CACHED_TOOLS = {}

all_tools = [
    "RocketApproval.RetrieveUserInformationFromGoogle",
]

checkpointer = MemorySaver()


class Config(RunnableConfig):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


###############################################################################
# 1) Define a "FlowNode" that runs the subgraph
###############################################################################
def run_purchase_flow(state: MortgageState):
    """
    Runs the compiled subgraph from `purchase_flow` for the user's current step.
    The subgraph returns the updated state, which we then pass back.
    """
    updated_state = subgraph.invoke(input=state)  # runs one node or until interrupt
    return updated_state.dict()


###############################################################################
# 2) Use an LLM to decide whether to run the "flow_node"
###############################################################################
def make_decision_chain(model: BaseChatModel):
    """
    Returns a small chain that interprets whether the user wants to submit a mortgage
    application. You can customize the prompt as needed.
    """
    prompt = PromptTemplate.from_template(
        """
        Here's the context of a conversation between a user and a mortgage assistant:

        {context}

        Now, the user has sent the following message:

        {message}

        Answer 'yes' if they want to submit a mortgage application, or 'no' if they do not.
        """
    )
    return prompt | model | BooleanOutputParser()


###############################################################################
# 3) Build the main React agent with Tools, add a conditional edge
###############################################################################
def make_graph(config: Config) -> CompiledStateGraph:
    global CACHED_TOOLS

    # set a user_id if it's not already set
    config["configurable"] = config.get("configurable", {})
    config["configurable"]["user_id"] = config["configurable"].get(
        "user_id", str(uuid.uuid4())
    )

    manager = ToolManager(api_key=arcade_api_key, base_url=arcade_base_url)

    if not CACHED_TOOLS:
        manager.init_tools(tools=all_tools)
        CACHED_TOOLS = manager.to_langchain()

    model = load_chat_model("openai/o3-mini")

    system_prompt = (
        "You are a helpful assistant that can handle rocket mortgage application logic. "
        ""
    )

    react_agent_node = create_react_agent(
        model=model,
        tools=ToolNode(tools=CACHED_TOOLS),
        prompt=system_prompt,
        checkpointer=checkpointer,
    )

    # Create our "decision chain" to interpret user's message
    decision_chain = make_decision_chain(model)

    # A small function that calls the chain, then returns "go" or "end"
    def decide_application_subgraph(state: MortgageState) -> str:
        """
        Decide whether to run the purchase_flow subgraph by calling a short chain to see
        if user wants to submit a mortgage application.
        """
        context = "\n".join([f"message: {m}" for m in state["messages"]])
        message = state["messages"][-1]

        user_wants_to_submit = decision_chain.invoke(
            {"context": context, "message": message}
        )
        return "apply" if user_wants_to_submit else "end"

    graph = StateGraph(MortgageState, config_schema=Config)

    # Add nodes
    graph.add_node("react_agent", react_agent_node)
    graph.add_node("application_subgraph", subgraph)

    # Entry point
    graph.set_entry_point("react_agent")

    # Use a conditional edge to decide whether to run flow_node
    graph.add_conditional_edges(
        "react_agent",
        decide_application_subgraph,  # returns "apply" or "end" after consulting the model
        {
            "apply": "application_subgraph",
            "end": END,
        },
    )

    # After application_subgraph, end the graph
    graph.add_edge("application_subgraph", END)

    compiled_graph = graph.compile(debug=True, checkpointer=checkpointer)
    compiled_graph.name = "Rocket Mortgage Agent"
    return compiled_graph
