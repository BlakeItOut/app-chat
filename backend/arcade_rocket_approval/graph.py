import asyncio
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_arcade import ToolManager
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, create_react_agent

# Import your subgraph and MortgageState from flow.py
from arcade_rocket_approval.flow import MortgageState, subgraph
from arcade_rocket_approval.utils import load_chat_model

load_dotenv()

arcade_api_key = os.environ.get("ARCADE_API_KEY")
arcade_base_url = os.environ.get("ARCADE_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")

cached_tools = {}


###############################################################################
# 1) Define a "FlowNode" that runs the subgraph
###############################################################################
def run_purchase_flow(state: MortgageState):
    """
    Runs the compiled subgraph from `purchase_flow` for the user's current step.
    Because the subgraph returns the updated state, we then pass that back.
    """
    # We can call `purchase_flow.invoke(...)` to run from the current node to next.
    # Or we can do it in a loop if we want multiple steps. For simplicity,
    # let's do a single step at a time. Then we return the updated state.

    updated_state = subgraph.invoke(input=state)  # runs one node or until interrupt
    return updated_state.dict()


###############################################################################
# 2) Create the main React agent with your Tools
###############################################################################
def make_graph(config: RunnableConfig) -> CompiledStateGraph:
    """
    Create a custom state graph for the Reasoning and Action agent, but also
    incorporate the purchase_flow subgraph.
    """
    global cached_tools

    manager = ToolManager(
        api_key=arcade_api_key,
        base_url=arcade_base_url,
    )

    # If you want your ReAct agent to call these tools spontaneously:
    # (Adjust the list as desired.)
    all_tools = [
        "RocketApproval.StartApplication",
        "RocketApproval.GetPurchaseApplicationStatus",
        "RocketApproval.SetBuyingPlans",
        "RocketApproval.SetHomeDetails",
        "RocketApproval.SetHomePrice",
        "RocketApproval.SetPersonalInfo",
        "RocketApproval.SetContactInfo",
        "RocketApproval.SetMilitaryStatus",
        "RocketApproval.SetIncome",
        "RocketApproval.SetFunds",
        "RocketApproval.DoSoftCreditPull",
    ]
    if not cached_tools:
        manager.init_tools(tools=all_tools)
        cached_tools = manager.to_langchain()

    # Load a chat model for the agent
    model = load_chat_model("openai/o3-mini")

    # Format the system prompt for the ReAct agent
    system_prompt = (
        "You are a helpful assistant that can handle rocket mortgage application logic. "
        "Use the subgraph to walk the user through the route flow or call these tools directly if needed."
    )

    # Create the ReAct agent node
    react_agent_node = create_react_agent(
        model=model,
        tools=ToolNode(tools=cached_tools),
        prompt=system_prompt,
    )

    ###############################################################################
    # 3) Build the combined graph
    ###############################################################################
    graph = StateGraph(MortgageState)

    # Add the agent node
    graph.add_node("react_agent", react_agent_node)
    graph.add_node("flow_node", subgraph)

    # Set the entry point to the flow or to the agent - up to you.
    # Let's start by going directly to the subgraph.
    graph.set_entry_point("react_agent")

    # Add the edges
    graph.add_edge("react_agent", "flow_node")
    graph.add_edge("flow_node", END)

    # Compile it
    compiled_graph = graph.compile()

    compiled_graph.name = "Rocket Mortgage Agent"
    return compiled_graph


graph = make_graph(config=None)
