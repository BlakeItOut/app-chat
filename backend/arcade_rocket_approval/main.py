import asyncio

from arcade_rocket_approval.flow import MortgageState
from arcade_rocket_approval.graph import make_graph


async def run_entire_logic():
    # Build the compiled graph
    graph = await make_graph(config=None)

    # Create an initial state
    state = MortgageState()

    # Now invoke the graph, e.g. for the first step:
    final = graph.invoke(input=state)
    print("DONE. Final State:", final)
