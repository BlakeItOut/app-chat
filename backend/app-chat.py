import os
 
arcade_api_key = os.environ.get("ARCADE_API_KEY")
openai_api_key = os.environ.get("OPENAI_API_KEY")

from langchain_arcade import ArcadeToolManager
 
manager = ArcadeToolManager(api_key=arcade_api_key)
 
# Fetch the "ScrapeUrl" tool from the "Web" toolkit
tools = manager.get_tools(tools=["Web.ScrapeUrl"])
print(manager.tools)
 
# Get all tools from the "Google" toolkit
tools = manager.get_tools(toolkits=["Google"])
print(manager.tools)

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
 
model = ChatOpenAI(model="gpt-4o", api_key=openai_api_key)
bound_model = model.bind_tools(tools)
 
memory = MemorySaver()

from langgraph.prebuilt import create_react_agent
 
graph = create_react_agent(model=bound_model, tools=tools, checkpointer=memory)

config = {
    "configurable": {
        "thread_id": "1",
        "user_id": "blake.m.shaw@gmail.com"
    }
}
user_input = {
    "messages": [
        ("user", "List any new and important emails in my inbox.")
    ]
}

from langgraph.errors import NodeInterrupt
 
try:
    for chunk in graph.stream(user_input, config, stream_mode="values"):
        chunk["messages"][-1].pretty_print()
except NodeInterrupt as exc:
    print(f"\nNodeInterrupt occurred: {exc}")
    print("Please authorize the tool or update the request, then re-run.")