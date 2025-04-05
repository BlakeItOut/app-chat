import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_arcade import ToolManager
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

arcade_api_key = os.environ.get("ARCADE_API_KEY")
arcade_base_url = os.environ.get("ARCADE_BASE_URL")
openai_api_key = os.environ.get("OPENAI_API_KEY")


INFO_MODEL = "openai/gpt-4o"
MODEL = "openai/o3-mini"
TOOLKIT = "RocketApproval"
TOOLS = [
    "RocketApproval.RetrieveUserInformationFromGoogle",
]
CHECKPOINTER = MemorySaver()


def get_cached_tools(tools: list[str] = TOOLS) -> list[BaseTool]:
    tools_manager = ToolManager(api_key=arcade_api_key, base_url=arcade_base_url)
    tools_manager.init_tools(tools=tools, limit=100)
    return tools_manager.to_langchain()


@lru_cache(maxsize=12)
def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
