import logging
from functools import lru_cache

from httpx import Client, HTTPStatusError
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from arcade_rocket_approval.env import APPROVAL_BASE_URL

logger = logging.getLogger(__name__)


def send_request(
    url: str,
    method: str,
    data: dict | None = None,
    headers: dict | None = None,
    json: dict | None = None,
    client: Client | None = None,
    base_url: str | None = APPROVAL_BASE_URL,
) -> dict:
    """Send a request to the given URL with the given payload

    Args:
            url: The URL to send the request to
            method: The HTTP method to use
            data: The data to send to the URL
            headers: The headers to send to the URL
            json: The JSON data to send to the URL
            client: The httpx client to use to send the request

    Returns:
            The response from the URL
    """
    if not client:
        client = Client(
            base_url=url,
            headers=headers,
        )
    try:
        logger.info(f"Sending request to {url} with method {method}")

        response = client.request(method, url, data=data, json=json)
        response.raise_for_status()

        logger.info(f"Response: {response.json()}")
        logger.info(f"Response status code: {response.status_code}")
        logger.info(f"Response headers: {response.headers}")

        return response.json()

    except HTTPStatusError as e:
        if e.response.status_code == 401:
            logger.error("Unauthorized request to %s with error %s", url, e)
        elif e.response.status_code == 404:
            logger.error("Not found request to %s with error %s", url, e)
        else:
            logger.error("Error sending request to %s: %s", url, e)
        raise e


@lru_cache(maxsize=12)
def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)
