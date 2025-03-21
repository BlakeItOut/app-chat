import logging
from typing import Any, Generic, TypeVar

from httpx import Client, HTTPStatusError
from pydantic import BaseModel

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



T = TypeVar("T")


class Response(BaseModel, Generic[T]):
    """Standard response format for all API operations"""

    status: str
    """Status of the operation, either 'success' or 'error'"""

    message: str
    """Human-readable message about the operation"""

    data: T | None = None
    """Optional data returned from the operation"""

    raw_response: dict[str, Any] | None = None
    """Raw JSON response from the API call"""

    @classmethod
    def success(
        cls,
        message: str,
        data: T | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> "Response[T]":
        """Create a success response"""
        return cls(
            status="success", message=message, data=data, raw_response=raw_response
        )

    @classmethod
    def error(
        cls,
        message: str,
        data: T | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> "Response[T]":
        """Create an error response"""
        return cls(
            status="error", message=message, data=data, raw_response=raw_response
        )


# Helper function for handling common request exceptions
def handle_request_exception(e: Exception) -> Response[None]:
    """Create error response from exception"""
    return Response.error(f"Request failed: {str(e)}")
