from typing import Annotated

from arcade.sdk import ToolContext, tool
from arcade.sdk.auth import OAuth2

from arcade_rocket_approval.api import Response


@tool(
    requires_auth=OAuth2(
        client_id="",
        client_secret="",
        redirect_uri="",
        scopes=["auth0"],
    )
)
async def login_user(context: ToolContext) -> Annotated[Response[str], "status"]:
    """
    Login to the Rocket Mortgage website.
    """
    pass


@tool(
    requires_auth=OAuth2(
        client_id="",
        client_secret="",
        redirect_uri="",
        scopes=["auth0"],
    )
)
async def get_user_info(context: ToolContext) -> Annotated[Response[str], "status"]:
    """
    Get the user's information.
    """
    pass


@tool(
    requires_auth=OAuth2(
        client_id="",
        client_secret="",
        redirect_uri="",
        scopes=["auth0"],
    )
)
async def forgot_password(context: ToolContext) -> Annotated[Response[str], "status"]:
    """
    Forgot password.
    """
    pass
