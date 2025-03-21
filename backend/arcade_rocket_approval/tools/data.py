from typing import Annotated

from arcade.sdk import ToolContext, tool

from arcade_rocket_approval.api import ContactInfo


def extract_contact_record_to_rocket_user_context(
    context: ToolContext,
    contact_info: Annotated[ContactInfo, "Contact information for the user"],
) -> ContactInfo:
    """
    Extract contact information from the user's Google People Record.
    """
    return contact_info
