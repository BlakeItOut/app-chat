"""
This module provides a tool to retrieve all available information about the
authenticated user from the Google People API by specifying every possible field.

See: https://developers.google.com/people/api/rest/v1/people/get
     https://developers.google.com/people/api/rest/v1/people#Person
"""

import asyncio
from typing import Annotated, Optional

from arcade.sdk import ToolContext, tool
from arcade.sdk.auth import Google
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

# A comma-separated list of all valid person fields to fetch
ALL_PERSON_FIELDS = (
    "addresses,"
    "ageRanges,"
    "biographies,"
    "birthdays,"
    "calendarUrls,"
    "clientData,"
    "coverPhotos,"
    "emailAddresses,"
    "events,"
    "externalIds,"
    "genders,"
    "imClients,"
    "interests,"
    "locales,"
    "locations,"
    "memberships,"
    "metadata,"
    "miscKeywords,"
    "names,"
    "nicknames,"
    "occupations,"
    "organizations,"
    "phoneNumbers,"
    "photos,"
    "relations,"
    "sipAddresses,"
    "skills,"
    "urls,"
    "userDefined"
)


def build_people_service(token: str):
    """
    Build and return a 'people' service object with the provided token.
    """
    credentials = Credentials(token=token)
    return build("people", "v1", credentials=credentials)


@tool(
    requires_auth=Google(
        # Any of the listed scopes grants permission to retrieve personal user data;
        # include at least one that suits your usage scenario.
        scopes=["https://www.googleapis.com/auth/contacts.readonly"]
    )
)
async def retrieve_user_information_from_google(
    context: ToolContext,
) -> Annotated[dict, "Contains all available fields for the authenticated user"]:
    """
    Retrieve information for the authenticated user from the Google People API.
    Returns all possible fields in the Person object.
    """
    # Build the People API service.
    service = build_people_service(
        context.authorization.token
        if context.authorization and context.authorization.token
        else ""
    )

    # Retrieve all available fields from 'people/me'.
    response = (
        service.people()
        .get(resourceName="people/me", personFields=ALL_PERSON_FIELDS)
        .execute()
    )

    # Return a dictionary that includes the user's full Person resource.
    return {"person": response}
