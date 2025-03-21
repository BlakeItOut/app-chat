"""
This module provides a tool to retrieve all available information about the
authenticated user from the Google People API by specifying every possible field.

See: https://developers.google.com/people/api/rest/v1/people/get
     https://developers.google.com/people/api/rest/v1/people#Person
"""

import json
import logging
from json.decoder import JSONDecodeError
from typing import Annotated

from arcade.sdk import ToolContext, tool
from arcade.sdk.auth import Google
from arcade.sdk.errors import ToolExecutionError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from arcade_rocket_approval.api import (
    Address,
    ContactInfo,
    PersonalInfo,
    PhoneNumber,
    RocketUserContext,
)

logger = logging.getLogger(__name__)

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

    try:
        return extract_rocket_user_context_from_google_api(response)
    except JSONDecodeError as e:
        logger.exception("Failed to parse Google API JSON data")
        raise ToolExecutionError("Failed to parse Google API JSON data")
    except Exception as e:
        logger.exception("Failed to return RocketUserContext from Google API")
        raise ToolExecutionError("Failed to return RocketUserContext from Google API")


def extract_rocket_user_context_from_google_api(json_data: str) -> str:
    """
    Parse Google API JSON response and extract fields relevant to RocketUserContext.

    Args:
        json_data: A string containing JSON data from Google People API

    Returns:
        A JSON string containing a partially populated RocketUserContext object
    """
    # Parse the JSON data
    try:
        data = json.loads(json_data)
        person_data = data.get("person", {})

        # Extract name information
        names = person_data.get("names", [])
        first_name = None
        last_name = None
        if names:
            first_name = names[0].get("givenName")
            last_name = names[0].get("familyName")

        # Extract email information
        emails = person_data.get("emailAddresses", [])
        email = emails[0].get("value") if emails else None

        # Extract and parse phone number
        phones = person_data.get("phoneNumbers", [])
        phone_number = None
        if phones:
            phone_str = phones[0].get("value", "")
            # Parse phone number format like "(804) 840-4783"
            import re

            phone_match = re.search(r"\((\d{3})\)\s*(\d{3})-(\d{4})", phone_str)
            if phone_match:
                area_code, prefix, line = phone_match.groups()
                phone_number = PhoneNumber(
                    area_code=area_code, prefix=prefix, line=line
                )

        # Create PersonalInfo
        personal_info = PersonalInfo(
            first_name=first_name or "",
            last_name=last_name or "",
            date_of_birth="",  # Not available in the data
            marital_status="Single",  # Default value
            is_spouse_on_loan=False,  # Default value
        )

        # Create ContactInfo
        contact_info = ContactInfo(
            first_name=first_name or "",
            last_name=last_name or "",
            date_of_birth=None,
            email=email or "",
            phone_number=phone_number or PhoneNumber(area_code="", prefix="", line=""),
            has_promotional_sms_consent=False,  # Default value
        )

        address = Address(
            street=person_data.get("addresses", [{}])[0].get("streetAddress", ""),
            city=person_data.get("addresses", [{}])[0].get("city", ""),
            state=person_data.get("addresses", [{}])[0].get("state", ""),
            zip_code=person_data.get("addresses", [{}])[0].get("postalCode", ""),
        )

        rocket_user_context = RocketUserContext(
            contact_info=contact_info,
            address=address,
            personal_info=personal_info,
        )
        return rocket_user_context.model_dump_json()
    except json.JSONDecodeError:
        # Return empty RocketUserContext if JSON parsing fails
        logger.exception("Failed to parse Google API JSON data")
        raise ToolExecutionError("Failed to parse Google API JSON data")