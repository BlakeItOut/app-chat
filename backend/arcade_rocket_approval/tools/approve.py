import logging
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional

import requests
from arcade.sdk import ToolContext, tool
from arcade.sdk.annotations import Inferrable
from arcade.sdk.errors import ToolExecutionError
from httpx import Client, HTTPStatusError
from pydantic import BaseModel

from arcade_rocket_approval.env import APPROVAL_BASE_URL
from arcade_rocket_approval.utils import (
    Response,
    handle_request_exception,
    send_request,
)

logger = logging.getLogger(__name__)


class PropertyType(str, Enum):
    SINGLE_FAMILY = "single"
    MULTI_FAMILY = "multi"
    CONDO = "condo"
    TOWNHOUSE = "townhouse"


class OccupancyType(str, Enum):
    PRIMARY = "primary"
    INVESTMENT = "investment"
    VACATION = "vacation"


class MaritalStatus(str, Enum):
    MARRIED = "married"
    SINGLE = "single"


class MilitaryStatus(str, Enum):
    ACTIVE_DUTY = "currentlyServing"
    RESERVE = "reserve"
    PAST_DUTY = "dischgd"
    NONE = "none"


class MilitaryBranch(str, Enum):
    ARMY = "army"
    NAVY = "navy"
    AIR_FORCE = "airForce"
    MARINE_CORPS = "marineCorps"
    SPACE_FORCE = "spaceForce"
    COAST_GUARD = "coastGuard"
    NONE = "none"


class ServiceExpiration(BaseModel):
    day: str
    month: str
    year: str

    def to_api_format(self) -> dict[str, str]:
        return {
            "day": self.day,
            "month": self.month,
            "year": self.year,
        }


class ServiceType(str, Enum):
    REGULAR = "regularMilitary"
    RESERVE = "reserves"
    NONE = "none"


class LivingSituation(str, Enum):
    RENTER = "Renter"
    OWNER = "Homeowner"


@tool
def start_mortgage_application(
    context: ToolContext,
) -> Annotated[dict[str, str], "rm_loan_id and sessionToken"]:
    """
    Start a new mortgage application and get a session token and rmLoanId.
    """
    endpoint = APPROVAL_BASE_URL + "/api/welcome"
    headers = {"Content-Type": "application/json"}
    payload = {"loanPurpose": "Purchase"}

    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)

        session_token = response.cookies.get("sessionToken")

        if not session_token:
            raise ToolExecutionError(
                "No session token found after starting mortgage application"
            )

        data = response.json()

        rm_loan_id = data.get("context", {}).get("rmLoanId", "")

        return {
            "rmLoanId": rm_loan_id,
            "sessionToken": session_token,
        }
    except HTTPStatusError as e:
        raise ToolExecutionError(f"Error starting mortgage application: {e}") from e


@tool
def set_new_home_details(
    context: ToolContext,
    new_home_city: Annotated[str, "City of the new home"],
    new_home_state: Annotated[
        str, "State of the new home in two letter format i.e. (CA, TX, NY, etc.)"
    ],
    new_home_zip_code: Annotated[str, "Zip code of the new home"],
    new_home_occupancy_type: Annotated[
        OccupancyType, "Type of occupancy"
    ] = OccupancyType.PRIMARY,
    rm_loan_id: Annotated[
        str, "loan ID from start_mortgage_application", Inferrable(False)
    ] = None,
    session_token: Annotated[
        str, "session token from start_mortgage_application", Inferrable(False)
    ] = None,
) -> Annotated[dict[str, str], "status and message"]:
    """
    record the user's new home details weather they have found a new home or not.

    make sure to use "primary", "investment", or "vacation" for the occupancy type.
    make sure to use the two letter format for the state. ex. CA for California.
    """
    endpoint = APPROVAL_BASE_URL + "/api/home-info/buying-plans/home-details"
    headers = {"Content-Type": "application/json"}
    payload = {
        "rmLoanId": rm_loan_id,
        "buyingPlans": False,  # instructed to ignore this one
    }

    try:
        send_request(
            endpoint,
            "POST",
            json=payload,
            headers=headers,
            cookies={"sessionToken": session_token},
        )
    except HTTPStatusError as e:
        raise ToolExecutionError(f"Error setting new home details: {e}") from e


    # even if the user has not found a new home, we need to record the location
    # that they plan to live in
    data = {
        "rmLoanId": rm_loan_id,
        "location": {
            "city": new_home_city,
            "state": new_home_state,
            "zipCode": new_home_zip_code,
        },
        "propertyType": None,
        "occupancyType": new_home_occupancy_type.value,
    }

    try:
        send_request(
            endpoint,
            "POST",
            json=data,
            headers=headers,
            cookies={"sessionToken": session_token},
        )

        return {
            "status": "success",
            "message": "Home details set successfully",
        }
    except HTTPStatusError as e:
        raise ToolExecutionError(f"Error setting new home details: {e}") from e


@tool
def set_home_price(
    context: ToolContext,
    minimum_price: Annotated[int, "Minimum price of the home"] = None,
    rm_loan_id: Annotated[
        str, "loan ID from start_mortgage_application", Inferrable(False)
    ] = None,
    session_token: Annotated[
        str, "session token from start_mortgage_application", Inferrable(False)
    ] = None,
) -> Annotated[dict[str, str], "status and message"]:
    """
    Record the user's minimum price for a home that they are interested in.
    """
    endpoint = APPROVAL_BASE_URL + "/api/home-info/buying-plans/home-price"
    headers = {"Content-Type": "application/json"}
    data = {
        "rmLoanId": rm_loan_id,
        "purchase": {
            "hasBudget": False,
            "minimumPrice": minimum_price,
        },
    }

    try:
        send_request(
            endpoint,
            "POST",
            json=data,
            headers=headers,
            cookies={"sessionToken": session_token},
        )

        return {
            "status": "success",
            "message": "Home price set successfully",
        }
    except HTTPStatusError as e:
        raise ToolExecutionError(f"Error setting home price: {e}") from e


@tool
def set_real_estate_agent(
    context: ToolContext,
    has_agent: Annotated[bool, "Whether the user has a real estate agent"] = False,
    first_name: Annotated[str, "First name of the real estate agent"] = None,
    last_name: Annotated[str, "Last name of the real estate agent"] = None,
    email_address: Annotated[str, "Email address of the real estate agent"] = None,
    work_phone: Annotated[str, "Work phone number of the real estate agent"] = None,
    rm_loan_id: Annotated[
        str, "loan ID from start_mortgage_application", Inferrable(False)
    ] = None,
    session_token: Annotated[
        str, "session token from start_mortgage_application", Inferrable(False)
    ] = None,
) -> Annotated[dict[str, str], "status and message"]:
    """
    Record the user's real estate agent information even if they don't have one.
    if they don't have a real estate agent, set has_agent to False and do not include the other fields.
    """
    endpoint = APPROVAL_BASE_URL + "/api/real-estate-agent"
    headers = {"Content-Type": "application/json"}
    data = {
        "rmLoanId": rm_loan_id,
        "hasAgent": has_agent,
        "firstName": first_name if has_agent else None,
        "lastName": last_name if has_agent else None,
        "emailAddress": email_address if has_agent else None,
    }

    if has_agent:
        phone = _format_phone_number(work_phone)
        data["workPhone"] = phone

    try:
        send_request(
            endpoint,
            "POST",
            json=data,
            headers=headers,
            cookies={"sessionToken": session_token},
        )

        return {
            "status": "success",
            "message": "Real estate agent set successfully",
        }
    except HTTPStatusError as e:
        raise ToolExecutionError(f"Error setting real estate agent: {e}") from e


@tool
def set_living_situation(
    context: ToolContext,
    owner: Annotated[bool, "Whether the user owns their home"],
    street: Annotated[str, "Street of the home"],
    city: Annotated[str, "City of the home"],
    state: Annotated[
        str, "State of the home in two letter format i.e. (CA, TX, NY, etc.)"
    ],
    zip_code: Annotated[str, "Zip code of the home"],
    street2: Annotated[str, "Apartment or unit number if applicable"] = None,
    rm_loan_id: Annotated[
        str, "loan ID from start_mortgage_application", Inferrable(False)
    ] = None,
    session_token: Annotated[
        str, "session token from start_mortgage_application", Inferrable(False)
    ] = None,
) -> Annotated[dict[str, str], "status and message"]:
    """
    Set current living situation (own/rent) of the applicant and their address.
    """
    endpoint = APPROVAL_BASE_URL + "/api/home-info/own-rent-address"
    headers = {"Content-Type": "application/json"}

    data = {
        "rmLoanId": rm_loan_id,
        "currentLivingSituation": {
            "rentOrOwn": "Homeowner" if owner else "Renter",
            "address": {
                "street": street,
                "street2": street2 or "",
                "city": city,
                "state": state,
                "zipCode": zip_code,
            },
        },
    }
    try:
        send_request(
            endpoint,
            "POST",
            json=data,
            headers=headers,
            cookies={"sessionToken": session_token},
        )

        return {
            "status": "success",
            "message": "Living situation set successfully",
        }
    except HTTPStatusError as e:
        raise ToolExecutionError(f"Error setting living situation: {e}") from e


def _format_phone_number(phone_number: str) -> Dict[str, str]:
    """
    Format a phone number string into the required API format.
    Expects format like "1234567890" and converts to
    {"areaCode": "123", "prefix": "456", "line": "7890"}
    """
    # Strip any non-numeric characters
    digits = "".join(filter(str.isdigit, phone_number))

    if len(digits) >= 10:
        return {
            "areaCode": digits[0:3],
            "prefix": digits[3:6],
            "line": digits[6:10],
        }
    else:
        # Default fallback if phone number is incomplete
        return {
            "areaCode": digits[0:3] if len(digits) > 3 else "000",
            "prefix": digits[3:6] if len(digits) > 6 else "000",
            "line": digits[6:] if len(digits) > 6 else "0000",
        }