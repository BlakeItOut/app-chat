import logging
from enum import Enum
from typing import Annotated, Any, Dict, Literal, Optional

import requests
from arcade.sdk import ToolContext, tool
from arcade.sdk.annotations import Inferrable
from arcade.sdk.errors import ToolExecutionError
from httpx import Client
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
        token, response = send_request(endpoint, "POST", json=payload, headers=headers)
        if not isinstance(response, dict):
            raise ToolExecutionError("Invalid response format")

        context_data = response.get("context", {})
        rm_loan_id = context_data.get("rmLoanId", "")

        if not isinstance(rm_loan_id, str):
            raise ToolExecutionError("Invalid rmLoanId format")

        return {
            "rmLoanId": rm_loan_id,
            "sessionToken": token,
        }
    except (requests.exceptions.RequestException, ValueError) as e:
        return handle_request_exception(e)


@tool
def set_new_home_details(
    context: ToolContext,
    new_home_city: Annotated[str, "City of the new home"],
    new_home_state: Annotated[str, "State of the new home"],
    new_home_zip_code: Annotated[str, "Zip code of the new home"],
    new_home_occupancy_type: Annotated[
        OccupancyType, "Type of occupancy"
    ] = OccupancyType.PRIMARY,
    found_new_home: Annotated[bool, "Whether the user has found a new home"] = False,
    rm_loan_id: Annotated[
        str, "loan ID from start_mortgage_application", Inferrable(False)
    ] = None,
    session_token: Annotated[
        str, "session token from start_mortgage_application", Inferrable(False)
    ] = None,
) -> Annotated[dict[str, str], "response from API"]:
    """
    record the user's new home details weather they have found a new home or not.
    POST /api/home-info/buying-plans/home-details
    """
    client = Client(base_url=APPROVAL_BASE_URL)
    client.cookies.set("sessionToken", session_token)

    # assume user has not found a new home
    data = {"rmLoanId": rm_loan_id, "buyingPlans": found_new_home}

    print(data)
    response = client.post("/api/home-info/buying-plans", json=data)
    print(f"response from API: {response.json()}")

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
    response = client.post("/api/home-info/buying-plans/home-details", json=data)
    print(f"response from API: {response.json()}")
    return {
        "status": "success",
        "message": "Home details set successfully",
    }
