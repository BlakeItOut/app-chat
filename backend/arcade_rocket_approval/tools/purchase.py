from typing import Annotated

import requests
from arcade.sdk import ToolContext, tool

from arcade_rocket_approval.utils import send_request


@tool
def start_application(context: ToolContext) -> Annotated[str, "rmLoanId"]:
    """
    Create a purchase application and return the rmLoanId.
    We'll POST to /api/welcome, which returns an rmLoanId in the "context".
    """
    endpoint = "/api/welcome"
    headers = {"Content-Type": "application/json"}
    payload = {"loanPurpose": "Purchase"}

    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        if not isinstance(response, dict):
            return ""
        context_data = response.get("context", {})
        rm_loan_id = context_data.get("rmLoanId", "")
        if not isinstance(rm_loan_id, str):
            return ""
        return rm_loan_id
    except (requests.exceptions.RequestException, ValueError):
        return ""


@tool
def get_purchase_application_status(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
) -> Annotated[str, "status"]:
    """
    Retrieve the status of a purchase application by rmLoanId.
    Calls GET /api/welcome/{rmLoanId}.
    """
    endpoint = f"/api/welcome/{rm_loan_id}"
    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "GET", headers=headers)
        # For demonstration, we'll just return a mock string
        return "Application status: In Progress"
    except (requests.exceptions.RequestException, ValueError):
        return "Status unavailable"


@tool
def set_home_details(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    details: Annotated[str, "Home Details"],
) -> Annotated[str, "status"]:
    """
    Update the 'home-info/buying-plans/home-details' endpoint for the application.
    POST /api/home-info/buying-plans/home-details with JSON including rmLoanId.
    """
    endpoint = "/api/home-info/buying-plans/home-details"
    payload = {"rmLoanId": rm_loan_id, "homeDetails": details}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Home details updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating home details"


@tool
def set_home_price(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    price_info: Annotated[str, "Home Price"],
) -> Annotated[str, "status"]:
    """
    Update the 'home-info/buying-plans/home-price' endpoint with rmLoanId and price.
    POST /api/home-info/buying-plans/home-price
    """
    endpoint = "/api/home-info/buying-plans/home-price"
    payload = {"rmLoanId": rm_loan_id, "price": price_info}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Home price updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating home price"


@tool
def set_personal_info(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    info: Annotated[str, "Personal Info"],
) -> Annotated[str, "status"]:
    """
    Update personal info. We'll POST /api/personal-info
    """
    endpoint = "/api/personal-info"
    payload = {"rmLoanId": rm_loan_id, "personalInfo": info}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Personal info updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating personal info"


@tool
def set_contact_info(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    email: Annotated[str, "Email"],
    phone: Annotated[str, "Phone"],
) -> Annotated[str, "status"]:
    """
    Update contact info. We'll POST /api/personal-info/contact-info
    """
    endpoint = "/api/personal-info/contact-info"
    payload = {"rmLoanId": rm_loan_id, "email": email, "phone": phone}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Contact info updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating contact info"


@tool
def set_military_status(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    status: Annotated[str, "Military Status"],
) -> Annotated[str, "status"]:
    """
    Update user's military status. We'll POST /api/personal-info/military-status
    """
    endpoint = "/api/personal-info/military-status"
    payload = {"rmLoanId": rm_loan_id, "militaryStatus": status}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Military status updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating military status"


@tool
def set_income(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    annual_income: Annotated[str, "Annual Income"],
) -> Annotated[str, "status"]:
    """
    Update the user's annual income. We'll POST /api/finances/income
    """
    endpoint = "/api/finances/income"
    payload = {"rmLoanId": rm_loan_id, "annualIncome": annual_income}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Income updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating income"


@tool
def set_funds(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    downpayment_funds: Annotated[str, "Downpayment Funds"],
) -> Annotated[str, "status"]:
    """
    Update how the user will fund their down payment. We'll POST /api/finances/funds
    """
    endpoint = "/api/finances/funds"
    payload = {"rmLoanId": rm_loan_id, "downPaymentFunds": downpayment_funds}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Funds info updated"
    except (requests.exceptions.RequestException, ValueError):
        return "Error updating funds info"


@tool
def do_soft_credit_pull(
    context: ToolContext,
    rm_loan_id: Annotated[str, "Rocket Mortgage Loan ID"],
    birthdate: Annotated[str, "Birthdate"],
    ssn_last4: Annotated[str, "SSN Last 4"],
) -> Annotated[str, "status"]:
    """
    Perform a soft credit pull using birthdate & SSN last 4.
    We'll POST /api/credit-info/birthdate-SSN
    """
    endpoint = "/api/credit-info/birthdate-SSN"
    payload = {"rmLoanId": rm_loan_id, "birthdate": birthdate, "ssnLast4": ssn_last4}
    headers = {"Content-Type": "application/json"}
    try:
        send_request(endpoint, "POST", json=payload, headers=headers)
        return "Soft credit pull completed"
    except (requests.exceptions.RequestException, ValueError):
        return "Error performing soft credit pull"
