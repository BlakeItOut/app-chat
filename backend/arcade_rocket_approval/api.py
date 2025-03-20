from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from arcade_rocket_approval.utils import (
    Response,
    handle_request_exception,
    send_request,
)


# Schema definitions
class PhoneNumber(BaseModel):
    """User's phone number"""

    area_code: str
    """3-digit area code (e.g., "313")"""

    prefix: str
    """3-digit prefix (e.g., "555")"""

    line: str
    """4-digit line number (e.g., "1234")"""

    def to_api_format(self) -> dict[str, str]:
        return {"areaCode": self.area_code, "prefix": self.prefix, "line": self.line}


class ContactInfo(BaseModel):
    """User's contact information"""

    first_name: str
    """User's first name"""

    last_name: str
    """User's last name"""

    date_of_birth: str | None = None
    """User's date of birth"""

    email: str
    """User's email address"""

    phone_number: PhoneNumber
    """User's phone number"""

    has_promotional_sms_consent: bool
    """Whether the user has consented to promotional SMS messages"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "firstName": self.first_name,
            "lastName": self.last_name,
            "dateOfBirth": self.date_of_birth,
            "email": self.email,
            "phoneNumber": self.phone_number.to_api_format(),
            "hasPromotionalSmsConsent": self.has_promotional_sms_consent,
        }


class Address(BaseModel):
    """User's address"""

    street: str
    """Primary street address"""

    city: str
    """City name"""

    state: str
    """Two-letter state code"""

    zip_code: str
    """5-digit or 9-digit ZIP code"""

    street2: str | None = None
    """Optional secondary address (apt, unit, etc.)"""

    def to_api_format(self) -> dict[str, str | None]:
        return {
            "street": self.street,
            "street2": self.street2,
            "city": self.city,
            "state": self.state,
            "zipCode": self.zip_code,
        }


class CurrentLivingSituation(BaseModel):
    """User's current living situation"""

    rent_or_own: Literal["Renter", "Owner"]
    """Type of living situation, either 'Renter' or 'Owner'"""

    address: Address
    """Current living address"""

    def to_api_format(self) -> dict[str, Any]:
        return {"type": self.rent_or_own, "address": self.address.to_api_format()}


class Location(BaseModel):
    """Location of a property"""

    city: str
    """City name for the property location"""

    state: str
    """Two-letter state code"""

    zip_code: str
    """5-digit or 9-digit ZIP code"""

    def to_api_format(self) -> dict[str, str]:
        return {"city": self.city, "state": self.state, "zipCode": self.zip_code}


class HomePurchase(BaseModel):
    """User's future home purchase information"""

    has_budget: bool
    """Whether the user has a budget in mind"""

    desired_price: int | None = None
    """Target purchase price in U.S. dollars"""

    minimum_price: int | None = None
    """Minimum acceptable price in U.S. dollars"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "hasBudget": self.has_budget,
            "desiredPrice": self.desired_price,
            "minimumPrice": self.minimum_price,
        }


class BankingAsset(BaseModel):
    """User's banking assets today"""

    bank_amount: int
    """Amount in the bank account in U.S. dollars"""

    bank_name: str
    """Name of the banking institution"""

    type_code: Literal["Checking", "Savings", "Retirement401k"]
    """Type of account (e.g., 'Checking', 'Savings', 'Retirement401k')"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "bankAmount": self.bank_amount,
            "bankName": self.bank_name,
            "typeCode": self.type_code,
        }


class GiftFund(BaseModel):
    """User's gift funds contributing to their overall wealth"""

    gift_amount: int
    """Amount of gift money in U.S. dollars"""

    source: str
    """Source of the gift (e.g., 'Family', 'Friend')"""

    def to_api_format(self) -> dict[str, Any]:
        return {"giftAmount": self.gift_amount, "source": self.source}


class ProceedsFromHomeSale(BaseModel):
    """User's proceeds from selling their current home"""

    listing_price: int
    """listing price of current home in U.S. dollars"""

    current_balance: int
    """Current mortgage balance in U.S. dollars"""

    def to_api_format(self) -> dict[str, int]:
        return {
            "listingPrice": self.listing_price,
            "currentBalance": self.current_balance,
        }


class AssetGroup(BaseModel):
    """Group of assets"""

    assets: list[BankingAsset] = Field(default_factory=list)
    """list of assets (banking, gift, proceeds from home sale)"""

    def to_api_format(self) -> dict[str, list[dict[str, Any]]]:
        return {"assets": [asset.to_api_format() for asset in self.assets]}


class PrimaryAssets(AssetGroup):
    """User's primary assets"""

    proceeds_from_home_sale: ProceedsFromHomeSale = Field(
        default_factory=ProceedsFromHomeSale
    )
    """Proceeds from selling current home"""

    gift_funds: list[GiftFund] = Field(default_factory=list)
    """list of monetary gifts"""

    def to_api_format(self) -> dict[str, Any]:
        result = super().to_api_format()
        result["proceedsFromHomeSale"] = self.proceeds_from_home_sale.to_api_format()
        result["giftFunds"] = [gift.to_api_format() for gift in self.gift_funds]
        return result


class SpouseAssets(AssetGroup):
    pass
    """Inherits banking assets from AssetGroup"""


class Income(BaseModel):
    """User's income"""

    annual_income: int
    """Annual income in U.S. dollars"""

    income_type: Literal[
        "Employment",
        "Self-Employed",
        "Unemployment",
        "Social Security",
        "Pension",
        "Other",
    ]
    """Type of income"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "annualIncome": self.annual_income,
            "incomeType": self.income_type,
        }


class MilitaryStatus(BaseModel):
    """User's military status"""

    military_status: Literal["Active Duty", "Reserve", "None"]
    """Military status"""

    military_branch: Literal["Army", "Navy", "Air Force", "Marine Corps", "None"]
    """Military branch"""

    service_type: Literal["Regular", "Reserve", "None"]
    """Service type"""

    expiration_date: str
    """Expiration date of the military status"""

    eligible_for_va: bool
    """Whether the user is eligible for VA benefits"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "militaryStatus": self.military_status or None,
            "militaryBranch": self.military_branch or None,
            "serviceType": self.service_type or None,
            "expirationDate": self.expiration_date or None,
            "eligibleForVA": self.eligible_for_va or False,
        }


class RealEstateAgent(BaseModel):
    """Real estate agent information"""

    has_agent: bool
    """Whether the user has a real estate agent"""

    first_name: str
    """First name of the real estate agent"""

    last_name: str
    """Last name of the real estate agent"""

    email_address: str
    """Email address of the real estate agent"""

    work_phone: PhoneNumber
    """Work phone number of the real estate agent"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "hasAgent": self.has_agent,
            "firstName": self.first_name,
            "lastName": self.last_name,
            "emailAddress": self.email_address,
            "workPhone": self.work_phone.to_api_format() if self.work_phone else None,
        }


class HomeDetails(BaseModel):
    """User's home details"""

    location: Location
    """Location of the home"""

    property_type: Literal["Single Family", "Multi Family", "Condo", "Townhouse"]
    """Type of property"""

    occupancy_type: Literal["Primary Residence", "Investment Property", "Second Home"]
    """Occupancy type"""


class IdealHomePrice(BaseModel):
    """Ideal home price"""

    desired_price: int
    """Desired price in U.S. dollars"""

    minimum_price: int
    """Minimum acceptable price in U.S. dollars"""

    def to_api_format(self) -> dict[str, Any]:
        return {
            "desiredPrice": self.desired_price,
            "minimumPrice": self.minimum_price,
        }


class PersonalInfo(BaseModel):
    """User's personal information"""

    first_name: str
    """User's first name"""

    last_name: str
    """User's last name"""

    date_of_birth: str
    """User's date of birth"""

    marital_status: Literal["Married", "Single"]
    """User's marital status"""

    is_spouse_on_loan: bool
    """Whether the user's spouse is on the loan"""

    def to_api_format(self) -> dict[str, Any]:
        result = {
            "firstName": self.first_name,
            "lastName": self.last_name,
        }
        if self.date_of_birth:
            result["dateOfBirth"] = self.date_of_birth
        if self.marital_status:
            result["maritalStatus"] = self.marital_status
        if self.is_spouse_on_loan is not None:
            result["isSpouseOnLoan"] = self.is_spouse_on_loan
        return result


class RocketUserContext(BaseModel):
    """
    Context for a Rocket Mortgage user.
    """

    personal_info: PersonalInfo
    """Information like name, """

    contact_info: ContactInfo
    """User's contact information"""

    phone_number: PhoneNumber
    """User's phone number"""

    address: Address
    """User's address"""

    living_situation: CurrentLivingSituation
    """User's living situation"""

    home_purchase: HomePurchase
    """User's home purchase information"""

    primary_assets: PrimaryAssets
    """User's primary assets"""

    spouse_assets: SpouseAssets
    """User's spouse's assets"""

    marital_status: Literal["Married", "Single"]
    """User's marital status"""

    income: Income
    """User's income"""

    military_status: MilitaryStatus
    """User's military status"""

    real_estate_agent: RealEstateAgent
    """User's real estate agent"""

    home_details: HomeDetails
    """User's home details"""

    ideal_home_price: IdealHomePrice
    """User's ideal home price"""

    has_promotional_sms_consent: bool
    """Whether the user has consented to promotional SMS messages"""

    def info_display(self) -> str:
        info_display = f"""
        Please verify your information:

        Name: {self.personal_info.first_name} {self.personal_info.last_name}
        Email: {self.contact_info.email}
        Phone: {self.contact_info.phone_number if hasattr(self.contact_info, 'phone_number') else 'Not provided'}
        Annual Income: ${self.income.annual_income if hasattr(self.income, 'annual_income') else 'Not provided'}
        Military Status: {self.military_status.military_status if hasattr(self.military_status, 'military_status') else 'Not provided'}

        Is this information correct? (yes/no)
        """
        return info_display


# API Functions
def start_application() -> Response[dict[str, str]]:
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
            return Response.error("Invalid response format")

        context_data = response.get("context", {})
        rm_loan_id = context_data.get("rmLoanId", "")

        if not isinstance(rm_loan_id, str):
            return Response.error("Invalid rmLoanId format")

        return Response.success(
            "Application started successfully",
            {"rmLoanId": rm_loan_id},
            raw_response=response,
        )
    except (requests.exceptions.RequestException, ValueError) as e:
        return handle_request_exception(e)


def get_purchase_application_status(rm_loan_id: str) -> Response[dict[str, str]]:
    """
    Retrieve the status of a purchase application by rmLoanId.
    Calls GET /api/welcome/{rmLoanId}.
    """
    endpoint = f"/api/welcome/{rm_loan_id}"
    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "GET", headers=headers)
        # For demonstration, we'll just return a mock string
        return Response.success(
            "Application status retrieved",
            {"appStatus": "In Progress"},
            raw_response=response,
        )
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Status unavailable: {str(e)}")


def set_home_details(
    rm_loan_id: str,
    home_details: HomeDetails,
) -> Response[None]:
    """
    Update the 'home-info/buying-plans/home-details' endpoint for the application.
    POST /api/home-info/buying-plans/home-details with JSON including rmLoanId.
    """
    endpoint = "/api/home-info/buying-plans/home-details"
    payload = {
        "rmLoanId": rm_loan_id,
        "homeDetails": home_details.to_api_format(),
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Home details updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating home details: {str(e)}")


def set_home_price(rm_loan_id: str, purchase: HomePurchase) -> Response[None]:
    """
    Update the 'home-info/buying-plans/home-price' endpoint with rmLoanId and price.
    POST /api/home-info/buying-plans/home-price
    """
    endpoint = "/api/home-info/buying-plans/home-price"
    payload = {"rmLoanId": rm_loan_id, "purchase": purchase.to_api_format()}
    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Home price updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating home price: {str(e)}")


def set_real_estate_agent(
    rm_loan_id: str,
    real_estate_agent: RealEstateAgent,
) -> Response[None]:
    """
    Set real estate agent information.
    POST /api/home-info/buying-plans/agent
    """
    endpoint = "/api/home-info/buying-plans/agent"
    payload = {
        "rmLoanId": rm_loan_id,
        "realEstateAgent": real_estate_agent.to_api_format(),
    }

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Real estate agent info updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating real estate agent info: {str(e)}")


def set_living_situation(
    rm_loan_id: str, living_situation: CurrentLivingSituation
) -> Response[None]:
    """
    Set current living situation (own/rent) and address.
    POST /api/home-info/own-rent-address
    """
    endpoint = "/api/home-info/own-rent-address"
    payload = {
        "rmLoanId": rm_loan_id,
        "currentLivingSituation": living_situation.to_api_format(),
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Living situation updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating living situation: {str(e)}")


def set_personal_info(
    rm_loan_id: str,
    first_name: str,
    last_name: str,
    date_of_birth: str | None = None,
    marital_status: Literal["Married", "Single"] | None = None,
    is_spouse_on_loan: bool | None = None,
) -> Response[None]:
    """
    Update personal info. We'll POST /api/personal-info
    """
    endpoint = "/api/personal-info"

    # Create the PersonalInfo object
    personal_info = PersonalInfo(
        first_name=first_name,
        last_name=last_name,
        date_of_birth=date_of_birth,
        marital_status=marital_status,
        is_spouse_on_loan=is_spouse_on_loan,
    )

    payload = {"rmLoanId": rm_loan_id, "personalInfo": personal_info.to_api_format()}

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Personal info updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating personal info: {str(e)}")


def set_contact_info(
    rm_loan_id: str,
    first_name: str = None,
    last_name: str = None,
    email: str = None,
    phone_number: PhoneNumber = None,
    has_promotional_sms_consent: bool = False,
    contact_info: ContactInfo = None,
) -> Response[None]:
    """
    Update contact info. We'll POST /api/personal-info/contact-info
    Can accept either individual parameters or a ContactInfo object.
    """
    endpoint = "/api/personal-info/contact-info"

    if contact_info:
        # Use the contact_info object if provided
        payload = {
            "rmLoanId": rm_loan_id,
            "firstName": contact_info.first_name,
            "lastName": contact_info.last_name,
            "email": contact_info.email,
            "phoneNumber": contact_info.phone_number.to_api_format(),
            "hasPromotionalSmsConsent": contact_info.has_promotional_sms_consent,
        }
    else:
        # Otherwise use individual parameters
        payload = {
            "rmLoanId": rm_loan_id,
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            "phoneNumber": phone_number.to_api_format() if phone_number else None,
            "hasPromotionalSmsConsent": has_promotional_sms_consent,
        }

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Contact info updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating contact info: {str(e)}")


def set_military_status(
    rm_loan_id: str,
    military_status: Literal["Active Duty", "Reserve", "None"] | None = None,
    military_branch: Literal["Army", "Navy", "Air Force", "Marine Corps", "None"]
    | None = None,
    service_type: Literal["Regular", "Reserve", "None"] | None = None,
    expiration_date: str | None = None,
    eligible_for_va: bool = False,
) -> Response[None]:
    """
    Update user's military status. We'll POST /api/personal-info/military-status
    """
    endpoint = "/api/personal-info/military-status"
    payload = {
        "rmLoanId": rm_loan_id,
        "militaryStatus": military_status,
        "eligibleForVA": eligible_for_va,
    }

    if military_status and military_status != "None":
        payload.update(
            {
                "militaryBranch": military_branch,
                "serviceType": service_type,
                "expirationDate": expiration_date,
            }
        )

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Military status updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating military status: {str(e)}")


def set_marital_status(
    rm_loan_id: str,
    marital_status: str,  # "Married" or "Single"
    is_spouse_on_loan: bool | None = None,
) -> Response[None]:
    """
    Set marital status information.
    POST /api/personal-info/marital-status
    """
    endpoint = "/api/personal-info/marital-status"
    payload = {"rmLoanId": rm_loan_id, "maritalStatus": marital_status}

    if marital_status == "Married":
        payload["isSpouseOnLoan"] = is_spouse_on_loan

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Marital status updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating marital status: {str(e)}")


def set_income(
    rm_loan_id: str,
    annual_income: int,
    income_type: str = "Employment",
    employer_name: str | None = None,
    job_title: str | None = None,
    years_at_employer: int | None = None,
    months_at_employer: int | None = None,
) -> Response[None]:
    """
    Update the user's annual income. We'll POST /api/finances/income
    """
    endpoint = "/api/finances/income"
    payload = {
        "rmLoanId": rm_loan_id,
        "annualIncome": annual_income,
        "incomeType": income_type,
    }

    if income_type == "Employment" and employer_name:
        payload.update(
            {
                "employerName": employer_name,
                "jobTitle": job_title,
                "yearsAtEmployer": years_at_employer,
                "monthsAtEmployer": months_at_employer,
            }
        )

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Income updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating income: {str(e)}")


def set_funds(
    rm_loan_id: str,
    primary_assets: PrimaryAssets,
    spouse_assets: SpouseAssets | None = None,
    down_payment_percentage: int | None = None,
) -> Response[None]:
    """
    Update how the user will fund their down payment. We'll POST /api/finances/funds
    """
    endpoint = "/api/finances/funds"
    payload = {"rmLoanId": rm_loan_id, "primaryAssets": primary_assets.to_api_format()}

    if spouse_assets:
        payload["spouseAssets"] = spouse_assets.to_api_format()

    if down_payment_percentage is not None:
        payload["downPaymentPercentage"] = down_payment_percentage

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Funds info updated", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error updating funds info: {str(e)}")


def do_soft_credit_pull(
    rm_loan_id: str, birthdate: str, ssn_last4: str, full_ssn: str | None = None
) -> Response[None]:
    """
    Perform a soft credit pull using birthdate & SSN last 4.
    We'll POST /api/credit-info/birthdate-SSN
    """
    endpoint = "/api/credit-info/birthdate-SSN"
    payload = {"rmLoanId": rm_loan_id, "birthdate": birthdate, "ssnLast4": ssn_last4}

    if full_ssn:
        payload["fullSsn"] = full_ssn

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)
        return Response.success("Soft credit pull completed", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error performing soft credit pull: {str(e)}")


def create_account(
    rm_loan_id: str,
    client_first_name: str,
    client_last_name: str,
    client_username: str,
    password: str,
    redirect: str = "https://dashboard.rocketmortgage.com/?RocketAccountIntent=rmapplication",
    rm_client_id: str | None = None,
) -> Response[dict[str, str]]:
    """
    Create a Rocket Mortgage account after completing the application steps.
    We'll POST /api/account-create with the necessary account information.
    """
    endpoint = "/api/account-create"
    payload = {
        "clientFirstName": client_first_name,
        "clientLastName": client_last_name,
        "clientUsername": client_username,
        "password": password,
        "rmLoanId": rm_loan_id,
        "redirect": redirect,
    }

    if rm_client_id:
        payload["rmClientId"] = rm_client_id

    headers = {"Content-Type": "application/json"}
    try:
        response = send_request(endpoint, "POST", json=payload, headers=headers)

        # Extract the rocketAccountId and update the return message
        context_data = response.get("context", {})
        rocket_account_id = context_data.get("rocketAccountId")

        if rocket_account_id:
            return Response.success(
                "Account created successfully!",
                {"rocketAccountId": rocket_account_id},
                raw_response=response,
            )
        return Response.success("Account created successfully!", raw_response=response)
    except (requests.exceptions.RequestException, ValueError) as e:
        return Response.error(f"Error creating account: {str(e)}")
