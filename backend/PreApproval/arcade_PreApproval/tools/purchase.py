from typing import Annotated, Optional, List
from arcade.sdk import tool
import requests
import json
from dataclasses import dataclass, field

@dataclass
class ChatData:
    firstName: str = ""
    lastName: str = ""
    email: str = ""
    loanPurpose: str = "Purchase"
    state: Optional[str] = None
    phNumber: Optional[str] = None
    loanNumber: Optional[str] = None

@dataclass
class Analytics:
    account_created: str = "no"
    conversion_value: str = ""

@dataclass
class FeatureFlags:
    featureEnableAccountCreateModalWithAuth0: bool = True
    featureEnableRocketAssistChat: bool = False
    featureEnableLivChat: bool = False
    featureEnableHelSolutions: bool = False
    featureEnableWelcomeBanner: bool = False
    enableRefinanceFullFlow: bool = False
    enableRocketHomes: bool = False

@dataclass
class ChatClientAttributes:
    firstName: str = ""
    lastName: str = ""
    emailAddress: str = ""
    loanPurpose: str = "Purchase"
    loanState: Optional[str] = None
    phoneNumber: Optional[str] = None
    loanNumber: Optional[str] = None
    rocketAccountId: Optional[str] = None
    rmLoanId: Optional[str] = None
    rmClientId: Optional[str] = None

@dataclass
class AvoData:
    userId: Optional[str] = None
    mortgagePropertyType: Optional[str] = None
    mortgageEstYearlyPropertyTaxes: Optional[str] = None
    mortgageEstYearlyHomeownersInsurance: Optional[str] = None
    mortgageZipCode: Optional[str] = None
    mortgageEstCashToBuy: Optional[str] = None
    mortgageEstMonthlyRentPayment: Optional[str] = None
    mortgageEstCurrentMonthlyPayment: Optional[str] = None
    mortgageRemainingTermInYears: Optional[str] = None
    accountCreated: Optional[str] = None
    accountType: Optional[str] = None
    creditPullType: Optional[str] = None
    creditPullConsent: bool = False
    creditPullTriggerSource: Optional[str] = None
    creditScore: int = 0
    mortgagePropertyUse: Optional[str] = None
    mortgageStatusAtEngagement: Optional[str] = None
    mortgageTimeframeToBuy: Optional[str] = None
    military: bool = False
    militaryAssociation: Optional[str] = None
    militaryVaDisabilityBenefits: bool = False
    mortgageCoborrowerType: str = "none"
    mortgageLoanNumberCreated: bool = True
    mortgageDownPaymentFundsType: List[str] = field(default_factory=list)

@dataclass
class Context:
    rmLoanId: Optional[str] = None
    rocketAccountId: Optional[str] = None
    primaryFirstName: str = ""
    primaryLastName: str = ""
    primaryEmail: str = ""
    rmClientId: Optional[str] = None
    qls: Optional[str] = None
    affiliateSubId: Optional[str] = None
    leadTypeCode: List[str] = field(default_factory=lambda: ["RLHAP"])
    prefillSource: Optional[str] = None
    isLoggedIn: bool = False
    isSignedIn: bool = False
    isSPA: bool = False
    isSpouseOnLoan: Optional[bool] = None
    chatData: ChatData = field(default_factory=ChatData)
    analytics: Analytics = field(default_factory=Analytics)
    featureFlags: FeatureFlags = field(default_factory=FeatureFlags)
    fullFlowPhaseConfig: Optional[str] = None
    timeFrameToBuy: Optional[str] = None
    loanList: List[str] = field(default_factory=list)
    loanPurpose: str = "Purchase"
    chatClientAttributes: ChatClientAttributes = field(default_factory=ChatClientAttributes)
    refinanceGoal: Optional[str] = None
    isDebtConSolution: bool = False
    avoData: AvoData = field(default_factory=AvoData)

    @classmethod
    def from_dict(cls, data: dict) -> 'Context':
        if not data:
            return cls()

        # Create nested objects
        chat_data = ChatData(**data.get('chatData', {})) if data.get('chatData') else ChatData()
        analytics = Analytics(**data.get('analytics', {})) if data.get('analytics') else Analytics()
        feature_flags = FeatureFlags(**data.get('featureFlags', {})) if data.get('featureFlags') else FeatureFlags()
        chat_client_attributes = ChatClientAttributes(**data.get('chatClientAttributes', {})) if data.get('chatClientAttributes') else ChatClientAttributes()
        avo_data = AvoData(**data.get('avoData', {})) if data.get('avoData') else AvoData()

        return cls(
            rmLoanId=data.get('rmLoanId'),
            rocketAccountId=data.get('rocketAccountId'),
            primaryFirstName=data.get('primaryFirstName', ""),
            primaryLastName=data.get('primaryLastName', ""),
            primaryEmail=data.get('primaryEmail', ""),
            rmClientId=data.get('rmClientId'),
            leadTypeCode=data.get('leadTypeCode', ["RLHAP"]),
            chatData=chat_data,
            analytics=analytics,
            featureFlags=feature_flags,
            chatClientAttributes=chat_client_attributes,
            avoData=avo_data,
            isLoggedIn=data.get('isLoggedIn', False),
            isSignedIn=data.get('isSignedIn', False),
            isSPA=data.get('isSPA', False),
            loanPurpose=data.get('loanPurpose', "Purchase"),
            loanList=data.get('loanList', []),
            isDebtConSolution=data.get('isDebtConSolution', False)
        )

@tool
def start_application() -> str:
    """Create a purchase application and return the rmLoanId"""
    
    url = "https://application.beta.rocketmortgage.com/api/welcome"
    headers = {
        "Content-Type": "application/json"
    }
    payload = {
        "loanPurpose": "Purchase"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Add explicit type checking and conversion
        response_data = response.json()
        if not isinstance(response_data, dict):
            return ''
            
        context_data = response_data.get('context', {})
        if not isinstance(context_data, dict):
            return ''
            
        rm_loan_id = context_data.get('rmLoanId')
        if not isinstance(rm_loan_id, str):
            return ''
            
        return rm_loan_id
        
    except (requests.exceptions.RequestException, ValueError) as e:
        # Log the error (you might want to add proper logging)
        print(f"Error making request: {str(e)}")
        # Return empty string in case of error
        return ''


