import json
from dataclasses import dataclass, field
from typing import Annotated, List, Optional

import requests


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
    enteredName: bool = False
    enteredPhone: bool = False
    enteredEmail: bool = False
    enteredCurrentAddress: bool = False
    mortgageCurrentLivingSituation: Optional[str] = None
    mortgageFoundHome: bool = False
    mortgageHasBudget: bool = False
    mortgagePurchaseState: Optional[str] = None
    maritalStatus: Optional[str] = None
    mortgageEstPurchasePrice: Optional[int] = None
    creditPullSucceeded: Optional[bool] = None
    creditReportStatus: Optional[str] = None
    militaryBranchOfService: Optional[str] = None
    militaryServiceType: Optional[str] = None


@dataclass
class RocketContext:
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
    chatClientAttributes: ChatClientAttributes = field(
        default_factory=ChatClientAttributes
    )
    refinanceGoal: Optional[str] = None
    isDebtConSolution: bool = False
    avoData: AvoData = field(default_factory=AvoData)

    @classmethod
    def from_dict(cls, data: dict) -> "RocketContext":
        if not data:
            return cls()

        # Create nested objects
        chat_data = (
            ChatData(**data.get("chatData", {})) if data.get("chatData") else ChatData()
        )
        analytics = (
            Analytics(**data.get("analytics", {}))
            if data.get("analytics")
            else Analytics()
        )
        feature_flags = (
            FeatureFlags(**data.get("featureFlags", {}))
            if data.get("featureFlags")
            else FeatureFlags()
        )
        chat_client_attributes = (
            ChatClientAttributes(**data.get("chatClientAttributes", {}))
            if data.get("chatClientAttributes")
            else ChatClientAttributes()
        )
        avo_data = (
            AvoData(**data.get("avoData", {})) if data.get("avoData") else AvoData()
        )

        return cls(
            rmLoanId=data.get("rmLoanId"),
            rocketAccountId=data.get("rocketAccountId"),
            primaryFirstName=data.get("primaryFirstName", ""),
            primaryLastName=data.get("primaryLastName", ""),
            primaryEmail=data.get("primaryEmail", ""),
            rmClientId=data.get("rmClientId"),
            leadTypeCode=data.get("leadTypeCode", ["RLHAP"]),
            chatData=chat_data,
            analytics=analytics,
            featureFlags=feature_flags,
            chatClientAttributes=chat_client_attributes,
            avoData=avo_data,
            isLoggedIn=data.get("isLoggedIn", False),
            isSignedIn=data.get("isSignedIn", False),
            isSPA=data.get("isSPA", False),
            loanPurpose=data.get("loanPurpose", "Purchase"),
            loanList=data.get("loanList", []),
            isDebtConSolution=data.get("isDebtConSolution", False),
        )
