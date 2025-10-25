from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field


ProductType = Literal["personal", "housing", "cash", "car"]


class Applicant(BaseModel):
    cnic: str = Field(..., description="CNIC without dashes, 13 digits")
    full_name: str
    age_years: int = Field(..., ge=18, le=75)
    monthly_income: float = Field(..., ge=0)
    employer_type: Optional[Literal["salaried", "self_employed", "other"]] = None
    months_at_job: Optional[int] = Field(default=None, ge=0)
    dependents: Optional[int] = Field(default=0, ge=0)
    existing_monthly_debt_payments: float = Field(default=0, ge=0)
    e_cib_negative: Optional[bool] = Field(default=False, description="Has negative e-CIB history")


class LoanRequest(BaseModel):
    amount: float = Field(..., gt=0)
    tenor_months: int = Field(..., gt=0)
    down_payment: Optional[float] = Field(default=0, ge=0)
    property_value: Optional[float] = Field(default=None, ge=0)
    vehicle_value: Optional[float] = Field(default=None, ge=0)
    purpose: Optional[str] = None


class ScoreRequest(BaseModel):
    product_type: ProductType
    applicant: Applicant
    loan: LoanRequest


class ScoreResponse(BaseModel):
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    product_type: ProductType
    probability_of_default: float = Field(..., ge=0.0, le=1.0)
    reasons: List[str]
    rule_flags: Dict[str, bool]
    features_used: Dict[str, float]


