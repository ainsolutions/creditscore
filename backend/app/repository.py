from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.models import ScoreEvent, PolicyConfig


def create_score_event(session: Session, *,
                       product_type: str,
                       decision: str,
                       probability_of_default: float,
                       applicant_cnic: str,
                       request_json: dict,
                       response_json: dict) -> ScoreEvent:
    ev = ScoreEvent(
        product_type=product_type,
        decision=decision,
        probability_of_default=probability_of_default,
        applicant_cnic=applicant_cnic,
        request_json=request_json,
        response_json=response_json,
    )
    session.add(ev)
    session.flush()
    return ev


def list_score_events(session: Session, *,
                      product_type: Optional[str] = None,
                      decision: Optional[str] = None,
                      cnic: Optional[str] = None,
                      limit: int = 50) -> List[ScoreEvent]:
    stmt = select(ScoreEvent).order_by(ScoreEvent.id.desc()).limit(limit)
    if product_type:
        stmt = stmt.where(ScoreEvent.product_type == product_type)
    if decision:
        stmt = stmt.where(ScoreEvent.decision == decision)
    if cnic:
        stmt = stmt.where(ScoreEvent.applicant_cnic == cnic)
    return list(session.scalars(stmt))


DEFAULT_POLICY = {
    "min_age": {"personal": 21, "cash": 21, "car": 21, "housing": 25},
    "max_age": {"personal": 65, "cash": 65, "car": 65, "housing": 70},
    "max_dbr": {"personal": 0.4, "cash": 0.4, "car": 0.5, "housing": 0.6},
    "auto": {"min_down": 0.15, "max_ltv": 0.85},
    "housing": {"max_ltv": 0.85},
    "tenor_limits": {"personal": 60, "cash": 36, "car": 84, "housing": 360},
}


def get_policy(session: Session) -> dict:
    row = session.get(PolicyConfig, 1)
    if row is None:
        row = PolicyConfig(id=1, data=DEFAULT_POLICY)
        session.add(row)
        session.flush()
    return row.data


def update_policy(session: Session, data: dict) -> dict:
    row = session.get(PolicyConfig, 1)
    if row is None:
        row = PolicyConfig(id=1, data=DEFAULT_POLICY)
        session.add(row)
        session.flush()
    # shallow merge top-level keys
    merged = {**row.data, **data}
    row.data = merged
    session.flush()
    return row.data


