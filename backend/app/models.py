from __future__ import annotations

from sqlalchemy import Integer, String, Float, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime
from app.db import Base


class ScoreEvent(Base):
    __tablename__ = "score_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    product_type: Mapped[str] = mapped_column(String(32), index=True)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    probability_of_default: Mapped[float] = mapped_column(Float)
    applicant_cnic: Mapped[str] = mapped_column(String(32), index=True)
    request_json: Mapped[dict] = mapped_column(JSON)
    response_json: Mapped[dict] = mapped_column(JSON)


class PolicyConfig(Base):
    __tablename__ = "policy_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    data: Mapped[dict] = mapped_column(JSON)


