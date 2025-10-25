# AI Credit Risk Scoring (SBP-aligned)

This project provides a Python FastAPI backend and a Next.js frontend for real-time credit risk scoring across personal, housing, cash, and car loans, with prudential checks aligned to State Bank of Pakistan (SBP) consumer financing regulations.

## Monorepo Layout

- `backend/`: FastAPI app with REST and WebSocket endpoints
- `frontend-ci/`: Next.js app (TypeScript, App Router, Tailwind)

## Backend (FastAPI)

### Run locally

```bash
# macOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

- Health: `GET http://localhost:8000/healthz`
- Score: `POST http://localhost:8000/api/v1/score`
- Realtime: `ws://localhost:8000/ws/score`

### Configuration

- `APPROVAL_THRESHOLD` (default 0.7): Threshold for model-based approval; rules can override to decline.

### Key modules

- `app/schemas.py`: Pydantic models for requests/responses
- `app/rules/sbp.py`: SBP prudential rules (age, DBR/DTI, LTV, tenor, e-CIB)
- `app/scoring/model.py`: Simple placeholder PD model and feature engineering
- `app/audit/logger.py`: File-based JSONL audit trail
- `app/config.py`: Settings

## Frontend (Next.js)

### Run locally

```bash
cd frontend-ci
npm install
npm run dev
```

- Configure API base via `.env.local`:

```bash
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

### Features

- Tabs for product types: personal, housing, cash, car
- Form for applicant and loan details
- HTTP scoring and WebSocket live streaming
- Displays decision, PD, rule flags, and reasons

## SBP Alignment Notes (Summary)

- e-CIB negative history: Immediate decline
- Debt Burden Ratio (DBR/DTI) limits (illustrative defaults):
  - Unsecured (personal/cash) ≤ 40%
  - Auto ≤ 50%
  - Housing ≤ 60%
- LTV and down payment caps:
  - Auto LTV ≤ 85%, min down payment ≥ 15%
  - Housing LTV ≤ 85% (illustrative)
- Tenor caps (illustrative): Personal 60m, Cash 36m, Car 84m, Housing 360m

These defaults are placeholders based on typical market guidance. For production, replace with institution-approved thresholds reflecting current SBP circulars and your Board-approved Credit Policy.

## Security & Compliance

- Audit trail for each scoring request/decision
- Deterministic reason codes from rule checks and model signals
- Environment-driven configuration for thresholds

## Next Steps

- Replace placeholder model with trained, validated model (e.g., scikit-learn)
- Parameterize all thresholds from a policy store or database
- Add authentication/authorization and role-based access
- Add rate limiting and request validation logging
- Integrate with SBP e-CIB / licensed credit bureau as per policy
