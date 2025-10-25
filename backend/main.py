from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import Dict, Any

from app.schemas import ScoreRequest, ScoreResponse
from app.rules.sbp import evaluate_sbp_rules
from app.scoring.model import score_application
from app.audit.logger import audit_event
from app.config import get_settings, Settings


app = FastAPI(title="AI Credit Risk Scoring API", version="1.0.0")

# CORS for local dev and Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"status": "ok"}


@app.post("/api/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest, settings: Settings = Depends(get_settings)) -> Any:
    rule_result = evaluate_sbp_rules(request)
    model_result = score_application(request, settings)

    # Combine model probability with hard rule outcomes
    deny_due_to_rules = not rule_result["pass"]
    final_probability = model_result["probability"]

    decision = "DECLINE" if deny_due_to_rules else ("APPROVE" if final_probability >= settings.approval_threshold else "REVIEW")

    response: ScoreResponse = ScoreResponse(
        decision=decision,
        product_type=request.product_type,
        probability_of_default=final_probability,
        reasons=rule_result["reasons"] + model_result["reasons"],
        rule_flags=rule_result["flags"],
        features_used=model_result["features"],
    )

    audit_event(
        event_type="score_decision",
        payload={
            "request": request.model_dump(),
            "response": response.model_dump(),
        },
    )

    return JSONResponse(content=response.model_dump())


@app.websocket("/ws/score")
async def ws_score(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            try:
                req = ScoreRequest(**data)
            except ValidationError as ve:
                await websocket.send_json({"type": "error", "message": str(ve)})
                continue

            # Stream rule evaluation first
            rule_result = evaluate_sbp_rules(req)
            await websocket.send_json({
                "type": "rules",
                "payload": rule_result,
            })

            # Then model output
            settings = get_settings()
            model_result = score_application(req, settings)
            await websocket.send_json({
                "type": "model",
                "payload": model_result,
            })

            deny_due_to_rules = not rule_result["pass"]
            final_probability = model_result["probability"]
            decision = "DECLINE" if deny_due_to_rules else ("APPROVE" if final_probability >= settings.approval_threshold else "REVIEW")

            response = ScoreResponse(
                decision=decision,
                product_type=req.product_type,
                probability_of_default=final_probability,
                reasons=rule_result["reasons"] + model_result["reasons"],
                rule_flags=rule_result["flags"],
                features_used=model_result["features"],
            )
            await websocket.send_json({
                "type": "final",
                "payload": response.model_dump(),
            })
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


