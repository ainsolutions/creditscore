import json
import time
from pathlib import Path
from typing import Any, Dict


AUDIT_DIR = Path(__file__).resolve().parent.parent.parent / "audit_logs"
AUDIT_DIR.mkdir(exist_ok=True)


def audit_event(event_type: str, payload: Dict[str, Any]) -> None:
    ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    filename = AUDIT_DIR / f"{ts}-{event_type}.jsonl"
    record = {"ts": ts, "event": event_type, "payload": payload}
    with open(filename, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


