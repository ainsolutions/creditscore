"use client";

import { useMemo, useRef, useState } from "react";

type ProductType = "personal" | "housing" | "cash" | "car";

interface Applicant {
  cnic: string;
  full_name: string;
  age_years: number;
  monthly_income: number;
  employer_type?: "salaried" | "self_employed" | "other";
  months_at_job?: number;
  dependents?: number;
  existing_monthly_debt_payments: number;
  e_cib_negative?: boolean;
}

interface LoanRequest {
  amount: number;
  tenor_months: number;
  down_payment?: number;
  property_value?: number;
  vehicle_value?: number;
  purpose?: string;
}

interface ScoreResponse {
  decision: "APPROVE" | "REVIEW" | "DECLINE";
  product_type: ProductType;
  probability_of_default: number;
  reasons: string[];
  rule_flags: Record<string, boolean>;
  features_used: Record<string, number>;
}

const DEFAULT_APPLICANT: Applicant = {
  cnic: "1234567890123",
  full_name: "Test User",
  age_years: 30,
  monthly_income: 150000,
  employer_type: "salaried",
  months_at_job: 24,
  dependents: 1,
  existing_monthly_debt_payments: 10000,
  e_cib_negative: false,
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function toWs(url: string): string {
  if (url.startsWith("https://")) return url.replace("https://", "wss://");
  if (url.startsWith("http://")) return url.replace("http://", "ws://");
  return `ws://${url}`;
}

export default function Home() {
  const [product, setProduct] = useState<ProductType>("personal");
  const [applicant, setApplicant] = useState<Applicant>(DEFAULT_APPLICANT);
  const [loan, setLoan] = useState<LoanRequest>({ amount: 500000, tenor_months: 24 });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ScoreResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [liveLog, setLiveLog] = useState<string[]>([]);
  const [liveFinal, setLiveFinal] = useState<ScoreResponse | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  type LoanField = "property_value" | "vehicle_value" | "down_payment";
  const loanSpecificFields = useMemo<LoanField[]>(() => {
    if (product === "housing") return ["property_value"];
    if (product === "car") return ["vehicle_value", "down_payment"];
    return [];
  }, [product]);

  async function score() {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${API_BASE}/api/v1/score`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ product_type: product, applicant, loan }),
      });
      if (!res.ok) throw new Error(`Request failed: ${res.status}`);
      const data: ScoreResponse = await res.json();
      setResult(data);
    } catch (e: any) {
      setError(e?.message || "Failed to score");
    } finally {
      setLoading(false);
    }
  }

  function liveScore() {
    setLiveLog([]);
    setLiveFinal(null);
    setError(null);
    try {
      const ws = new WebSocket(`${toWs(API_BASE)}/ws/score`);
      wsRef.current = ws;
      ws.onopen = () => {
        ws.send(JSON.stringify({ product_type: product, applicant, loan }));
      };
      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "rules") {
            setLiveLog((l) => [...l, `Rules evaluated: ${JSON.stringify(msg.payload)}`]);
          } else if (msg.type === "model") {
            setLiveLog((l) => [...l, `Model scored: ${JSON.stringify(msg.payload)}`]);
          } else if (msg.type === "final") {
            setLiveLog((l) => [...l, `Final decision ready`]);
            setLiveFinal(msg.payload as ScoreResponse);
            ws.close();
          }
        } catch (err) {
          setLiveLog((l) => [...l, `Malformed message`]);
        }
      };
      ws.onerror = () => setError("WebSocket error");
      ws.onclose = () => {};
    } catch (e: any) {
      setError(e?.message || "Failed to open WebSocket");
    }
  }

  return (
    <div className="min-h-screen p-6 max-w-5xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">AI Credit Risk Scoring</h1>

      <div className="flex gap-2 mb-6">
        {(["personal", "housing", "cash", "car"] as ProductType[]).map((p) => (
          <button
            key={p}
            onClick={() => setProduct(p)}
            className={`px-3 py-1 rounded border ${product === p ? "bg-blue-600 text-white" : "bg-white"}`}
          >
            {p}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <section className="border rounded p-4">
          <h2 className="font-medium mb-2">Applicant</h2>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <label className="col-span-1">CNIC
              <input className="w-full border rounded p-1"
                value={applicant.cnic}
                onChange={(e) => setApplicant({ ...applicant, cnic: e.target.value })}
              />
            </label>
            <label className="col-span-1">Full Name
              <input className="w-full border rounded p-1"
                value={applicant.full_name}
                onChange={(e) => setApplicant({ ...applicant, full_name: e.target.value })}
              />
            </label>
            <label>Age
              <input type="number" className="w-full border rounded p-1"
                value={applicant.age_years}
                onChange={(e) => setApplicant({ ...applicant, age_years: Number(e.target.value) })}
              />
            </label>
            <label>Monthly Income
              <input type="number" className="w-full border rounded p-1"
                value={applicant.monthly_income}
                onChange={(e) => setApplicant({ ...applicant, monthly_income: Number(e.target.value) })}
              />
            </label>
            <label>Dependents
              <input type="number" className="w-full border rounded p-1"
                value={applicant.dependents ?? 0}
                onChange={(e) => setApplicant({ ...applicant, dependents: Number(e.target.value) })}
              />
            </label>
            <label>Existing Debt (monthly)
              <input type="number" className="w-full border rounded p-1"
                value={applicant.existing_monthly_debt_payments}
                onChange={(e) => setApplicant({ ...applicant, existing_monthly_debt_payments: Number(e.target.value) })}
              />
            </label>
            <label className="col-span-2 flex items-center gap-2">
              <input type="checkbox" checked={!!applicant.e_cib_negative}
                onChange={(e) => setApplicant({ ...applicant, e_cib_negative: e.target.checked })}
              /> Negative e-CIB
            </label>
          </div>
        </section>

        <section className="border rounded p-4">
          <h2 className="font-medium mb-2">Loan</h2>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <label>Amount
              <input type="number" className="w-full border rounded p-1" value={loan.amount}
                onChange={(e) => setLoan({ ...loan, amount: Number(e.target.value) })}
              />
            </label>
            <label>Tenor (months)
              <input type="number" className="w-full border rounded p-1" value={loan.tenor_months}
                onChange={(e) => setLoan({ ...loan, tenor_months: Number(e.target.value) })}
              />
            </label>
            {loanSpecificFields.includes("down_payment") && (
              <label>Down Payment
                <input type="number" className="w-full border rounded p-1" value={loan.down_payment ?? 0}
                  onChange={(e) => setLoan({ ...loan, down_payment: Number(e.target.value) })}
                />
              </label>
            )}
            {loanSpecificFields.includes("property_value") && (
              <label>Property Value
                <input type="number" className="w-full border rounded p-1" value={loan.property_value ?? 0}
                  onChange={(e) => setLoan({ ...loan, property_value: Number(e.target.value) })}
                />
              </label>
            )}
            {loanSpecificFields.includes("vehicle_value") && (
              <label>Vehicle Value
                <input type="number" className="w-full border rounded p-1" value={loan.vehicle_value ?? 0}
                  onChange={(e) => setLoan({ ...loan, vehicle_value: Number(e.target.value) })}
                />
              </label>
            )}
          </div>
        </section>
      </div>

      <div className="mt-4 flex gap-2">
        <button onClick={score} disabled={loading} className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50">
          {loading ? "Scoring..." : "Score (HTTP)"}
        </button>
        <button onClick={liveScore} className="px-4 py-2 rounded border">
          Live Score (WebSocket)
        </button>
      </div>

      {error && <p className="mt-4 text-red-600">{error}</p>}

      {result && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border rounded p-4">
            <h3 className="font-medium mb-2">Decision</h3>
            <p><span className="font-medium">Decision:</span> {result.decision}</p>
            <p><span className="font-medium">PD:</span> {(result.probability_of_default * 100).toFixed(2)}%</p>
            <h4 className="mt-3 font-medium">Reasons</h4>
            <ul className="list-disc ml-6 text-sm">
              {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
            </ul>
          </div>
          <div className="border rounded p-4">
            <h3 className="font-medium mb-2">Rule Flags</h3>
            <ul className="text-sm">
              {Object.entries(result.rule_flags).map(([k, v]) => (
                <li key={k}><span className="font-medium">{k}</span>: {v ? "OK" : "FAIL"}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {(liveLog.length > 0 || liveFinal) && (
        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border rounded p-4">
            <h3 className="font-medium mb-2">Live Stream</h3>
            <ul className="text-xs space-y-1">
              {liveLog.map((m, i) => <li key={i}>{m}</li>)}
            </ul>
          </div>
          {liveFinal && (
            <div className="border rounded p-4">
              <h3 className="font-medium mb-2">Final</h3>
              <p><span className="font-medium">Decision:</span> {liveFinal.decision}</p>
              <p><span className="font-medium">PD:</span> {(liveFinal.probability_of_default * 100).toFixed(2)}%</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
