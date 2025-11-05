"use client";

import { useEffect, useState } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

type Policy = {
	min_age: Record<string, number>;
	max_age: Record<string, number>;
	max_dbr: Record<string, number>;
	auto: { min_down: number; max_ltv: number };
	housing: { max_ltv: number };
	tenor_limits: Record<string, number>;
};

export default function AdminPage() {
	const [policy, setPolicy] = useState<Policy | null>(null);
	const [saving, setSaving] = useState(false);
	const [error, setError] = useState<string | null>(null);

	useEffect(() => {
		fetch(`${API_BASE}/api/v1/config`)
			.then((r) => r.json())
			.then((d) => setPolicy(d.policy))
			.catch((e) => setError(e?.message || "Failed to load policy"));
	}, []);

	function updateNested(path: (string | number)[], value: number) {
		setPolicy((p) => {
			if (!p) return p;
			const clone: any = JSON.parse(JSON.stringify(p));
			let ref: any = clone;
			for (let i = 0; i < path.length - 1; i++) ref = ref[path[i] as any];
			ref[path[path.length - 1] as any] = value;
			return clone as Policy;
		});
	}

	async function save() {
		if (!policy) return;
		setSaving(true);
		setError(null);
		try {
			const res = await fetch(`${API_BASE}/api/v1/config`, {
				method: "PUT",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify(policy),
			});
			if (!res.ok) throw new Error(`Save failed: ${res.status}`);
		} catch (e: any) {
			setError(e?.message || "Save failed");
		} finally {
			setSaving(false);
		}
	}

	const products = ["personal", "cash", "car", "housing"];

	return (
		<div className="max-w-4xl mx-auto p-6">
			<h1 className="text-2xl font-semibold mb-4">Admin: Policy Configuration</h1>
			{error && <p className="text-red-600 mb-2">{error}</p>}
			{!policy ? (
				<p>Loading…</p>
			) : (
				<div className="space-y-6">
					<section className="border rounded p-4">
						<h2 className="font-medium mb-2">Age Limits</h2>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
							{products.map((p) => (
								<div key={`age-${p}`} className="space-y-1">
									<label className="block">Min Age ({p})</label>
									<input type="number" className="w-full border rounded p-1" value={policy.min_age[p]}
										onChange={(e) => updateNested(["min_age", p], Number(e.target.value))}
									/>
									<label className="block">Max Age ({p})</label>
									<input type="number" className="w-full border rounded p-1" value={policy.max_age[p]}
										onChange={(e) => updateNested(["max_age", p], Number(e.target.value))}
									/>
								</div>
							))}
						</div>
					</section>

					<section className="border rounded p-4">
						<h2 className="font-medium mb-2">Debt Burden Ratio (DBR) Caps</h2>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
							{products.map((p) => (
								<div key={`dbr-${p}`} className="space-y-1">
									<label className="block">Max DBR ({p})</label>
									<input type="number" step="0.01" className="w-full border rounded p-1" value={policy.max_dbr[p]}
										onChange={(e) => updateNested(["max_dbr", p], Number(e.target.value))}
									/>
								</div>
							))}
						</div>
					</section>

					<section className="border rounded p-4">
						<h2 className="font-medium mb-2">Secured Products</h2>
						<div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
							<div>
								<h3 className="font-medium">Auto</h3>
								<label className="block">Min Down Payment</label>
								<input type="number" step="0.01" className="w-full border rounded p-1" value={policy.auto.min_down}
									onChange={(e) => updateNested(["auto", "min_down"], Number(e.target.value))}
								/>
								<label className="block mt-2">Max LTV</label>
								<input type="number" step="0.01" className="w-full border rounded p-1" value={policy.auto.max_ltv}
									onChange={(e) => updateNested(["auto", "max_ltv"], Number(e.target.value))}
								/>
							</div>
							<div>
								<h3 className="font-medium">Housing</h3>
								<label className="block">Max LTV</label>
								<input type="number" step="0.01" className="w-full border rounded p-1" value={policy.housing.max_ltv}
									onChange={(e) => updateNested(["housing", "max_ltv"], Number(e.target.value))}
								/>
							</div>
						</div>
					</section>

					<section className="border rounded p-4">
						<h2 className="font-medium mb-2">Tenor Limits (months)</h2>
						<div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
							{products.map((p) => (
								<div key={`tenor-${p}`} className="space-y-1">
									<label className="block">{p}</label>
									<input type="number" className="w-full border rounded p-1" value={policy.tenor_limits[p]}
										onChange={(e) => updateNested(["tenor_limits", p], Number(e.target.value))}
									/>
								</div>
							))}
						</div>
					</section>

					<div className="flex gap-2">
						<button onClick={save} disabled={saving} className="px-4 py-2 rounded bg-blue-600 text-white disabled:opacity-50">
							{saving ? "Saving…" : "Save Policy"}
						</button>
					</div>
				</div>
			)}
		</div>
	);
}
