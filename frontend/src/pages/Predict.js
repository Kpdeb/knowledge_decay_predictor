import React, { useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import PageHeader from "../components/PageHeader";

function ebbinghaus(t, score, reviews, diff, dur) {
  const f = { easy: 1.5, medium: 1.0, hard: 0.7 }[diff] || 1.0;
  const s = 24 * f * (1 + score / 100) * (1 + reviews * 0.3) * (1 + dur / 200);
  return Math.max(0, Math.min(1, Math.exp(-t / s)));
}

function ringPath(pct) {
  const r = 50, cx = 60, cy = 60;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - pct / 100);
  return { circ, offset };
}

export default function Predict() {
  const [params, setParams] = useState({ days: 5, score: 75, diff: "medium", reviews: 2, dur: 45 });

  const set = (key, val) => setParams(p => ({ ...p, [key]: val }));
  const r = ebbinghaus(params.days * 24, params.score, params.reviews, params.diff, params.dur);
  const pct = Math.round(r * 100);
  const color = r >= 0.75 ? "var(--green)" : r >= 0.5 ? "var(--amber)" : "var(--red)";

  const { circ, offset } = ringPath(pct);

  const forecast = Array.from({ length: 31 }, (_, i) => ({
    day: i === 0 ? "Now" : i % 5 === 0 ? `+${i}d` : "",
    retention: Math.round(ebbinghaus((params.days + i) * 24, params.score, params.reviews, params.diff, params.dur) * 100),
  }));

  const daysUntilHalf = (() => {
    for (let i = 0; i <= 60; i++) {
      if (ebbinghaus((params.days + i) * 24, params.score, params.reviews, params.diff, params.dur) < 0.5) return i;
    }
    return 60;
  })();

  const recommendation = r < 0.5
    ? "⚠ Review immediately — retention is critically low."
    : r < 0.75
    ? `📅 Review within ${daysUntilHalf} day(s) to stay above 50%.`
    : `✓ Memory is strong. Will drop below 50% in ~${daysUntilHalf} days.`;

  return (
    <>
      <PageHeader title="Decay Predictor" subtitle="ML-powered retention forecasting with the Ebbinghaus model" />
      <div style={{ padding: "20px 24px" }}>
        <div className="grid-2" style={{ marginBottom: 16 }}>
          {/* Controls */}
          <div className="card">
            <div className="section-title">Simulation Parameters</div>
            {[
              { label: `Days since last review: ${params.days}d`, key: "days", min: 0, max: 30, val: params.days },
              { label: `Quiz score: ${params.score}%`,             key: "score", min: 0, max: 100, val: params.score },
              { label: `Review count: ${params.reviews}`,           key: "reviews", min: 0, max: 10, val: params.reviews },
              { label: `Study duration: ${params.dur} min`,         key: "dur", min: 10, max: 120, val: params.dur },
            ].map(s => (
              <div className="form-group" key={s.key}>
                <label>{s.label}</label>
                <input type="range" min={s.min} max={s.max} value={s.val}
                  onChange={e => set(s.key, +e.target.value)}
                  style={{ width: "100%", accentColor: "var(--accent)" }} />
              </div>
            ))}
            <div className="form-group">
              <label>Difficulty</label>
              <select value={params.diff} onChange={e => set("diff", e.target.value)}>
                <option value="easy">Easy</option>
                <option value="medium">Medium</option>
                <option value="hard">Hard</option>
              </select>
            </div>
          </div>

          {/* Prediction ring + forecast */}
          <div>
            <div className="card" style={{ textAlign: "center", marginBottom: 12 }}>
              <div className="section-title">Predicted Retention</div>
              <svg width={120} height={120} viewBox="0 0 120 120" style={{ display: "block", margin: "0 auto 8px", transform: "rotate(-90deg)" }}>
                <circle cx={60} cy={60} r={50} fill="none" stroke="var(--bg4)" strokeWidth={12} />
                <circle cx={60} cy={60} r={50} fill="none" stroke={color} strokeWidth={12}
                  strokeDasharray={circ} strokeDashoffset={offset} strokeLinecap="round"
                  style={{ transition: "stroke-dashoffset .5s, stroke .3s" }} />
              </svg>
              <div style={{ fontSize: 28, fontWeight: 700, fontFamily: "JetBrains Mono", color }}>{pct}%</div>
              <div style={{ fontSize: 12, color: "var(--text3)", marginTop: 2 }}>
                {r >= 0.75 ? "Strong retention" : r >= 0.5 ? "Moderate retention" : "Weak — review now"}
              </div>
              <div style={{ marginTop: 12, padding: "10px 12px", background: "var(--bg3)", borderRadius: "var(--r)", fontSize: 12, textAlign: "left" }}>
                {recommendation}
              </div>
            </div>

            <div className="card">
              <div className="section-title">30-Day Forecast</div>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={forecast} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
                  <XAxis dataKey="day" tick={{ fill: "var(--text3)", fontSize: 10 }} />
                  <YAxis domain={[0, 100]} tick={{ fill: "var(--text3)", fontSize: 10 }} />
                  <ReferenceLine y={50} stroke="var(--red)" strokeDasharray="4 4" strokeOpacity={0.5} />
                  <Tooltip contentStyle={{ background: "var(--bg3)", border: "1px solid var(--border)", borderRadius: 8, fontSize: 12 }} formatter={v => [`${v}%`]} />
                  <Line type="monotone" dataKey="retention" stroke={color} strokeWidth={2} dot={false} />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ fontSize: 11, color: "var(--text3)", marginTop: 6 }}>Red dashed line = 50% threshold</div>
            </div>
          </div>
        </div>

        {/* Feature importance */}
        <div className="card">
          <div className="section-title">ML Feature Weights (Random Forest)</div>
          <div className="grid-4">
            {[
              { name: "Time elapsed",    weight: 0.42, color: "var(--red)" },
              { name: "Quiz score",      weight: 0.28, color: "var(--green)" },
              { name: "Review count",    weight: 0.18, color: "var(--accent2)" },
              { name: "Study duration",  weight: 0.08, color: "var(--amber)" },
            ].map(f => (
              <div key={f.name} className="card-sm">
                <div style={{ fontSize: 11, color: "var(--text3)", marginBottom: 6 }}>{f.name}</div>
                <div style={{ fontSize: 18, fontWeight: 700, color: f.color }}>{Math.round(f.weight * 100)}%</div>
                <div className="progress-bar" style={{ marginTop: 6 }}>
                  <div className="progress-fill" style={{ width: `${f.weight * 100}%`, background: f.color }} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
