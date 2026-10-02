// A horizontal bar for P(pneumonia) with a small mark at the model's threshold, plus the same numbers as text.
import { formatPercent, formatThreshold } from "@/lib/format";

export default function ProbabilityBar({
  probability,
  threshold,
  isPneumonia,
}: {
  probability: number;
  threshold: number | null;
  isPneumonia: boolean;
}) {
  const text = `P(pneumonia): ${formatPercent(probability)}  |  threshold: ${formatThreshold(threshold)}`;
  return (
    <div>
      <div
        role="meter"
        aria-label="Probability of pneumonia"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={probability}
        aria-valuetext={text}
        className="relative h-3 rounded bg-slate-700"
      >
        <div
          className={`h-3 rounded ${isPneumonia ? "bg-red-400" : "bg-emerald-400"}`}
          style={{ width: `${probability * 100}%` }}
        />
        {threshold !== null && (
          <div className="absolute -top-1 h-5 w-0.5 bg-white" style={{ left: `${threshold * 100}%` }} />
        )}
      </div>
      <p className="mt-2 text-sm text-slate-300">{text}</p>
    </div>
  );
}
