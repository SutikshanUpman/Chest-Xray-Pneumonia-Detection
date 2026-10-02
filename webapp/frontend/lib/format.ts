// Small functions that turn numbers from the API into text. They only round; they never invent values.

const DASH = "—"; // shown when a value is missing

// 0.7234 -> "72.3%"
export function formatPercent(p: number | null): string {
  return p === null ? DASH : `${(p * 100).toFixed(1)}%`;
}

// Thresholds come from a 0.01 grid, so 0.3900000000000002 -> "0.39"
export function formatThreshold(t: number | null): string {
  return t === null ? DASH : t.toFixed(2);
}

// Benchmark numbers: 4 decimals, exactly as 03_evaluation writes them to results/*.csv
export function formatMetric(v: number | null): string {
  return v === null ? DASH : v.toFixed(4);
}

export function formatInterval(ci: { low: number; high: number } | null): string {
  return ci === null ? DASH : `[${ci.low.toFixed(4)}, ${ci.high.toFixed(4)}]`;
}

export function formatMs(ms: number | null): string {
  if (ms === null) return DASH;
  return ms < 1 ? "<1 ms" : `${Math.round(ms)} ms`;
}
