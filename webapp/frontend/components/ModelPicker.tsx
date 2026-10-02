"use client";

// Checkbox list for "Comparative analysis". Models that are not trained yet cannot be ticked.
import { useState } from "react";
import { formatMetric } from "@/lib/format";
import type { ModelDto } from "@/lib/types";

export default function ModelPicker({ models, onRun }: { models: ModelDto[]; onRun: (ids: string[]) => void }) {
  const [chosen, setChosen] = useState<string[]>([]);

  function toggle(id: string) {
    setChosen((old) => (old.includes(id) ? old.filter((x) => x !== id) : [...old, id]));
  }

  return (
    <fieldset className="mt-6 rounded border border-slate-700 bg-slate-800 p-4">
      <legend className="px-2 font-medium">Choose one or more models</legend>
      <ul className="space-y-2">
        {models.map((m) => (
          <li key={m.id}>
            <label className={`flex items-start gap-3 ${m.available ? "cursor-pointer" : "opacity-50"}`}>
              <input
                type="checkbox"
                className="mt-1 h-4 w-4"
                disabled={!m.available}
                checked={chosen.includes(m.id)}
                onChange={() => toggle(m.id)}
              />
              <span>
                {m.displayName} <span className="text-sm text-slate-300">({m.family})</span>
                <span className="block text-sm text-slate-300">
                  {m.available ? `Test macro F1: ${formatMetric(m.benchmark?.macroF1 ?? null)}` : "not trained yet"}
                  {m.available && m.thresholdSource === "default" && " · uncalibrated threshold"}
                </span>
              </span>
            </label>
          </li>
        ))}
      </ul>
      <button
        onClick={() => onRun(chosen)}
        disabled={chosen.length === 0}
        className="mt-4 rounded bg-blue-600 px-4 py-2 font-medium hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
      >
        Run selected models
      </button>
    </fieldset>
  );
}
