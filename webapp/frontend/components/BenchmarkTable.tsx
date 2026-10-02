// Plain table with the benchmark numbers of the chosen models. The numbers come from results/comparison*.csv via the API.
import { formatInterval, formatMetric } from "@/lib/format";
import type { BenchmarkTableDto, MetricWithCi } from "@/lib/types";

function Cell({ metric }: { metric: MetricWithCi }) {
  return (
    <td className="p-2">
      {formatMetric(metric.value)}
      <span className="block text-xs text-slate-300">95% CI {formatInterval(metric.ci)}</span>
    </td>
  );
}

export default function BenchmarkTable({ benchmark }: { benchmark: BenchmarkTableDto | null }) {
  if (benchmark === null) return <p className="text-slate-300">Benchmark: not available yet.</p>;

  return (
    <div className="overflow-x-auto">
      <table className="w-full border border-slate-700 text-left text-sm">
        <caption className="p-2 text-left text-slate-300">Benchmarks from results/comparison.csv</caption>
        <thead className="bg-slate-800">
          <tr>
            <th className="p-2">Model</th>
            <th className="p-2">Test macro F1</th>
            <th className="p-2">Normal recall</th>
            <th className="p-2">Pneumonia recall</th>
            <th className="p-2">ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {benchmark.rows.map((row) => (
            <tr key={row.modelId} className="border-t border-slate-700">
              <th className="p-2 font-medium">{row.displayName}</th>
              <td className="p-2">{formatMetric(row.macroF1)}</td>
              <Cell metric={row.recallNormal} />
              <Cell metric={row.recallPneumonia} />
              <Cell metric={row.rocAuc} />
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
