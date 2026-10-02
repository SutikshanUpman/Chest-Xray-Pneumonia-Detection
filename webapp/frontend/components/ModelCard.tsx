// One card per model on the results page: label, probability bar, threshold and inference time (or the error).
import { formatMs, formatThreshold } from "@/lib/format";
import type { ScanResultDto } from "@/lib/types";
import ProbabilityBar from "./ProbabilityBar";

export default function ModelCard({ result }: { result: ScanResultDto }) {
  const failed = result.status === "error" || result.probPneumonia === null;
  const isPneumonia = result.label === "PNEUMONIA";

  return (
    <article className="rounded border border-slate-700 bg-slate-800 p-4">
      <h3 className="font-semibold">{result.displayName}</h3>
      <p className="text-sm text-slate-300">
        {result.family}
        {result.kind === "ensemble" && ` · mean of ${result.memberCount} models`}
      </p>

      {failed ? (
        <p role="alert" className="mt-3 rounded border border-red-500 bg-red-950 p-3 text-sm">
          This model failed on this image. {result.error}
        </p>
      ) : (
        <>
          <p className={`mt-3 text-lg font-bold ${isPneumonia ? "text-red-400" : "text-emerald-400"}`}>
            {isPneumonia ? "Pneumonia" : "Normal"}
          </p>
          <div className="mt-3">
            <ProbabilityBar probability={result.probPneumonia!} threshold={result.threshold} isPneumonia={isPneumonia} />
          </div>
          <p className="mt-3 text-sm text-slate-300">
            Threshold used: {formatThreshold(result.threshold)}
            {result.thresholdSource === "default" && " (uncalibrated threshold)"} · Inference time:{" "}
            {formatMs(result.inferenceMs)}
          </p>
        </>
      )}
    </article>
  );
}
