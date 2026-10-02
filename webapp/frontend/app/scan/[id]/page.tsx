"use client";

// Results page for one scan: consensus, one card per model, the Grad-CAM viewer and (for a selection) the benchmark table.
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import BenchmarkTable from "@/components/BenchmarkTable";
import ErrorBox from "@/components/ErrorBox";
import HeatmapViewer from "@/components/HeatmapViewer";
import ModelCard from "@/components/ModelCard";
import { getScan } from "@/lib/api";
import { canRetry, friendlyMessage } from "@/lib/errors";
import type { ScanDetailDto } from "@/lib/types";

export default function ScanPage() {
  const { id } = useParams<{ id: string }>();
  const [scan, setScan] = useState<ScanDetailDto | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    getScan(id).then(setScan, setError);
  }, [id, reloads]);

  function retry() {
    setError(null);
    setReloads((n) => n + 1);
  }

  if (error) {
    return <ErrorBox message={friendlyMessage(error)} onRetry={canRetry(error) ? retry : undefined} />;
  }
  if (!scan) return <p role="status">Loading scan…</p>;

  const { pneumonia, normal, failed } = scan.consensus;
  // Only the Phase 2 fine-tuned model makes a heatmap
  const heatmapUrl = scan.results.find((r) => r.modelId === "phase2_stage2")?.heatmapUrl ?? null;

  return (
    <div className="space-y-6">
      {scan.mock && (
        <p role="note" className="rounded bg-yellow-400 p-3 font-bold text-black">
          MOCK DATA: this scan was made in mock mode. Probabilities and heatmap are fake.
        </p>
      )}

      <div>
        <Link href="/history" className="text-blue-400 hover:underline">
          ← History
        </Link>
        <h1 className="mt-2 text-3xl font-bold">Scan results</h1>
        <p className="text-sm text-slate-300">
          {new Date(scan.createdAt).toLocaleString()} · {scan.mode === "all" ? "All models" : "Comparative analysis"} ·{" "}
          {scan.image.filename}
        </p>
      </div>

      <p className="rounded border border-slate-700 bg-slate-800 p-4">
        <strong>
          {pneumonia} pneumonia, {normal} normal, {failed} failed
        </strong>{" "}
        (each model uses its own threshold). This is not a clinical result.
      </p>
      <p className="font-semibold">Educational demo, not a medical device. Not for diagnosis.</p>

      <HeatmapViewer
        imageUrl={scan.image.url}
        heatmapUrl={heatmapUrl}
        width={scan.image.width}
        height={scan.image.height}
        mock={scan.mock}
      />

      <div className="grid gap-4 sm:grid-cols-2">
        {scan.results.map((r) => (
          <ModelCard key={r.modelId} result={r} />
        ))}
      </div>

      {scan.mode === "selected" && <BenchmarkTable benchmark={scan.benchmark} />}
    </div>
  );
}
