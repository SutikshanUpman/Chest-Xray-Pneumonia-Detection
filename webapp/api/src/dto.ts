/** camelCase JSON returned to the frontend, and the mappers from DB rows / model-service data. */
import type { MsBenchmark, MsCsvStatus, MsInterval, MsModel, MsModels } from "./modelService.js";
import { GRADCAM_MODEL_IDS } from "./registry.js";
import type { ResultRow, ScanRow } from "./repo.js";

export interface Interval {
  low: number;
  high: number;
}

export interface BenchmarkDto {
  macroF1: number | null; // test set, no CI (03_evaluation does not bootstrap it)
  valMacroF1: number | null;
  accuracy: number | null;
  recallNormal: number | null;
  recallPneumonia: number | null;
  rocAuc: number | null;
  ci: {
    accuracy: Interval | null;
    recallNormal: Interval | null;
    recallPneumonia: Interval | null;
    rocAuc: Interval | null;
  } | null;
}

export interface ModelDto {
  id: string;
  displayName: string;
  evalName: string;
  family: string;
  kind: "single" | "ensemble";
  memberCount: number;
  members: { file: string; exists: boolean }[];
  available: boolean;
  colorMode: "grayscale" | "rgb";
  interpolation: string;
  rescale: "1/255" | "in_model";
  gradcamSupported: boolean;
  threshold: number;
  thresholdSource: "comparison_csv" | "default";
  uncalibrated: boolean;
  thresholdStale: boolean;
  benchmark: BenchmarkDto | null;
}

export interface CsvStatusDto {
  found: boolean;
  mtime: string | null;
  error: string | null;
}

export type ModelsDto =
  | {
      serviceOnline: true;
      mock: boolean;
      mockAllAvailable: boolean;
      total: number;
      available: number;
      imgSize: [number, number];
      comparisonCsv: CsvStatusDto;
      comparisonCiCsv: CsvStatusDto;
      models: ModelDto[];
    }
  | { serviceOnline: false; error: { code: string; message: string } };

export interface ConsensusDto {
  pneumonia: number;
  normal: number;
  failed: number;
  total: number;
}

export interface ScanResultDto {
  modelId: string;
  displayName: string;
  family: string;
  kind: "single" | "ensemble";
  memberCount: number;
  status: "ok" | "error";
  probPneumonia: number | null;
  threshold: number | null;
  thresholdSource: "comparison_csv" | "default" | null;
  uncalibrated: boolean;
  label: "NORMAL" | "PNEUMONIA" | null;
  inferenceMs: number | null;
  coldStart: boolean;
  heatmapUrl: string | null;
  heatmapNote: "not_supported" | "failed" | null;
  error: string | null;
}

export interface MetricWithCi {
  value: number | null;
  ci: Interval | null;
}

export interface BenchmarkTableDto {
  ciAvailable: boolean;
  rows: {
    modelId: string;
    displayName: string;
    macroF1: number | null;
    recallNormal: MetricWithCi;
    recallPneumonia: MetricWithCi;
    rocAuc: MetricWithCi;
  }[];
}

export interface ScanDetailDto {
  id: string;
  createdAt: string;
  mode: "all" | "selected";
  mock: boolean;
  image: {
    url: string;
    thumbUrl: string;
    width: number;
    height: number;
    filename: string;
    mimeType: string;
    sizeBytes: number;
  };
  modelsAvailable: number;
  modelsTotal: number;
  consensus: ConsensusDto;
  results: ScanResultDto[];
  benchmark: BenchmarkTableDto | null;
}

export interface ScanSummaryDto {
  id: string;
  createdAt: string;
  mode: "all" | "selected";
  mock: boolean;
  thumbUrl: string;
  consensus: ConsensusDto;
  modelCount: number;
}

// ---- URLs (same origin; the frontend proxies /api to this server)

export const scanUrls = {
  image: (id: string) => `/api/scans/${id}/image`,
  thumbnail: (id: string) => `/api/scans/${id}/thumbnail`,
  heatmap: (id: string, modelId: string) => `/api/scans/${id}/heatmaps/${modelId}`,
};

const interval = (v: MsInterval | null | undefined): Interval | null => (v ? { low: v.low, high: v.high } : null);
const csvStatus = (s: MsCsvStatus): CsvStatusDto => ({ found: s.found, mtime: s.mtime, error: s.error });

function benchmarkDto(b: MsBenchmark | null): BenchmarkDto | null {
  if (!b) return null;
  return {
    macroF1: b.macro_f1,
    valMacroF1: b.val_macro_f1,
    accuracy: b.accuracy,
    recallNormal: b.recall_normal,
    recallPneumonia: b.recall_pneumonia,
    rocAuc: b.roc_auc,
    ci: b.ci
      ? {
          accuracy: interval(b.ci.accuracy),
          recallNormal: interval(b.ci.recall_normal),
          recallPneumonia: interval(b.ci.recall_pneumonia),
          rocAuc: interval(b.ci.roc_auc),
        }
      : null,
  };
}

function modelDto(m: MsModel): ModelDto {
  return {
    id: m.id,
    displayName: m.display_name,
    evalName: m.eval_name,
    family: m.family,
    kind: m.kind,
    memberCount: m.member_count,
    members: m.members.map((x) => ({ file: x.file, exists: x.exists })),
    available: m.available,
    colorMode: m.color_mode,
    interpolation: m.interpolation,
    rescale: m.rescale,
    gradcamSupported: m.gradcam_supported,
    threshold: m.threshold,
    thresholdSource: m.threshold_source,
    uncalibrated: m.uncalibrated,
    thresholdStale: m.threshold_stale,
    benchmark: benchmarkDto(m.benchmark),
  };
}

export function modelsDto(ms: MsModels): ModelsDto {
  return {
    serviceOnline: true,
    mock: ms.mock,
    mockAllAvailable: ms.mock_all_available,
    total: ms.total,
    available: ms.available,
    imgSize: ms.img_size,
    comparisonCsv: csvStatus(ms.comparison_csv),
    comparisonCiCsv: csvStatus(ms.comparison_ci_csv),
    models: ms.models.map(modelDto),
  };
}

function consensus(scan: ScanRow): ConsensusDto {
  return { pneumonia: scan.n_pneumonia, normal: scan.n_normal, failed: scan.n_failed, total: scan.models_run };
}

function thumbUrl(scan: ScanRow): string {
  return scan.thumb_path ? scanUrls.thumbnail(scan.id) : scanUrls.image(scan.id);
}

function resultDto(scanId: string, r: ResultRow): ScanResultDto {
  // The API always asks for Grad-CAM, so a supported model without a heatmap means it failed.
  let heatmapNote: ScanResultDto["heatmapNote"] = null;
  if (r.status === "ok" && !r.heatmap_path) heatmapNote = GRADCAM_MODEL_IDS.has(r.model_id) ? "failed" : "not_supported";
  return {
    modelId: r.model_id,
    displayName: r.display_name,
    family: r.family,
    kind: r.kind,
    memberCount: r.member_count,
    status: r.status,
    probPneumonia: r.prob_pneumonia,
    threshold: r.threshold,
    thresholdSource: r.threshold_source,
    uncalibrated: r.threshold_source === "default",
    label: r.label,
    inferenceMs: r.inference_ms,
    coldStart: r.cold_start === 1,
    heatmapUrl: r.heatmap_path ? scanUrls.heatmap(scanId, r.model_id) : null,
    heatmapNote,
    error: r.error,
  };
}

/** Benchmark block for "selected" mode, built live from the model service (not stored). */
export function benchmarkTable(results: ResultRow[], catalog: MsModels | null): BenchmarkTableDto | null {
  if (!catalog || !catalog.comparison_csv.found) return null;
  const byId = new Map(catalog.models.map((m) => [m.id, m]));
  const rows: BenchmarkTableDto["rows"] = [];
  for (const r of results) {
    const b = byId.get(r.model_id)?.benchmark;
    if (!b) continue;
    rows.push({
      modelId: r.model_id,
      displayName: r.display_name,
      macroF1: b.macro_f1,
      recallNormal: { value: b.recall_normal, ci: interval(b.ci?.recall_normal) },
      recallPneumonia: { value: b.recall_pneumonia, ci: interval(b.ci?.recall_pneumonia) },
      rocAuc: { value: b.roc_auc, ci: interval(b.ci?.roc_auc) },
    });
  }
  return rows.length > 0 ? { ciAvailable: catalog.comparison_ci_csv.found, rows } : null;
}

export function scanDetailDto(scan: ScanRow, results: ResultRow[], catalog: MsModels | null): ScanDetailDto {
  return {
    id: scan.id,
    createdAt: scan.created_at,
    mode: scan.mode,
    mock: scan.mock === 1,
    image: {
      url: scanUrls.image(scan.id),
      thumbUrl: thumbUrl(scan),
      width: scan.width,
      height: scan.height,
      filename: scan.original_filename,
      mimeType: scan.mime_type,
      sizeBytes: scan.size_bytes,
    },
    modelsAvailable: scan.models_available,
    modelsTotal: scan.models_total,
    consensus: consensus(scan),
    results: results.map((r) => resultDto(scan.id, r)),
    benchmark: scan.mode === "selected" ? benchmarkTable(results, catalog) : null,
  };
}

export function scanSummaryDto(scan: ScanRow): ScanSummaryDto {
  return {
    id: scan.id,
    createdAt: scan.created_at,
    mode: scan.mode,
    mock: scan.mock === 1,
    thumbUrl: thumbUrl(scan),
    consensus: consensus(scan),
    modelCount: scan.models_run,
  };
}
