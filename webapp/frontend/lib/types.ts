// JSON shapes returned by the Express API (camelCase).
// The first block copies the types in webapp/api/src/dto.ts: change both together.

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

// ---- Shapes the API builds inline (webapp/api/src/routes/scans.ts, errors.ts), not in dto.ts

/** GET /api/scans */
export interface ScanListDto {
  items: ScanSummaryDto[];
  nextCursor: string | null;
}

/** Every error response: {error: {code, message, details?}} */
export interface ApiErrorBody {
  error: { code: string; message: string; details?: unknown };
}

export type ScanMode = ScanDetailDto["mode"];
export type OnlineModels = Extract<ModelsDto, { serviceOnline: true }>;
