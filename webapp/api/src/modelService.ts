import { ApiError } from "./errors.js";
import { isModelId } from "./registry.js";
import type { ImageMime } from "./validation.js";

// Response shapes of webapp/model-service (snake_case, see app/schemas.py)

export interface MsInterval {
  low: number;
  high: number;
}

export interface MsBenchmark {
  macro_f1: number | null;
  val_macro_f1: number | null;
  accuracy: number | null;
  recall_normal: number | null;
  recall_pneumonia: number | null;
  roc_auc: number | null;
  ci: {
    accuracy: MsInterval | null;
    recall_normal: MsInterval | null;
    recall_pneumonia: MsInterval | null;
    roc_auc: MsInterval | null;
  } | null;
}

export type ThresholdSource = "comparison_csv" | "default";

export interface MsModel {
  id: string;
  display_name: string;
  eval_name: string;
  family: string;
  kind: "single" | "ensemble";
  member_count: number;
  members: { file: string; exists: boolean }[];
  available: boolean;
  color_mode: "grayscale" | "rgb";
  interpolation: string;
  rescale: "1/255" | "in_model";
  gradcam_supported: boolean;
  threshold: number;
  threshold_source: ThresholdSource;
  uncalibrated: boolean;
  threshold_stale: boolean;
  benchmark: MsBenchmark | null;
}

export interface MsCsvStatus {
  found: boolean;
  mtime: string | null;
  error: string | null;
}

export interface MsModels {
  mock: boolean;
  mock_all_available: boolean;
  total: number;
  available: number;
  img_size: [number, number];
  comparison_csv: MsCsvStatus;
  comparison_ci_csv: MsCsvStatus;
  models: MsModel[];
}

export interface MsPredictResult {
  id: string;
  display_name: string;
  family: string;
  kind: "single" | "ensemble";
  member_count: number;
  status: "ok" | "error";
  prob_pneumonia: number | null;
  member_probs: number[];
  threshold: number;
  threshold_source: ThresholdSource;
  uncalibrated: boolean;
  threshold_stale: boolean;
  label: "NORMAL" | "PNEUMONIA" | null;
  inference_ms: number | null;
  cold_start: boolean;
  heatmap_png_b64: string | null;
  heatmap_synthetic: boolean;
  heatmap_note: "not_supported" | "not_requested" | "failed" | null;
  error: string | null;
}

export interface MsPredict {
  mock: boolean;
  image: {
    width: number;
    height: number;
    format: "JPEG" | "PNG";
    mime_type: ImageMime;
    size_bytes: number;
    sha256: string;
  };
  results: MsPredictResult[];
  skipped: { id: string; display_name: string; reason: "not_trained" }[];
}

export interface MsHealth {
  status: "ok";
  mock: boolean;
  mock_all_available: boolean;
  tensorflow: string | null;
  loaded_models: string[];
  models_dir_found: boolean;
  results_dir_found: boolean;
}

export interface PredictRequest {
  image: Buffer;
  mime: ImageMime;
  modelIds: "all" | string[];
  gradcam: boolean;
}

// ---- minimal runtime checks: a malformed answer becomes a 502, never a bad DB row

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const isNumOrNull = (v: unknown) => v === null || isNum(v);
const isStr = (v: unknown): v is string => typeof v === "string";
const isBool = (v: unknown): v is boolean => typeof v === "boolean";

function badResponse(what: string): ApiError {
  return new ApiError(502, "MODEL_SERVICE_ERROR", "Unexpected response from the model service.", { invalid: what });
}

function checkModels(body: unknown): MsModels {
  if (!isObj(body) || !isBool(body.mock) || !isNum(body.total) || !isNum(body.available)) throw badResponse("models");
  if (!isObj(body.comparison_csv) || !isObj(body.comparison_ci_csv) || !Array.isArray(body.models)) {
    throw badResponse("models");
  }
  for (const m of body.models) {
    if (!isObj(m) || !isModelId(m.id) || !isStr(m.display_name) || !isBool(m.available) || !isNum(m.threshold)) {
      throw badResponse("models[]");
    }
  }
  return body as unknown as MsModels;
}

function checkResult(r: unknown): void {
  if (!isObj(r) || !isModelId(r.id) || !isStr(r.display_name) || !isStr(r.family)) throw badResponse("results[]");
  if (r.kind !== "single" && r.kind !== "ensemble") throw badResponse("results[].kind");
  if (!isNum(r.member_count) || !isNum(r.threshold) || !isBool(r.cold_start)) throw badResponse("results[]");
  if (r.threshold_source !== "comparison_csv" && r.threshold_source !== "default") {
    throw badResponse("results[].threshold_source");
  }
  if (!isNumOrNull(r.inference_ms) || !(r.heatmap_png_b64 === null || isStr(r.heatmap_png_b64))) {
    throw badResponse("results[]");
  }
  if (r.status === "ok") {
    if (!isNum(r.prob_pneumonia) || r.prob_pneumonia < 0 || r.prob_pneumonia > 1) throw badResponse("results[].prob");
    if (r.label !== "NORMAL" && r.label !== "PNEUMONIA") throw badResponse("results[].label");
  } else if (r.status !== "error" || !isStr(r.error)) {
    throw badResponse("results[].status");
  }
}

function checkPredict(body: unknown): MsPredict {
  if (!isObj(body) || !isBool(body.mock) || !isObj(body.image) || !Array.isArray(body.results)) {
    throw badResponse("predict");
  }
  const img = body.image;
  if (!isNum(img.width) || !isNum(img.height) || !isStr(img.sha256) || !/^[0-9a-f]{64}$/.test(img.sha256)) {
    throw badResponse("predict.image");
  }
  if (body.results.length === 0) throw badResponse("predict.results");
  body.results.forEach(checkResult);
  const ids = body.results.map((r) => (r as Obj).id);
  if (new Set(ids).size !== ids.length) throw badResponse("predict.results (duplicate id)");
  return body as unknown as MsPredict;
}

// ---- error mapping

function fromFetchError(err: unknown): ApiError {
  const name = (err as { name?: string } | null)?.name;
  if (name === "TimeoutError" || name === "AbortError") {
    return new ApiError(504, "MODEL_SERVICE_TIMEOUT", "The model service did not answer in time.");
  }
  const code = (err as { cause?: { code?: string } } | null)?.cause?.code;
  if (code === "ECONNREFUSED" || code === "ENOTFOUND" || code === "EAI_AGAIN") {
    return new ApiError(503, "MODEL_SERVICE_DOWN", "The model service is not running.");
  }
  return new ApiError(502, "MODEL_SERVICE_ERROR", "Could not reach the model service.", { cause: code ?? name });
}

function fromServiceError(status: number, body: unknown): ApiError {
  const error = isObj(body) && isObj(body.error) ? body.error : null;
  const code = error && isStr(error.code) ? error.code : null;
  const message = error && isStr(error.message) ? error.message : "Model service error.";
  if (status === 409 && code) {
    // MODEL_UNAVAILABLE / NO_MODELS_AVAILABLE from the service
    return new ApiError(409, "MODELS_UNAVAILABLE", message, error?.details ?? { reason: code });
  }
  if ((status === 400 || status === 413 || status === 415) && code) {
    return new ApiError(status, code, message, error?.details ?? undefined);
  }
  return new ApiError(502, "MODEL_SERVICE_ERROR", "The model service failed.", { status, code });
}

export class ModelServiceClient {
  constructor(
    private readonly baseUrl: string,
    private readonly timeoutMs: number,
  ) {}

  async health(timeoutMs = Math.min(this.timeoutMs, 5000)): Promise<MsHealth> {
    const body = await this.request("/health", { method: "GET" }, timeoutMs);
    if (!isObj(body) || body.status !== "ok" || !isBool(body.mock)) throw badResponse("health");
    return body as unknown as MsHealth;
  }

  async models(): Promise<MsModels> {
    return checkModels(await this.request("/models", { method: "GET" }, Math.min(this.timeoutMs, 15_000)));
  }

  async predict(req: PredictRequest): Promise<MsPredict> {
    const form = new FormData();
    const ext = req.mime === "image/png" ? "png" : "jpg";
    form.append("file", new Blob([new Uint8Array(req.image)], { type: req.mime }), `upload.${ext}`);
    form.append("model_ids", req.modelIds === "all" ? "all" : req.modelIds.join(","));
    form.append("gradcam", req.gradcam ? "true" : "false");
    return checkPredict(await this.request("/predict", { method: "POST", body: form }, this.timeoutMs));
  }

  private async request(pathname: string, init: RequestInit, timeoutMs: number): Promise<unknown> {
    let status: number;
    let text: string;
    try {
      const res = await fetch(this.baseUrl + pathname, { ...init, signal: AbortSignal.timeout(timeoutMs) });
      status = res.status;
      text = await res.text();
    } catch (err) {
      throw fromFetchError(err);
    }
    let body: unknown;
    try {
      body = JSON.parse(text);
    } catch {
      throw new ApiError(502, "MODEL_SERVICE_ERROR", "The model service returned invalid JSON.", { status });
    }
    if (status < 200 || status >= 300) throw fromServiceError(status, body);
    return body;
  }
}
