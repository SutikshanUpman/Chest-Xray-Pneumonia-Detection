/**
 * A fake of webapp/model-service for API tests (same JSON shapes, snake_case).
 * All numbers here are synthetic test values, not project results.
 */
import { createHash } from "node:crypto";
import http from "node:http";
import type { AddressInfo } from "node:net";
import { MODEL_IDS } from "../src/registry.js";
import { PNG_MAGIC } from "../src/validation.js";

export type PredictBehaviour = "ok" | "slow" | "http500" | "badjson" | "badshape" | "invalid415";

export interface FakeState {
  available: Record<string, boolean>;
  comparisonFound: boolean;
  ciFound: boolean;
  mock: boolean;
  predict: PredictBehaviour;
  slowMs: number;
  failModel: string | null; // this model comes back with status "error"
}

export interface FakeCall {
  path: string;
  modelIds?: string;
  gradcam?: string;
  fileSize?: number;
  fileType?: string;
}

export const FAKE_HEATMAP_PNG = Buffer.concat([PNG_MAGIC, Buffer.from("synthetic-heatmap")]);

// Synthetic probabilities: at threshold 0.5 or 0.6 -> P, N, P, P, N, P
export const FAKE_PROBS: Record<string, number> = {
  class_weights: 0.91,
  oversampling: 0.22,
  kfold_ensemble: 0.71,
  kfoldc_ensemble: 0.64,
  phase2_stage1: 0.35,
  phase2_stage2: 0.83,
};

const ENSEMBLES = new Set(["kfold_ensemble", "kfoldc_ensemble"]);
const displayName = (id: string) => `Fake ${id}`;

function threshold(state: FakeState) {
  return state.comparisonFound
    ? { threshold: 0.6, threshold_source: "comparison_csv", uncalibrated: false }
    : { threshold: 0.5, threshold_source: "default", uncalibrated: true };
}

function catalog(state: FakeState) {
  const ci = (v: number) => ({ low: v - 0.05, high: v + 0.05 });
  const models = MODEL_IDS.map((id) => ({
    id,
    display_name: displayName(id),
    eval_name: `Eval ${id}`,
    family: id.startsWith("phase2") ? "Phase 2 · fake" : "Phase 1 · fake",
    kind: ENSEMBLES.has(id) ? "ensemble" : "single",
    member_count: ENSEMBLES.has(id) ? 3 : 1,
    members: [{ file: `models/fake/${id}.h5`, exists: state.available[id] ?? false }],
    available: state.available[id] ?? false,
    color_mode: id.startsWith("phase2") ? "rgb" : "grayscale",
    interpolation: id.startsWith("phase2") ? "bilinear" : "nearest",
    rescale: id.startsWith("phase2") ? "in_model" : "1/255",
    gradcam_supported: id === "phase2_stage2",
    ...threshold(state),
    threshold_stale: false,
    benchmark: state.comparisonFound
      ? {
          macro_f1: 0.8,
          val_macro_f1: 0.82,
          accuracy: 0.85,
          recall_normal: 0.7,
          recall_pneumonia: 0.9,
          roc_auc: 0.92,
          ci: state.ciFound
            ? { accuracy: ci(0.85), recall_normal: ci(0.7), recall_pneumonia: ci(0.9), roc_auc: ci(0.92) }
            : null,
        }
      : null,
  }));
  const csv = (found: boolean) => ({ found, mtime: found ? "2026-01-01T00:00:00+00:00" : null, error: null });
  return {
    mock: state.mock,
    mock_all_available: true,
    total: models.length,
    available: models.filter((m) => m.available).length,
    img_size: [224, 224],
    comparison_csv: csv(state.comparisonFound),
    comparison_ci_csv: csv(state.ciFound),
    models,
  };
}

function predictResult(id: string, state: FakeState, gradcam: boolean) {
  const t = threshold(state);
  const failed = state.failModel === id;
  const prob = FAKE_PROBS[id] ?? 0.5;
  const heatmap = !failed && gradcam && id === "phase2_stage2";
  return {
    id,
    display_name: displayName(id),
    family: "fake",
    kind: ENSEMBLES.has(id) ? "ensemble" : "single",
    member_count: ENSEMBLES.has(id) ? 3 : 1,
    status: failed ? "error" : "ok",
    prob_pneumonia: failed ? null : prob,
    member_probs: failed ? [] : [prob],
    ...t,
    threshold_stale: false,
    label: failed ? null : prob > t.threshold ? "PNEUMONIA" : "NORMAL",
    inference_ms: failed ? null : 12.5,
    cold_start: false,
    heatmap_png_b64: heatmap ? FAKE_HEATMAP_PNG.toString("base64") : null,
    heatmap_synthetic: heatmap,
    heatmap_note: heatmap || failed ? null : id === "phase2_stage2" ? "not_requested" : "not_supported",
    error: failed ? "RuntimeError: synthetic failure" : null,
  };
}

function json(res: http.ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "content-type": "application/json" });
  res.end(JSON.stringify(body));
}

export interface FakeModelService {
  url: string;
  state: FakeState;
  calls: FakeCall[];
  close(): Promise<void>;
}

export async function startFakeModelService(overrides: Partial<FakeState> = {}): Promise<FakeModelService> {
  const state: FakeState = {
    available: Object.fromEntries(MODEL_IDS.map((id) => [id, true])),
    comparisonFound: false,
    ciFound: false,
    mock: true,
    predict: "ok",
    slowMs: 0,
    failModel: null,
    ...overrides,
  };
  const calls: FakeCall[] = [];

  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(chunk as Buffer);
    const body = Buffer.concat(chunks);

    if (req.method === "GET" && req.url === "/health") {
      calls.push({ path: "/health" });
      return json(res, 200, {
        status: "ok", mock: state.mock, mock_all_available: true, tensorflow: null, loaded_models: [],
        models_dir_found: true, results_dir_found: true,
      });
    }
    if (req.method === "GET" && req.url === "/models") {
      calls.push({ path: "/models" });
      return json(res, 200, catalog(state));
    }
    if (req.method === "POST" && req.url === "/predict") {
      const form = await new Request("http://fake/predict", {
        method: "POST",
        headers: { "content-type": req.headers["content-type"] ?? "" },
        body,
      }).formData();
      const file = form.get("file") as File | null;
      const modelIds = String(form.get("model_ids"));
      const gradcam = String(form.get("gradcam"));
      calls.push({ path: "/predict", modelIds, gradcam, fileSize: file?.size, fileType: file?.type });

      if (state.predict === "slow") await new Promise((r) => setTimeout(r, state.slowMs));
      if (state.predict === "http500") {
        res.writeHead(500, { "content-type": "text/plain" });
        return res.end("Internal Server Error");
      }
      if (state.predict === "badjson") {
        res.writeHead(200, { "content-type": "application/json" });
        return res.end("{not json");
      }
      if (state.predict === "badshape") return json(res, 200, { mock: true, results: "nope" });
      if (state.predict === "invalid415") {
        return json(res, 415, { error: { code: "INVALID_IMAGE", message: "The file could not be decoded as an image." } });
      }

      const requested = modelIds === "all" ? MODEL_IDS.filter((id) => state.available[id]) : modelIds.split(",");
      const unavailable = requested.filter((id) => !state.available[id]);
      if (modelIds !== "all" && unavailable.length > 0) {
        return json(res, 409, {
          error: { code: "MODEL_UNAVAILABLE", message: "Selected model is not trained yet.", details: { unavailable } },
        });
      }
      if (requested.length === 0) {
        return json(res, 409, { error: { code: "NO_MODELS_AVAILABLE", message: "No trained models are available yet." } });
      }
      const bytes = Buffer.from(await file!.arrayBuffer());
      const ordered = MODEL_IDS.filter((id) => requested.includes(id));
      return json(res, 200, {
        mock: state.mock,
        image: {
          width: 640, height: 480, format: file!.type === "image/png" ? "PNG" : "JPEG", mime_type: file!.type,
          size_bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex"),
        },
        results: ordered.map((id) => predictResult(id, state, gradcam === "true")),
        skipped: MODEL_IDS.filter((id) => !state.available[id]).map((id) => ({
          id, display_name: displayName(id), reason: "not_trained",
        })),
      });
    }
    json(res, 404, { error: { code: "NOT_FOUND", message: "Not found" } });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    state,
    calls,
    close: () =>
      new Promise<void>((resolve) => {
        server.closeAllConnections();
        server.close(() => resolve());
      }),
  };
}

/** A URL where nothing is listening (connection refused). */
export async function closedPortUrl(): Promise<string> {
  const server = http.createServer();
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  await new Promise<void>((resolve) => server.close(() => resolve()));
  return `http://127.0.0.1:${port}`;
}
