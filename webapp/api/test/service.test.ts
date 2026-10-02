import { readFileSync } from "node:fs";
import path from "node:path";
import request from "supertest";
import { afterEach, describe, expect, it } from "vitest";
import { API_ROOT } from "../src/config.js";
import { GRADCAM_MODEL_IDS, MODEL_IDS } from "../src/registry.js";
import { closedPortUrl } from "./fakeModelService.js";
import { CORS_ORIGIN, setup, type TestContext } from "./helpers.js";

let ctx: TestContext;

afterEach(async () => {
  await ctx?.close();
});

function allKeys(value: unknown, keys: string[] = []): string[] {
  if (Array.isArray(value)) value.forEach((v) => allKeys(v, keys));
  else if (value && typeof value === "object") {
    for (const [k, v] of Object.entries(value)) {
      keys.push(k);
      allKeys(v, keys);
    }
  }
  return keys;
}

describe("registry ids", () => {
  it("match webapp/model-service/app/registry.py", () => {
    const source = readFileSync(path.join(API_ROOT, "../model-service/app/registry.py"), "utf8");
    const specs = source.split("ModelSpec(").slice(1); // one block per registry entry
    const ids = specs.map((block) => /\bid="([a-z0-9_]+)"/.exec(block)?.[1]);
    expect(ids).toEqual([...MODEL_IDS]);
    const gradcam = specs.filter((block) => block.includes("gradcam=True")).map((b) => /\bid="([a-z0-9_]+)"/.exec(b)?.[1]);
    expect(gradcam).toEqual([...GRADCAM_MODEL_IDS]);
  });
});

describe("GET /api/models", () => {
  it("returns the model-service catalog in camelCase", async () => {
    ctx = await setup({ fake: { comparisonFound: true, ciFound: true } });
    const res = await request(ctx.app).get("/api/models");
    expect(res.status).toBe(200);
    expect(res.body).toMatchObject({
      serviceOnline: true, mock: true, total: 6, available: 6, imgSize: [224, 224],
      comparisonCsv: { found: true }, comparisonCiCsv: { found: true },
    });
    expect(res.body.models[5]).toMatchObject({
      id: "phase2_stage2", displayName: "Fake phase2_stage2", gradcamSupported: true, colorMode: "rgb",
      interpolation: "bilinear", thresholdSource: "comparison_csv", uncalibrated: false, thresholdStale: false,
      benchmark: { macroF1: 0.8, valMacroF1: 0.82, ci: { rocAuc: { low: expect.any(Number), high: expect.any(Number) } } },
    });
    expect(allKeys(res.body).filter((k) => k.includes("_"))).toEqual([]);
  });

  it("reports serviceOnline=false (HTTP 200) when the service is down", async () => {
    ctx = await setup({ modelServiceUrl: await closedPortUrl() });
    const res = await request(ctx.app).get("/api/models");
    expect(res.status).toBe(200);
    expect(res.body).toEqual({
      serviceOnline: false,
      error: { code: "MODEL_SERVICE_DOWN", message: expect.any(String) },
    });
  });
});

describe("GET /api/health", () => {
  it("reports api, db and model service", async () => {
    ctx = await setup();
    const res = await request(ctx.app).get("/api/health");
    expect(res.body).toEqual({
      status: "ok", api: "ok", db: "ok",
      modelService: { online: true, mock: true, mockAllAvailable: true, tensorflow: null, loadedModels: [] },
    });
  });

  it("stays up when the model service is down", async () => {
    ctx = await setup({ modelServiceUrl: await closedPortUrl() });
    const res = await request(ctx.app).get("/api/health");
    expect(res.status).toBe(200);
    expect(res.body.modelService).toMatchObject({ online: false, error: { code: "MODEL_SERVICE_DOWN" } });
  });
});

describe("CORS and unknown routes", () => {
  it("allows only http://localhost:3000", async () => {
    ctx = await setup();
    const ok = await request(ctx.app).get("/api/scans").set("Origin", CORS_ORIGIN);
    expect(ok.headers["access-control-allow-origin"]).toBe(CORS_ORIGIN);

    const evil = await request(ctx.app).get("/api/scans").set("Origin", "http://evil.example");
    expect(evil.headers["access-control-allow-origin"]).toBeUndefined();

    const preflight = await request(ctx.app)
      .options("/api/scans")
      .set("Origin", "http://localhost:3001")
      .set("Access-Control-Request-Method", "DELETE");
    expect(preflight.headers["access-control-allow-origin"]).toBeUndefined();
  });

  it("answers unknown routes with the error shape", async () => {
    ctx = await setup();
    const res = await request(ctx.app).get("/api/nope");
    expect(res.status).toBe(404);
    expect(res.body).toEqual({ error: { code: "NOT_FOUND", message: "Route not found." } });
    expect(res.headers["x-powered-by"]).toBeUndefined();
  });
});
