import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import request from "supertest";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ScanDetailDto, ScanSummaryDto } from "../src/dto.js";
import { ScanRepository, type ScanRow } from "../src/repo.js";
import { FAKE_HEATMAP_PNG, closedPortUrl } from "./fakeModelService.js";
import { GIF, JPEG, PNG, WEBP_THUMB, countRows, postScan, scanFolders, setup, type TestContext } from "./helpers.js";

let ctx: TestContext;

afterEach(async () => {
  await ctx?.close();
});

function expectError(res: request.Response, status: number, code: string) {
  expect(res.status, JSON.stringify(res.body)).toBe(status);
  expect(res.body).toMatchObject({ error: { code, message: expect.any(String) } });
}

function expectNothingStored() {
  expect(countRows(ctx.db, "scans")).toBe(0);
  expect(countRows(ctx.db, "scan_results")).toBe(0);
  expect(scanFolders(ctx.storage)).toEqual([]);
}

describe("POST /api/scans", () => {
  beforeEach(async () => {
    ctx = await setup();
  });

  it("runs all available models and persists the scan", async () => {
    ctx.fake.state.failModel = "kfold_ensemble";
    const res = await postScan(ctx.app, { filename: "chest x-ray.jpg" });
    expect(res.status, JSON.stringify(res.body)).toBe(201);
    const scan = res.body as ScanDetailDto;

    expect(scan.id).toMatch(/^[0-9a-f-]{36}$/);
    expect(scan.mode).toBe("all");
    expect(scan.mock).toBe(true);
    expect(scan.modelsAvailable).toBe(6);
    expect(scan.modelsTotal).toBe(6);
    expect(scan.results.map((r) => r.modelId)).toEqual([
      "class_weights", "oversampling", "kfold_ensemble", "kfoldc_ensemble", "phase2_stage1", "phase2_stage2",
    ]);
    // synthetic labels P, N, (failed), P, N, P
    expect(scan.consensus).toEqual({ pneumonia: 3, normal: 2, failed: 1, total: 6 });
    expect(scan.benchmark).toBeNull();
    expect(scan.image).toMatchObject({
      url: `/api/scans/${scan.id}/image`, thumbUrl: `/api/scans/${scan.id}/image`, width: 640, height: 480,
      filename: "chest x-ray.jpg", mimeType: "image/jpeg", sizeBytes: JPEG.length,
    });

    const failed = scan.results.find((r) => r.modelId === "kfold_ensemble")!;
    expect(failed).toMatchObject({ status: "error", probPneumonia: null, label: null, heatmapNote: null });
    expect(failed.error).toContain("synthetic failure");

    const cw = scan.results.find((r) => r.modelId === "class_weights")!;
    expect(cw).toMatchObject({
      status: "ok", probPneumonia: 0.91, label: "PNEUMONIA", threshold: 0.5, thresholdSource: "default",
      uncalibrated: true, heatmapUrl: null, heatmapNote: "not_supported", coldStart: false, inferenceMs: 12.5,
    });
    const stage2 = scan.results.find((r) => r.modelId === "phase2_stage2")!;
    expect(stage2.heatmapUrl).toBe(`/api/scans/${scan.id}/heatmaps/phase2_stage2`);
    expect(stage2.heatmapNote).toBeNull();

    // what was forwarded to the model service
    const predictCall = ctx.fake.calls.find((c) => c.path === "/predict")!;
    expect(predictCall).toMatchObject({ modelIds: "all", gradcam: "true", fileSize: JPEG.length, fileType: "image/jpeg" });

    // persisted: rows, files, relative paths only, original filename never in a path
    const row = ctx.db.prepare("SELECT * FROM scans WHERE id = ?").get(scan.id) as unknown as ScanRow;
    expect(row).toMatchObject({ mock: 1, models_run: 6, n_pneumonia: 3, n_normal: 2, n_failed: 1 });
    expect(row.image_path).toBe(`scans/${scan.id}/original.jpg`);
    expect(countRows(ctx.db, "scan_results")).toBe(6);
    const heatmapPath = (
      ctx.db.prepare("SELECT heatmap_path FROM scan_results WHERE model_id = 'phase2_stage2'").get() as {
        heatmap_path: string;
      }
    ).heatmap_path;
    expect(heatmapPath).toBe(`scans/${scan.id}/heatmap_phase2_stage2.png`);
    expect(readFileSync(path.join(ctx.storage.root, heatmapPath))).toEqual(FAKE_HEATMAP_PNG);
    expect(scanFolders(ctx.storage)).toEqual([scan.id]);

    // GET returns the same detail
    const detail = await request(ctx.app).get(`/api/scans/${scan.id}`);
    expect(detail.status).toBe(200);
    expect(detail.body).toEqual(scan);

    // stored files are served back
    const image = await request(ctx.app).get(scan.image.url).buffer(true);
    expect(image.status).toBe(200);
    expect(image.headers["content-type"]).toBe("image/jpeg");
    expect(image.headers["x-content-type-options"]).toBe("nosniff");
    expect(Buffer.from(image.body as Buffer)).toEqual(JPEG);
    const heatmap = await request(ctx.app).get(stage2.heatmapUrl!).buffer(true);
    expect(heatmap.status).toBe(200);
    expect(heatmap.headers["content-type"]).toBe("image/png");
    expect(Buffer.from(heatmap.body as Buffer)).toEqual(FAKE_HEATMAP_PNG);
  });

  it("persists mock=false when the model service says so", async () => {
    ctx.fake.state.mock = false;
    const res = await postScan(ctx.app);
    expect(res.status).toBe(201);
    expect(res.body.mock).toBe(false);
  });

  it("runs only the selected models and adds the benchmark block", async () => {
    ctx.fake.state.comparisonFound = true;
    ctx.fake.state.ciFound = true;
    ctx.fake.state.available.phase2_stage1 = false;
    const res = await postScan(ctx.app, { mode: "selected", modelIds: "phase2_stage2, class_weights,class_weights" });
    expect(res.status, JSON.stringify(res.body)).toBe(201);
    const scan = res.body as ScanDetailDto;

    expect(ctx.fake.calls.find((c) => c.path === "/predict")!.modelIds).toBe("phase2_stage2,class_weights");
    expect(scan.results.map((r) => r.modelId)).toEqual(["class_weights", "phase2_stage2"]);
    expect(scan.modelsAvailable).toBe(5); // not the number of models run
    expect(scan.consensus).toEqual({ pneumonia: 2, normal: 0, failed: 0, total: 2 });
    expect(scan.results[0]).toMatchObject({ threshold: 0.6, thresholdSource: "comparison_csv", uncalibrated: false });
    expect(scan.benchmark).toEqual({
      ciAvailable: true,
      rows: ["class_weights", "phase2_stage2"].map((modelId) => ({
        modelId,
        displayName: `Fake ${modelId}`,
        macroF1: 0.8,
        recallNormal: { value: 0.7, ci: { low: expect.closeTo(0.65), high: expect.closeTo(0.75) } },
        recallPneumonia: { value: 0.9, ci: { low: expect.closeTo(0.85), high: expect.closeTo(0.95) } },
        rocAuc: { value: 0.92, ci: { low: expect.closeTo(0.87), high: expect.closeTo(0.97) } },
      })),
    });
    const row = ctx.db.prepare("SELECT models_run, models_available FROM scans").get();
    expect(row).toEqual({ models_run: 2, models_available: 5 });
  });

  it("has no benchmark block when comparison.csv is missing, and no CI when only the CI file is missing", async () => {
    const none = await postScan(ctx.app, { mode: "selected", modelIds: "class_weights" });
    expect(none.body.benchmark).toBeNull();

    ctx.fake.state.comparisonFound = true;
    const noCi = await postScan(ctx.app, { mode: "selected", modelIds: "class_weights" });
    expect(noCi.body.benchmark.ciAvailable).toBe(false);
    expect(noCi.body.benchmark.rows[0].recallNormal).toEqual({ value: 0.7, ci: null });
  });

  it("stores a client-made thumbnail and uses it in thumbUrl", async () => {
    const res = await postScan(ctx.app, { thumbnail: WEBP_THUMB });
    expect(res.status).toBe(201);
    expect(res.body.image.thumbUrl).toBe(`/api/scans/${res.body.id}/thumbnail`);
    const thumb = await request(ctx.app).get(res.body.image.thumbUrl).buffer(true);
    expect(thumb.status).toBe(200);
    expect(thumb.headers["content-type"]).toBe("image/webp");
  });

  it("accepts PNG by magic bytes even with a .jpg name and JPEG content type", async () => {
    const res = await postScan(ctx.app, { image: PNG, filename: "x.jpg", contentType: "image/jpeg" });
    expect(res.status).toBe(201);
    expect(res.body.image.mimeType).toBe("image/png");
    expect(ctx.fake.calls.find((c) => c.path === "/predict")!.fileType).toBe("image/png");
  });

  describe("validation (nothing is stored)", () => {
    it.each([
      ["GIF", { image: GIF }, 415, "UNSUPPORTED_TYPE"],
      ["text pretending to be JPEG", { image: Buffer.from("hello"), contentType: "image/jpeg" }, 415, "UNSUPPORTED_TYPE"],
      ["empty file", { image: Buffer.alloc(0) }, 400, "EMPTY_FILE"],
      ["bad mode", { mode: "everything" }, 400, "VALIDATION"],
      ["selected without ids", { mode: "selected" }, 400, "VALIDATION"],
      ["selected with blank ids", { mode: "selected", modelIds: " , " }, 400, "VALIDATION"],
      ["unknown id", { mode: "selected", modelIds: "class_weights,kfold_fold_1" }, 400, "UNKNOWN_MODEL"],
      ["path-like id", { mode: "selected", modelIds: "../../etc" }, 400, "UNKNOWN_MODEL"],
      ["thumbnail too large", { thumbnail: Buffer.concat([WEBP_THUMB, Buffer.alloc(200 * 1024)]) }, 413, "THUMBNAIL_TOO_LARGE"],
      ["thumbnail wrong type", { thumbnail: GIF }, 415, "UNSUPPORTED_TYPE"],
    ])("%s", async (_name, opts, status, code) => {
      expectError(await postScan(ctx.app, opts), status, code);
      expect(ctx.fake.calls.filter((c) => c.path === "/predict")).toEqual([]);
      expectNothingStored();
    });

    it("rejects an image over 10 MB", async () => {
      const big = Buffer.concat([JPEG, Buffer.alloc(10 * 1024 * 1024)]);
      expectError(await postScan(ctx.app, { image: big }), 413, "FILE_TOO_LARGE");
      expectNothingStored();
    });

    it("rejects a request without an image", async () => {
      expectError(await request(ctx.app).post("/api/scans").field("mode", "all"), 400, "VALIDATION");
      expectNothingStored();
    });

    it("rejects an unexpected file field", async () => {
      const res = await postScan(ctx.app).attach("other", JPEG, "x.jpg");
      expectError(res, 400, "VALIDATION");
      expectNothingStored();
    });
  });

  describe("model service failures (nothing is stored)", () => {
    it("maps a selected-but-untrained model to 409", async () => {
      ctx.fake.state.available.phase2_stage2 = false;
      const res = await postScan(ctx.app, { mode: "selected", modelIds: "phase2_stage2" });
      expectError(res, 409, "MODELS_UNAVAILABLE");
      expect(res.body.error.details).toEqual({ unavailable: ["phase2_stage2"] });
      expectNothingStored();
    });

    it("maps 'all' with nothing trained to 409", async () => {
      for (const id of Object.keys(ctx.fake.state.available)) ctx.fake.state.available[id] = false;
      expectError(await postScan(ctx.app), 409, "MODELS_UNAVAILABLE");
      expectNothingStored();
    });

    it("passes through an image the service cannot decode (415)", async () => {
      ctx.fake.state.predict = "invalid415";
      expectError(await postScan(ctx.app), 415, "INVALID_IMAGE");
      expectNothingStored();
    });

    it.each([
      ["HTTP 500", "http500"],
      ["invalid JSON", "badjson"],
      ["unexpected shape", "badshape"],
    ] as const)("maps %s to 502", async (_name, behaviour) => {
      ctx.fake.state.predict = behaviour;
      expectError(await postScan(ctx.app), 502, "MODEL_SERVICE_ERROR");
      expectNothingStored();
    });
  });
});

describe("model service down or slow", () => {
  it("returns 503 when the service is not running", async () => {
    ctx = await setup({ modelServiceUrl: await closedPortUrl() });
    expectError(await postScan(ctx.app), 503, "MODEL_SERVICE_DOWN");
    expectNothingStored();
  });

  it("returns 504 when the service does not answer in time", async () => {
    ctx = await setup({ fake: { predict: "slow", slowMs: 1500 }, timeoutMs: 200 });
    expectError(await postScan(ctx.app), 504, "MODEL_SERVICE_TIMEOUT");
    expectNothingStored();
  });

  it("still serves a stored selected scan (without benchmark) when the service goes down", async () => {
    ctx = await setup({ fake: { comparisonFound: true } });
    const created = await postScan(ctx.app, { mode: "selected", modelIds: "class_weights" });
    expect(created.body.benchmark).not.toBeNull();
    await ctx.fake.close();
    const res = await request(ctx.app).get(`/api/scans/${created.body.id}`);
    expect(res.status).toBe(200);
    expect(res.body.benchmark).toBeNull();
    expect(res.body.results).toHaveLength(1);
  });
});

describe("history", () => {
  beforeEach(async () => {
    ctx = await setup();
  });

  it("paginates newest first with a cursor, without repeats", async () => {
    const ids: string[] = [];
    for (let i = 0; i < 5; i++) ids.push((await postScan(ctx.app, { mode: i % 2 ? "selected" : "all", modelIds: "oversampling" })).body.id);

    const seen: ScanSummaryDto[] = [];
    let cursor: string | null = null;
    let pages = 0;
    do {
      const res: request.Response = await request(ctx.app)
        .get("/api/scans")
        .query(cursor ? { limit: 2, before: cursor } : { limit: 2 });
      expect(res.status).toBe(200);
      seen.push(...res.body.items);
      cursor = res.body.nextCursor;
      pages++;
    } while (cursor);

    expect(pages).toBe(3);
    expect(seen.map((s) => s.id).sort()).toEqual([...ids].sort());
    const keys = seen.map((s) => `${s.createdAt}|${s.id}`);
    expect(keys).toEqual([...keys].sort().reverse());
    expect(seen.find((s) => s.mode === "selected")).toMatchObject({ modelCount: 1, mock: true });
    expect(seen.find((s) => s.mode === "all")).toMatchObject({
      modelCount: 6, consensus: { pneumonia: 4, normal: 2, failed: 0, total: 6 },
    });
  });

  it("breaks created_at ties by id", async () => {
    const repo = new ScanRepository(ctx.db);
    const ids = [
      "11111111-1111-4111-8111-111111111111",
      "22222222-2222-4222-8222-222222222222",
      "33333333-3333-4333-8333-333333333333",
    ];
    for (const id of ids) {
      repo.insert(
        {
          id, created_at: "2026-10-03T10:00:00.000Z", mode: "all", original_filename: "x.jpg", mime_type: "image/jpeg",
          size_bytes: 10, width: 1, height: 1, sha256: "0".repeat(64), image_path: `scans/${id}/original.jpg`,
          thumb_path: null, models_available: 0, models_total: 6, models_run: 0, n_pneumonia: 0, n_normal: 0,
          n_failed: 0, mock: 1,
        },
        [],
      );
    }
    const first = await request(ctx.app).get("/api/scans").query({ limit: 2 });
    const second = await request(ctx.app).get("/api/scans").query({ limit: 2, before: first.body.nextCursor });
    expect([...first.body.items, ...second.body.items].map((s: ScanSummaryDto) => s.id)).toEqual([...ids].reverse());
    expect(second.body.nextCursor).toBeNull();
  });

  it.each([
    [{ limit: 0 }, "VALIDATION"],
    [{ limit: 51 }, "VALIDATION"],
    [{ limit: "abc" }, "VALIDATION"],
    [{ before: "not-a-cursor" }, "INVALID_CURSOR"],
    [{ before: Buffer.from(JSON.stringify({ c: "2026", i: "../x" })).toString("base64url") }, "INVALID_CURSOR"],
  ])("rejects bad query %j", async (query, code) => {
    expectError(await request(ctx.app).get("/api/scans").query(query), 400, code);
  });

  it("returns an empty first page", async () => {
    const res = await request(ctx.app).get("/api/scans");
    expect(res.body).toEqual({ items: [], nextCursor: null });
  });
});

describe("GET and DELETE /api/scans/:id", () => {
  beforeEach(async () => {
    ctx = await setup();
  });

  it("returns 404 for an unknown id and 400 for a non-UUID", async () => {
    expectError(await request(ctx.app).get("/api/scans/9b2f1c1e-1a2b-4c3d-8e4f-0123456789ab"), 404, "NOT_FOUND");
    expectError(await request(ctx.app).get("/api/scans/123"), 400, "INVALID_ID");
  });

  it("deletes the scan, its results (cascade) and its storage folder", async () => {
    const keep = (await postScan(ctx.app)).body.id as string;
    const id = (await postScan(ctx.app, { thumbnail: WEBP_THUMB })).body.id as string;
    expect(countRows(ctx.db, "scan_results")).toBe(12);
    expect(existsSync(ctx.storage.scanDir(id))).toBe(true);

    const res = await request(ctx.app).delete(`/api/scans/${id}`);
    expect(res.status).toBe(204);

    expect(ctx.db.prepare("SELECT COUNT(*) AS n FROM scan_results WHERE scan_id = ?").get(id)).toEqual({ n: 0 });
    expect(countRows(ctx.db, "scan_results")).toBe(6);
    expect(existsSync(ctx.storage.scanDir(id))).toBe(false);
    expect(scanFolders(ctx.storage)).toEqual([keep]);

    expectError(await request(ctx.app).get(`/api/scans/${id}`), 404, "NOT_FOUND");
    expectError(await request(ctx.app).get(`/api/scans/${id}/image`), 404, "NOT_FOUND");
    expectError(await request(ctx.app).delete(`/api/scans/${id}`), 404, "NOT_FOUND");
    expectError(await request(ctx.app).delete("/api/scans/nope"), 400, "INVALID_ID");
  });
});

describe("file routes", () => {
  let id: string;
  beforeEach(async () => {
    ctx = await setup();
    id = (await postScan(ctx.app)).body.id;
  });

  it.each([
    "/api/scans/..%2F..%2Fpackage.json/image",
    "/api/scans/..%5C..%5Cpackage.json/image",
    "/api/scans/%2e%2e/thumbnail",
    "/api/scans/..%2Fapp.db/heatmaps/phase2_stage2",
  ])("rejects a traversal id: %s", async (url) => {
    const res = await request(ctx.app).get(url);
    expect([400, 404]).toContain(res.status);
    expect(res.headers["content-type"]).toMatch(/application\/json/);
  });

  it.each(["..%2F..%2Fapp.db", "..%2Foriginal.jpg", "phase2_stage2.png", "kfold_fold_1"])(
    "rejects a model id that is not in the registry: %s",
    async (modelId) => {
      expectError(await request(ctx.app).get(`/api/scans/${id}/heatmaps/${modelId}`), 400, "UNKNOWN_MODEL");
    },
  );

  it("never serves a file outside the scan folder, even if the DB path is tampered with", async () => {
    for (const bad of [`scans/${id}/../../app.db`, "../package.json", "app.db", path.resolve("package.json")]) {
      ctx.db.prepare("UPDATE scans SET image_path = ? WHERE id = ?").run(bad, id);
      const res = await request(ctx.app).get(`/api/scans/${id}/image`);
      expectError(res, 404, "NOT_FOUND");
    }
  });

  it("returns 404 for a missing thumbnail or heatmap", async () => {
    expectError(await request(ctx.app).get(`/api/scans/${id}/thumbnail`), 404, "NOT_FOUND");
    expectError(await request(ctx.app).get(`/api/scans/${id}/heatmaps/class_weights`), 404, "NOT_FOUND");
  });
});
