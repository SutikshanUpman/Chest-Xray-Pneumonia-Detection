import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { SCHEMA_VERSION, openDatabase } from "../src/db.js";
import { ScanRepository, type ResultRow, type ScanRow } from "../src/repo.js";

const ID = "9b2f1c1e-1a2b-4c3d-8e4f-0123456789ab";

function scan(overrides: Partial<ScanRow> = {}): ScanRow {
  return {
    id: ID, created_at: "2026-10-03T10:00:00.000Z", mode: "all", original_filename: "x.jpg", mime_type: "image/jpeg",
    size_bytes: 1000, width: 640, height: 480, sha256: "a".repeat(64), image_path: `scans/${ID}/original.jpg`,
    thumb_path: null, models_available: 2, models_total: 6, models_run: 2, n_pneumonia: 1, n_normal: 1, n_failed: 0,
    mock: 1, ...overrides,
  };
}

function result(overrides: Partial<ResultRow> = {}): ResultRow {
  return {
    scan_id: ID, model_id: "class_weights", position: 0, display_name: "Class Weights", family: "Phase 1",
    kind: "single", member_count: 1, status: "ok", prob_pneumonia: 0.7, threshold: 0.5, threshold_source: "default",
    label: "PNEUMONIA", inference_ms: 10, cold_start: 0, heatmap_path: null, error: null, ...overrides,
  };
}

let dir: string;
let db: DatabaseSync;
let repo: ScanRepository;

beforeEach(async () => {
  dir = await mkdtemp(path.join(tmpdir(), "pneumonia-db-"));
  db = openDatabase(path.join(dir, "app.db"));
  repo = new ScanRepository(db);
});

afterEach(async () => {
  db.close();
  await rm(dir, { recursive: true, force: true });
});

describe("openDatabase", () => {
  it("turns foreign keys on and sets the schema version", () => {
    expect(db.prepare("PRAGMA foreign_keys").get()).toEqual({ foreign_keys: 1 });
    expect(db.prepare("PRAGMA user_version").get()).toEqual({ user_version: SCHEMA_VERSION });
  });

  it("turns foreign keys on for every new connection and does not re-apply the schema", () => {
    repo.insert(scan(), [result(), result({ model_id: "oversampling", position: 1, label: "NORMAL", prob_pneumonia: 0.2 })]);
    const second = openDatabase(path.join(dir, "app.db"));
    try {
      expect(second.prepare("PRAGMA foreign_keys").get()).toEqual({ foreign_keys: 1 });
      expect(second.prepare("SELECT COUNT(*) AS n FROM scans").get()).toEqual({ n: 1 });
    } finally {
      second.close();
    }
  });

  it("refuses an unknown schema version", () => {
    const file = path.join(dir, "future.db");
    const raw = new DatabaseSync(file);
    raw.exec("PRAGMA user_version = 99");
    raw.close();
    expect(() => openDatabase(file)).toThrow(/Unsupported database schema version 99/);
  });
});

describe("foreign keys", () => {
  it("rejects a result for a scan that does not exist", () => {
    const insert = db.prepare(
      "INSERT INTO scan_results (scan_id, model_id, position, display_name, family, kind, member_count, status, prob_pneumonia, threshold, threshold_source, label) VALUES (?, 'class_weights', 0, 'x', 'x', 'single', 1, 'ok', 0.5, 0.5, 'default', 'NORMAL')",
    );
    expect(() => insert.run("00000000-0000-4000-8000-000000000000")).toThrow(/FOREIGN KEY constraint failed/);
  });

  it("cascades scan deletion to scan_results", () => {
    repo.insert(scan(), [result(), result({ model_id: "oversampling", position: 1, label: "NORMAL", prob_pneumonia: 0.2 })]);
    expect(db.prepare("SELECT COUNT(*) AS n FROM scan_results").get()).toEqual({ n: 2 });
    expect(repo.delete(ID)).toBe(true);
    expect(db.prepare("SELECT COUNT(*) AS n FROM scan_results").get()).toEqual({ n: 0 });
    expect(repo.delete(ID)).toBe(false);
  });
});

describe("CHECK constraints", () => {
  const twoResults = [result(), result({ model_id: "oversampling", position: 1, label: "NORMAL", prob_pneumonia: 0.2 })];

  it.each<[string, Partial<ScanRow>]>([
    ["consensus does not add up to models_run", { n_pneumonia: 2 }],
    ["models_run disagrees with the counts", { models_run: 3 }],
    ["unknown mode", { mode: "bogus" as ScanRow["mode"] }],
    ["unsupported mime type", { mime_type: "image/gif" as ScanRow["mime_type"] }],
    ["empty file", { size_bytes: 0 }],
    ["file over 10 MB", { size_bytes: 10 * 1024 * 1024 + 1 }],
    ["mock not 0/1", { mock: 2 as ScanRow["mock"] }],
  ])("scans: %s", (_name, overrides) => {
    expect(() => repo.insert(scan(overrides), twoResults)).toThrow(/CHECK constraint failed/);
    expect(db.prepare("SELECT COUNT(*) AS n FROM scans").get()).toEqual({ n: 0 }); // transaction rolled back
  });

  it.each<[string, Partial<ResultRow>]>([
    ["probability above 1", { prob_pneumonia: 1.5 }],
    ["probability below 0", { prob_pneumonia: -0.1 }],
    ["ok without probability", { prob_pneumonia: null }],
    ["ok without label", { label: null }],
    ["ok without threshold", { threshold: null }],
    ["error without message", { status: "error", prob_pneumonia: null, label: null, error: null }],
    ["unknown status", { status: "maybe" as ResultRow["status"] }],
    ["unknown label", { label: "MAYBE" as ResultRow["label"] }],
    ["unknown threshold source", { threshold_source: "test_set" as ResultRow["threshold_source"] }],
    ["unknown kind", { kind: "triple" as ResultRow["kind"] }],
    ["cold_start not 0/1", { cold_start: 5 as ResultRow["cold_start"] }],
  ])("scan_results: %s", (_name, overrides) => {
    const results = [result(overrides), twoResults[1]!];
    expect(() => repo.insert(scan(), results)).toThrow(/CHECK constraint failed/);
    expect(db.prepare("SELECT COUNT(*) AS n FROM scans").get()).toEqual({ n: 0 });
  });

  it("accepts a valid error row", () => {
    repo.insert(scan({ n_pneumonia: 0, n_failed: 1 }), [
      result({ status: "error", prob_pneumonia: null, label: null, inference_ms: null, error: "boom" }),
      twoResults[1]!,
    ]);
    expect(repo.get(ID)!.results.map((r) => r.status)).toEqual(["error", "ok"]);
  });
});
