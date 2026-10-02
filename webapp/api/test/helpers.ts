import { existsSync, readdirSync } from "node:fs";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import type { DatabaseSync } from "node:sqlite";
import type { Express } from "express";
import request from "supertest";
import { createApp } from "../src/app.js";
import { openDatabase } from "../src/db.js";
import { ModelServiceClient } from "../src/modelService.js";
import { Storage } from "../src/storage.js";
import { PNG_MAGIC } from "../src/validation.js";
import { startFakeModelService, type FakeModelService, type FakeState } from "./fakeModelService.js";

export const CORS_ORIGIN = "http://localhost:3000";

// Only magic bytes are checked by the API; the fake model service does not decode images
export const JPEG = Buffer.concat([Buffer.from([0xff, 0xd8, 0xff, 0xe0]), Buffer.alloc(256, 7)]);
export const PNG = Buffer.concat([PNG_MAGIC, Buffer.alloc(256, 3)]);
export const GIF = Buffer.concat([Buffer.from("GIF89a"), Buffer.alloc(64, 1)]);
export const WEBP_THUMB = Buffer.concat([Buffer.from("RIFF"), Buffer.alloc(4), Buffer.from("WEBP"), Buffer.alloc(64, 2)]);

export interface TestContext {
  app: Express;
  db: DatabaseSync;
  storage: Storage;
  fake: FakeModelService;
  close(): Promise<void>;
}

export async function setup(
  opts: { fake?: Partial<FakeState>; modelServiceUrl?: string; timeoutMs?: number } = {},
): Promise<TestContext> {
  const fake = await startFakeModelService(opts.fake);
  const dir = await mkdtemp(path.join(tmpdir(), "pneumonia-api-"));
  const storage = new Storage(path.join(dir, "storage"));
  const db = openDatabase(path.join(dir, "storage", "app.db"));
  const app = createApp({
    db,
    storage,
    modelService: new ModelServiceClient(opts.modelServiceUrl ?? fake.url, opts.timeoutMs ?? 5000),
    corsOrigin: CORS_ORIGIN,
  });
  return {
    app,
    db,
    storage,
    fake,
    async close() {
      db.close();
      await fake.close();
      await rm(dir, { recursive: true, force: true });
    },
  };
}

export function postScan(
  app: Express,
  opts: {
    image?: Buffer;
    filename?: string;
    contentType?: string;
    mode?: string;
    modelIds?: string;
    thumbnail?: Buffer;
  } = {},
) {
  const req = request(app)
    .post("/api/scans")
    .attach("image", opts.image ?? JPEG, { filename: opts.filename ?? "chest.jpg", contentType: opts.contentType ?? "image/jpeg" })
    .field("mode", opts.mode ?? "all");
  if (opts.modelIds !== undefined) req.field("model_ids", opts.modelIds);
  if (opts.thumbnail) req.attach("thumbnail", opts.thumbnail, { filename: "thumb.webp", contentType: "image/webp" });
  return req;
}

export function countRows(db: DatabaseSync, table: "scans" | "scan_results"): number {
  return (db.prepare(`SELECT COUNT(*) AS n FROM ${table}`).get() as { n: number }).n;
}

export function scanFolders(storage: Storage): string[] {
  const dir = path.join(storage.root, "scans");
  return existsSync(dir) ? readdirSync(dir) : [];
}
