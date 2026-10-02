import type { DatabaseSync } from "node:sqlite";
import { transaction } from "./db.js";

/** One row of `scans`, column names as in db/schema.sql. */
export interface ScanRow {
  id: string;
  created_at: string;
  mode: "all" | "selected";
  original_filename: string;
  mime_type: "image/jpeg" | "image/png";
  size_bytes: number;
  width: number;
  height: number;
  sha256: string;
  image_path: string;
  thumb_path: string | null;
  models_available: number;
  models_total: number;
  models_run: number;
  n_pneumonia: number;
  n_normal: number;
  n_failed: number;
  mock: 0 | 1;
}

/** One row of `scan_results`. */
export interface ResultRow {
  scan_id: string;
  model_id: string;
  position: number;
  display_name: string;
  family: string;
  kind: "single" | "ensemble";
  member_count: number;
  status: "ok" | "error";
  prob_pneumonia: number | null;
  threshold: number | null;
  threshold_source: "comparison_csv" | "default" | null;
  label: "NORMAL" | "PNEUMONIA" | null;
  inference_ms: number | null;
  cold_start: 0 | 1 | null;
  heatmap_path: string | null;
  error: string | null;
}

export interface Cursor {
  createdAt: string;
  id: string;
}

const SCAN_COLUMNS = [
  "id", "created_at", "mode", "original_filename", "mime_type", "size_bytes", "width", "height", "sha256",
  "image_path", "thumb_path", "models_available", "models_total", "models_run", "n_pneumonia", "n_normal",
  "n_failed", "mock",
] as const satisfies readonly (keyof ScanRow)[];

const RESULT_COLUMNS = [
  "scan_id", "model_id", "position", "display_name", "family", "kind", "member_count", "status",
  "prob_pneumonia", "threshold", "threshold_source", "label", "inference_ms", "cold_start", "heatmap_path", "error",
] as const satisfies readonly (keyof ResultRow)[];

function insertSql(table: string, columns: readonly string[]): string {
  return `INSERT INTO ${table} (${columns.join(", ")}) VALUES (${columns.map((c) => `@${c}`).join(", ")})`;
}

/** All SQL lives here; every value is a bound parameter. */
export class ScanRepository {
  constructor(private readonly db: DatabaseSync) {}

  insert(scan: ScanRow, results: ResultRow[]): void {
    const insertScan = this.db.prepare(insertSql("scans", SCAN_COLUMNS));
    const insertResult = this.db.prepare(insertSql("scan_results", RESULT_COLUMNS));
    transaction(this.db, () => {
      insertScan.run({ ...scan });
      for (const result of results) insertResult.run({ ...result });
    });
  }

  get(id: string): { scan: ScanRow; results: ResultRow[] } | null {
    const scan = this.db.prepare("SELECT * FROM scans WHERE id = ?").get(id) as ScanRow | undefined;
    if (!scan) return null;
    const results = this.db
      .prepare("SELECT * FROM scan_results WHERE scan_id = ? ORDER BY position")
      .all(id) as unknown as ResultRow[];
    return { scan, results };
  }

  getResult(scanId: string, modelId: string): ResultRow | null {
    const row = this.db
      .prepare("SELECT * FROM scan_results WHERE scan_id = ? AND model_id = ?")
      .get(scanId, modelId) as ResultRow | undefined;
    return row ?? null;
  }

  /** Newest first; keyset pagination on (created_at, id) so equal timestamps never repeat or skip rows. */
  list(limit: number, before?: Cursor): ScanRow[] {
    if (before) {
      return this.db
        .prepare(
          "SELECT * FROM scans WHERE (created_at, id) < (?, ?) ORDER BY created_at DESC, id DESC LIMIT ?",
        )
        .all(before.createdAt, before.id, limit) as unknown as ScanRow[];
    }
    return this.db
      .prepare("SELECT * FROM scans ORDER BY created_at DESC, id DESC LIMIT ?")
      .all(limit) as unknown as ScanRow[];
  }

  /** Deletes the scan (scan_results go by ON DELETE CASCADE). Returns whether a row existed. */
  delete(id: string): boolean {
    return Number(this.db.prepare("DELETE FROM scans WHERE id = ?").run(id).changes) > 0;
  }

  ping(): boolean {
    return (this.db.prepare("SELECT 1 AS ok").get() as { ok: number }).ok === 1;
  }
}
