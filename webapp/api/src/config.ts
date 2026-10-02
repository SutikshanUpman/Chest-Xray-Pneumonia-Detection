import path from "node:path";
import { fileURLToPath } from "node:url";

/** webapp/api */
export const API_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export interface Config {
  host: string;
  port: number;
  modelServiceUrl: string;
  modelServiceTimeoutMs: number;
  storageDir: string;
  dbPath: string;
  corsOrigin: string;
}

function intFromEnv(raw: string | undefined, fallback: number): number {
  if (raw === undefined || raw.trim() === "") return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) throw new Error(`Invalid integer setting: ${raw}`);
  return value;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): Config {
  const storageDir = path.resolve(API_ROOT, env.STORAGE_DIR || "../storage");
  return {
    host: env.HOST || "127.0.0.1",
    port: intFromEnv(env.PORT, 4000),
    modelServiceUrl: (env.MODEL_SERVICE_URL || "http://127.0.0.1:8000").replace(/\/+$/, ""),
    modelServiceTimeoutMs: intFromEnv(env.MODEL_SERVICE_TIMEOUT_MS, 120_000),
    storageDir,
    dbPath: env.DB_PATH ? path.resolve(API_ROOT, env.DB_PATH) : path.join(storageDir, "app.db"),
    corsOrigin: env.CORS_ORIGIN || "http://localhost:3000",
  };
}
