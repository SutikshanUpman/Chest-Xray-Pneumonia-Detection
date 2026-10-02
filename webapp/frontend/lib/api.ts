// Functions that call the web API. The browser only talks to same-origin /api (Next forwards it to Express).
import { ApiRequestError } from "./errors";
import type { ModelsDto, ScanDetailDto, ScanListDto, ScanMode } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store", ...init });
  } catch {
    throw new ApiRequestError(0, "NETWORK", "Could not reach the API");
  }
  if (res.status === 204) return undefined as T;

  // If the API is down, the Next proxy answers with a non-JSON error page
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const code = body?.error?.code ?? "NETWORK";
    throw new ApiRequestError(res.status, code, body?.error?.message ?? `HTTP ${res.status}`);
  }
  return body as T;
}

export function getModels() {
  return request<ModelsDto>("/api/models");
}

export function createScan(file: File, mode: ScanMode, modelIds: string[]) {
  const form = new FormData();
  form.append("image", file);
  form.append("mode", mode);
  if (mode === "selected") form.append("model_ids", modelIds.join(","));
  return request<ScanDetailDto>("/api/scans", { method: "POST", body: form });
}

export function getScan(id: string) {
  return request<ScanDetailDto>(`/api/scans/${id}`);
}

export function listScans(before: string | null) {
  const query = before ? `&before=${encodeURIComponent(before)}` : "";
  return request<ScanListDto>(`/api/scans?limit=12${query}`);
}

export function deleteScan(id: string) {
  return request<void>(`/api/scans/${id}`, { method: "DELETE" });
}
