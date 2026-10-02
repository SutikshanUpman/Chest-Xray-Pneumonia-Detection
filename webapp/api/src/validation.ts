import { ApiError } from "./errors.js";
import { isModelId } from "./registry.js";

export const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
export const MAX_THUMBNAIL_BYTES = 200 * 1024;

export type ImageMime = "image/jpeg" | "image/png";
export type ThumbnailMime = ImageMime | "image/webp";

export const EXTENSION_BY_MIME: Record<ThumbnailMime, string> = {
  "image/jpeg": "jpg",
  "image/png": "png",
  "image/webp": "webp",
};

const JPEG_MAGIC = Buffer.from([0xff, 0xd8, 0xff]);
export const PNG_MAGIC = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
// crypto.randomUUID() output: lowercase v4
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

export function isUuid(value: unknown): value is string {
  return typeof value === "string" && UUID_RE.test(value);
}

/** Magic bytes only; filename and Content-Type are never trusted. */
export function sniffImage(buf: Buffer): ImageMime | null {
  if (buf.subarray(0, 3).equals(JPEG_MAGIC)) return "image/jpeg";
  if (buf.subarray(0, 8).equals(PNG_MAGIC)) return "image/png";
  return null;
}

export function sniffThumbnail(buf: Buffer): ThumbnailMime | null {
  const image = sniffImage(buf);
  if (image) return image;
  if (buf.subarray(0, 4).toString("latin1") === "RIFF" && buf.subarray(8, 12).toString("latin1") === "WEBP") {
    return "image/webp";
  }
  return null;
}

export type ScanMode = "all" | "selected";

export function parseMode(raw: unknown): ScanMode {
  if (raw === "all" || raw === "selected") return raw;
  throw new ApiError(400, "VALIDATION", 'mode must be "all" or "selected".');
}

/** model_ids: comma-separated registry ids; duplicates removed, order kept. */
export function parseModelIds(raw: unknown): string[] {
  const text = Array.isArray(raw) ? raw.join(",") : typeof raw === "string" ? raw : "";
  const ids = [...new Set(text.split(",").map((s) => s.trim()).filter(Boolean))];
  if (ids.length === 0) {
    throw new ApiError(400, "VALIDATION", 'Select at least one model (model_ids) in "selected" mode.');
  }
  const unknown = ids.filter((id) => !isModelId(id));
  if (unknown.length > 0) throw new ApiError(400, "UNKNOWN_MODEL", "Unknown model id.", { unknown });
  return ids;
}
