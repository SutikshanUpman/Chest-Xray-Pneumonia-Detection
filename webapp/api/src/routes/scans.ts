import { randomUUID } from "node:crypto";
import { Router, type NextFunction, type Response } from "express";
import multer from "multer";
import { scanDetailDto, scanSummaryDto } from "../dto.js";
import { ApiError } from "../errors.js";
import type { ModelServiceClient, MsModels } from "../modelService.js";
import { isModelId } from "../registry.js";
import type { Cursor, ResultRow, ScanRepository, ScanRow } from "../repo.js";
import type { Storage, StoredFile } from "../storage.js";
import {
  EXTENSION_BY_MIME,
  MAX_IMAGE_BYTES,
  MAX_THUMBNAIL_BYTES,
  PNG_MAGIC,
  isUuid,
  parseMode,
  parseModelIds,
  sniffImage,
  sniffThumbnail,
} from "../validation.js";

export interface ScanRouteDeps {
  repo: ScanRepository;
  storage: Storage;
  modelService: ModelServiceClient;
}

const DEFAULT_PAGE_SIZE = 20;
const MAX_PAGE_SIZE = 50;

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: MAX_IMAGE_BYTES, files: 2, fields: 4, fieldSize: 2048, parts: 6 },
}).fields([
  { name: "image", maxCount: 1 },
  { name: "thumbnail", maxCount: 1 },
]);

function requireScanId(raw: unknown): string {
  if (!isUuid(raw)) throw new ApiError(400, "INVALID_ID", "Invalid scan id.");
  return raw;
}

function notFound(): ApiError {
  return new ApiError(404, "NOT_FOUND", "Scan not found.");
}

function encodeCursor(scan: ScanRow): string {
  return Buffer.from(JSON.stringify({ c: scan.created_at, i: scan.id })).toString("base64url");
}

function decodeCursor(raw: unknown): Cursor | undefined {
  if (raw === undefined) return undefined;
  try {
    const parsed: unknown = JSON.parse(Buffer.from(String(raw), "base64url").toString("utf8"));
    const { c, i } = parsed as { c?: unknown; i?: unknown };
    if (typeof c === "string" && !Number.isNaN(Date.parse(c)) && isUuid(i)) return { createdAt: c, id: i };
  } catch {
    // fall through
  }
  throw new ApiError(400, "INVALID_CURSOR", "Invalid pagination cursor.");
}

function parseLimit(raw: unknown): number {
  if (raw === undefined) return DEFAULT_PAGE_SIZE;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1 || value > MAX_PAGE_SIZE) {
    throw new ApiError(400, "VALIDATION", `limit must be an integer from 1 to ${MAX_PAGE_SIZE}.`);
  }
  return value;
}

/** Display-only copy of the original filename (never used in a path). */
function cleanFilename(name: string | undefined): string {
  // multer/busboy decodes the header as latin1
  const decoded = Buffer.from(name ?? "", "latin1").toString("utf8");
  // eslint-disable-next-line no-control-regex
  const cleaned = decoded.replace(/[\u0000-\u001f\u007f]/g, "").trim().slice(0, 255);
  return cleaned || "upload";
}

export function scansRouter({ repo, storage, modelService }: ScanRouteDeps): Router {
  const router = Router();

  async function catalogOrNull(): Promise<MsModels | null> {
    try {
      return await modelService.models();
    } catch {
      return null; // benchmark block is optional
    }
  }

  router.post("/", upload, async (req, res) => {
    const files = req.files as Record<string, Express.Multer.File[]> | undefined;
    const image = files?.image?.[0];
    if (!image) throw new ApiError(400, "VALIDATION", "Missing image file (field \"image\").");
    const mode = parseMode(req.body?.mode);
    const selectedIds = mode === "selected" ? parseModelIds(req.body?.model_ids) : [];

    if (image.size === 0) throw new ApiError(400, "EMPTY_FILE", "The uploaded file is empty.");
    const mime = sniffImage(image.buffer);
    if (!mime) throw new ApiError(415, "UNSUPPORTED_TYPE", "Only JPEG and PNG images are accepted.");

    const thumbnail = files?.thumbnail?.[0];
    const thumbMime = thumbnail ? sniffThumbnail(thumbnail.buffer) : null;
    if (thumbnail) {
      if (thumbnail.size > MAX_THUMBNAIL_BYTES) {
        throw new ApiError(413, "THUMBNAIL_TOO_LARGE", "The thumbnail is larger than 200 KB.");
      }
      if (!thumbMime) throw new ApiError(415, "UNSUPPORTED_TYPE", "The thumbnail must be JPEG, PNG or WebP.");
    }

    const catalog = await modelService.models();
    const prediction = await modelService.predict({
      image: image.buffer,
      mime,
      modelIds: mode === "all" ? "all" : selectedIds,
      gradcam: true,
    });

    const id = randomUUID();
    const imageName = `original.${EXTENSION_BY_MIME[mime]}`;
    const stored: StoredFile[] = [{ name: imageName, data: image.buffer }];
    let thumbPath: string | null = null;
    if (thumbnail && thumbMime) {
      const name = `thumb.${EXTENSION_BY_MIME[thumbMime]}`;
      stored.push({ name, data: thumbnail.buffer });
      thumbPath = storage.relativePath(id, name);
    }

    const results: ResultRow[] = prediction.results.map((r, position) => {
      let heatmapPath: string | null = null;
      if (r.status === "ok" && r.heatmap_png_b64) {
        const png = Buffer.from(r.heatmap_png_b64, "base64");
        if (png.subarray(0, 8).equals(PNG_MAGIC)) {
          const name = `heatmap_${r.id}.png`; // r.id is a checked registry id
          stored.push({ name, data: png });
          heatmapPath = storage.relativePath(id, name);
        }
      }
      return {
        scan_id: id,
        model_id: r.id,
        position,
        display_name: r.display_name,
        family: r.family,
        kind: r.kind,
        member_count: r.member_count,
        status: r.status,
        prob_pneumonia: r.status === "ok" ? r.prob_pneumonia : null,
        threshold: r.threshold,
        threshold_source: r.threshold_source,
        label: r.status === "ok" ? r.label : null,
        inference_ms: r.inference_ms,
        cold_start: r.cold_start ? 1 : 0,
        heatmap_path: heatmapPath,
        error: r.status === "error" ? r.error : null,
      };
    });

    // Consensus uses each model's own label (its own val-tuned threshold); failures are counted apart
    const scan: ScanRow = {
      id,
      created_at: new Date().toISOString(),
      mode,
      original_filename: cleanFilename(image.originalname),
      mime_type: mime,
      size_bytes: image.size,
      width: prediction.image.width,
      height: prediction.image.height,
      sha256: prediction.image.sha256,
      image_path: storage.relativePath(id, imageName),
      thumb_path: thumbPath,
      models_available: catalog.available,
      models_total: catalog.total,
      models_run: results.length,
      n_pneumonia: results.filter((r) => r.status === "ok" && r.label === "PNEUMONIA").length,
      n_normal: results.filter((r) => r.status === "ok" && r.label === "NORMAL").length,
      n_failed: results.filter((r) => r.status === "error").length,
      mock: prediction.mock ? 1 : 0,
    };

    await storage.writeScan(id, stored);
    try {
      repo.insert(scan, results);
    } catch (err) {
      await storage.removeScan(id);
      throw err;
    }
    res.status(201).json(scanDetailDto(scan, results, catalog));
  });

  router.get("/", (req, res) => {
    const limit = parseLimit(req.query.limit);
    const before = decodeCursor(req.query.before);
    const rows = repo.list(limit + 1, before);
    const page = rows.slice(0, limit);
    const last = page.at(-1);
    res.json({
      items: page.map(scanSummaryDto),
      nextCursor: rows.length > limit && last ? encodeCursor(last) : null,
    });
  });

  router.get("/:id", async (req, res) => {
    const found = repo.get(requireScanId(req.params.id));
    if (!found) throw notFound();
    const catalog = found.scan.mode === "selected" ? await catalogOrNull() : null;
    res.json(scanDetailDto(found.scan, found.results, catalog));
  });

  router.delete("/:id", async (req, res) => {
    const id = requireScanId(req.params.id);
    if (!repo.delete(id)) throw notFound();
    await storage.removeScan(id);
    res.status(204).end();
  });

  function sendScanFile(res: Response, next: NextFunction, id: string, relative: string | null): void {
    const abs = relative ? storage.resolveScanFile(id, relative) : null;
    if (!abs) return next(new ApiError(404, "NOT_FOUND", "File not found."));
    res.sendFile(
      abs,
      {
        dotfiles: "deny",
        headers: { "Cache-Control": "private, max-age=31536000, immutable", "X-Content-Type-Options": "nosniff" },
      },
      (err) => {
        if (err && !res.headersSent) next(new ApiError(404, "NOT_FOUND", "File not found."));
      },
    );
  }

  router.get("/:id/image", (req, res, next) => {
    const id = requireScanId(req.params.id);
    const found = repo.get(id);
    if (!found) throw notFound();
    sendScanFile(res, next, id, found.scan.image_path);
  });

  router.get("/:id/thumbnail", (req, res, next) => {
    const id = requireScanId(req.params.id);
    const found = repo.get(id);
    if (!found) throw notFound();
    sendScanFile(res, next, id, found.scan.thumb_path);
  });

  router.get("/:id/heatmaps/:modelId", (req, res, next) => {
    const id = requireScanId(req.params.id);
    const { modelId } = req.params;
    if (!isModelId(modelId)) throw new ApiError(400, "UNKNOWN_MODEL", "Unknown model id.");
    if (!repo.get(id)) throw notFound();
    sendScanFile(res, next, id, repo.getResult(id, modelId)?.heatmap_path ?? null);
  });

  return router;
}
