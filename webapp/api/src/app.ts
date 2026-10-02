import cors from "cors";
import express, { type Express } from "express";
import type { DatabaseSync } from "node:sqlite";
import { modelsDto, type ModelsDto } from "./dto.js";
import { ApiError, errorBody, errorHandler } from "./errors.js";
import type { ModelServiceClient } from "./modelService.js";
import { ScanRepository } from "./repo.js";
import { scansRouter } from "./routes/scans.js";
import type { Storage } from "./storage.js";

export interface AppDeps {
  db: DatabaseSync;
  storage: Storage;
  modelService: ModelServiceClient;
  corsOrigin: string;
}

function asApiError(err: unknown): ApiError {
  return err instanceof ApiError ? err : new ApiError(502, "MODEL_SERVICE_ERROR", "Model service error.");
}

export function createApp({ db, storage, modelService, corsOrigin }: AppDeps): Express {
  const repo = new ScanRepository(db);
  const app = express();
  app.disable("x-powered-by");
  // An array makes `cors` echo the origin only when it matches; other origins get no CORS headers
  app.use(cors({ origin: [corsOrigin], methods: ["GET", "POST", "DELETE"] }));

  app.use("/api/scans", scansRouter({ repo, storage, modelService }));

  // Never an error status: the UI shows "model service offline" from serviceOnline=false
  app.get("/api/models", async (_req, res) => {
    let body: ModelsDto;
    try {
      body = modelsDto(await modelService.models());
    } catch (err) {
      const e = asApiError(err);
      body = { serviceOnline: false, error: { code: e.code, message: e.message } };
    }
    res.json(body);
  });

  app.get("/api/health", async (_req, res) => {
    let dbStatus: "ok" | "error" = "error";
    try {
      if (repo.ping()) dbStatus = "ok";
    } catch {
      // reported below
    }
    let modelServiceStatus;
    try {
      const h = await modelService.health();
      modelServiceStatus = {
        online: true,
        mock: h.mock,
        mockAllAvailable: h.mock_all_available,
        tensorflow: h.tensorflow,
        loadedModels: h.loaded_models,
      };
    } catch (err) {
      const e = asApiError(err);
      modelServiceStatus = { online: false, error: { code: e.code, message: e.message } };
    }
    res.json({ status: "ok", api: "ok", db: dbStatus, modelService: modelServiceStatus });
  });

  app.use((_req, res) => {
    res.status(404).json(errorBody("NOT_FOUND", "Route not found."));
  });
  app.use(errorHandler);
  return app;
}
