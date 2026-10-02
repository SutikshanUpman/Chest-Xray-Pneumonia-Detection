import { existsSync } from "node:fs";
import path from "node:path";
import { createApp } from "./app.js";
import { API_ROOT, loadConfig } from "./config.js";
import { openDatabase } from "./db.js";
import { ModelServiceClient } from "./modelService.js";
import { Storage } from "./storage.js";

const envFile = path.join(API_ROOT, ".env");
if (existsSync(envFile)) process.loadEnvFile(envFile);

const config = loadConfig();
const db = openDatabase(config.dbPath);
const app = createApp({
  db,
  storage: new Storage(config.storageDir),
  modelService: new ModelServiceClient(config.modelServiceUrl, config.modelServiceTimeoutMs),
  corsOrigin: config.corsOrigin,
});

const server = app.listen(config.port, config.host, () => {
  console.log(`API listening on http://${config.host}:${config.port} (model service ${config.modelServiceUrl})`);
});

function shutdown(): void {
  server.close(() => {
    db.close();
    process.exit(0);
  });
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
