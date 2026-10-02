-- Applied at startup; schema version tracked with PRAGMA user_version (no migrations table for a 2-table DB).
-- PRAGMA foreign_keys = ON is set on every connection by openDatabase() in src/db.ts (SQLite default is OFF,
-- and the pragma is a per-connection setting, so it cannot live in this file).
-- No `models` table: the registry and file availability are owned by the model service; a copy here would drift.
-- No `users` table: single-user local demo.

-- One row per uploaded image + run.
CREATE TABLE IF NOT EXISTS scans (
  id                TEXT PRIMARY KEY,      -- UUIDv4: URL /scan/[id] and storage folder name (not guessable, safe as a path)
  created_at        TEXT NOT NULL,         -- ISO-8601 UTC: history sort order + display
  mode              TEXT NOT NULL CHECK (mode IN ('all','selected')),  -- decides comparison layout + benchmark table
  original_filename TEXT NOT NULL,         -- display only; never used to build a path
  mime_type         TEXT NOT NULL CHECK (mime_type IN ('image/jpeg','image/png')), -- from magic bytes; Content-Type when serving
  size_bytes        INTEGER NOT NULL CHECK (size_bytes BETWEEN 1 AND 10485760),     -- enforces the 10 MB rule at rest too
  width             INTEGER NOT NULL,      -- original pixel size: UI reserves aspect ratio, heatmap aligns, no layout shift
  height            INTEGER NOT NULL,
  sha256            TEXT NOT NULL,         -- integrity + spotting re-scans of the same image (deliberately not UNIQUE)
  image_path        TEXT NOT NULL,         -- relative to STORAGE_DIR, so the folder can move
  thumb_path        TEXT,                  -- nullable: thumbnail is optional (history falls back to the original)
  models_available  INTEGER NOT NULL,      -- "N of 6 models available" as it was at scan time
  models_total      INTEGER NOT NULL,
  models_run        INTEGER NOT NULL,      -- models actually run in this scan (in "selected" mode not the same as models_available)
  n_pneumonia       INTEGER NOT NULL,      -- consensus kept on the row so the history list needs no aggregation;
  n_normal          INTEGER NOT NULL,      --   safe because results are immutable once written
  n_failed          INTEGER NOT NULL DEFAULT 0,
  mock              INTEGER NOT NULL CHECK (mock IN (0,1)),  -- mock output must never be mistaken for a real model output
  CHECK (n_pneumonia + n_normal + n_failed = models_run)    -- every model run is counted exactly once in the consensus
);
CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at DESC);  -- history pagination

-- One row per model run within a scan. Snapshot values: thresholds/CSV can change after re-evaluation,
-- and history must show what actually produced each label.
CREATE TABLE IF NOT EXISTS scan_results (
  scan_id          TEXT NOT NULL REFERENCES scans(id) ON DELETE CASCADE, -- delete a scan = delete its results
  model_id         TEXT NOT NULL,           -- registry id, e.g. 'kfoldc_ensemble'
  position         INTEGER NOT NULL,        -- stable registry display order
  display_name     TEXT NOT NULL,           -- snapshot (registry text may change)
  family           TEXT NOT NULL,           -- snapshot, shown on card
  kind             TEXT NOT NULL CHECK (kind IN ('single','ensemble')),
  member_count     INTEGER NOT NULL,        -- "mean of 3 folds" on ensemble cards
  status           TEXT NOT NULL CHECK (status IN ('ok','error')),  -- one model failing must not lose the others
  prob_pneumonia   REAL CHECK (prob_pneumonia BETWEEN 0 AND 1),     -- the gauge value
  threshold        REAL,                    -- the threshold actually applied
  threshold_source TEXT CHECK (threshold_source IN ('comparison_csv','default')), -- 'default' => "uncalibrated" badge
  label            TEXT CHECK (label IN ('NORMAL','PNEUMONIA')),
  inference_ms     REAL,                    -- excludes model loading
  cold_start       INTEGER CHECK (cold_start IN (0,1)),  -- this call also loaded the model: explains slow first runs
  heatmap_path     TEXT,                    -- only phase2_stage2; NULL => "heatmap not available"
  error            TEXT,
  PRIMARY KEY (scan_id, model_id),
  CHECK ((status='ok' AND prob_pneumonia IS NOT NULL AND label IS NOT NULL AND threshold IS NOT NULL)
      OR (status='error' AND error IS NOT NULL))
);

PRAGMA user_version = 1;
