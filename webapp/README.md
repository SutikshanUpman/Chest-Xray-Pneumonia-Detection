# Chest X-ray Pneumonia Demo (web app)

A small local web app on top of the models trained in `notebooks/`. You upload one chest X-ray,
the app runs it through the trained models, and it shows what each model predicts:
the probability of pneumonia, the label (Normal or Pneumonia), the threshold used, and for the
Phase 2 fine-tuned model a Grad-CAM heatmap of the regions that drove the prediction.
Past scans are saved and listed on a History page.

> **Educational demo, not a medical device. Not for diagnosis.**

## The three parts

| Folder | What it is | Port |
|---|---|---|
| `frontend/` | The website (Next.js + React + Tailwind) | 3000 |
| `api/` | The web API (Express + TypeScript). Saves scans in SQLite and files in `storage/` | 4000 |
| `model-service/` | The model server (FastAPI, Python). Loads the `.h5` models and makes predictions | 8000 |

All three listen on `127.0.0.1` only, so nothing is reachable from other computers.

### How a request flows

```
browser ──> frontend (:3000) ──/api──> API (:4000) ──> model service (:8000)
```

- The browser only ever calls `/api/...` on the frontend. Next.js forwards those calls to the API
  (see `frontend/next.config.ts`). The browser never talks to the model service directly.
- The API checks the upload (JPEG/PNG by its first bytes, max 10 MB), sends it to the model service,
  then stores the image, heatmap and results so the History page can show them later.
- The model service preprocesses the image exactly like `notebooks/03_evaluation.ipynb` and runs the models.

### Folder map

```
webapp/
├─ frontend/          Next.js website
│  ├─ app/            the pages: / (upload), /scan/[id] (results), /history
│  ├─ components/     the small UI pieces (upload box, model card, heatmap viewer, ...)
│  └─ lib/            API calls, types, error messages, number formatting
├─ api/               Express API
│  ├─ db/schema.sql   the two SQLite tables (scans, scan_results), with a comment on every column
│  ├─ src/            routes, database, storage, and the client that calls the model service
│  └─ test/           API tests (they use a fake model service)
├─ model-service/     FastAPI model server
│  ├─ app/            registry of the 6 models, preprocessing, inference, Grad-CAM, mock mode
│  └─ tests/          pytest tests (mock mode, no TensorFlow needed)
└─ storage/           created at runtime: app.db + uploaded images and heatmaps (gitignored)
```

## Mock mode vs real mode

**Mock mode** (`MOCK_MODELS=1`) is for working on the website without trained models.
The model service returns made-up but repeatable probabilities (the same image always gets the same
numbers) and a synthetic heatmap. It never imports TensorFlow, so it runs in the small `ai-web` env.
The website shows a yellow **MOCK DATA** banner on every page where mock data appears, and every scan
made in mock mode is saved with a mock flag.

`MOCK_ALL_AVAILABLE=1` (mock mode only) pretends all 6 models are trained, so every button can be tried.

**Real mode** is simply the model service started *without* those variables, in the `ai-cpu` env
(which has TensorFlow). It loads each `.h5` file from `models/phase1` and `models/phase2` the first
time it is needed, so the first scan is slow. A model whose file is missing is shown as
"not trained yet" and is skipped.

## Running it (three terminals, PowerShell)

Node is not on PATH on this machine, so each Node terminal first adds it.
The model service is started with `python -m uvicorn` because Windows Device Guard blocks `uvicorn.exe`.

### Terminal 1, model service

Mock mode (env `ai-web`):

```powershell
cd X:\AI-Projects\Chest-Xray-Pneumonia-Detection\webapp\model-service
$env:MOCK_MODELS="1"; $env:MOCK_ALL_AVAILABLE="1"
conda run -n ai-web --no-capture-output python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Real mode (env `ai-cpu`, in a fresh terminal so no mock variables are set):

```powershell
cd X:\AI-Projects\Chest-Xray-Pneumonia-Detection\webapp\model-service
conda run -n ai-cpu --no-capture-output python -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
```

Real mode needs FastAPI in `ai-cpu`. Check first that installing it would not change TensorFlow,
Keras, NumPy or protobuf:
`conda run -n ai-cpu pip install --dry-run fastapi "uvicorn[standard]" python-multipart httpx`

Other optional settings (set them in the terminal the same way): `PROJECT_ROOT` (repository root,
found automatically), `MAX_CONCURRENT_INFERENCE` (default 1), `TF_INTRA_OP_THREADS` (default 2),
`TF_INTER_OP_THREADS` (default 1). The model service does not read a `.env` file.

### Terminal 2, API

```powershell
$env:Path = "C:\Program Files\nodejs;$env:Path"
cd X:\AI-Projects\Chest-Xray-Pneumonia-Detection\webapp\api
npm run dev
```

Settings are in `api/.env.example` (copy it to `api/.env` to change them). Scans are saved in `webapp/storage/`.

### Terminal 3, website

```powershell
$env:Path = "C:\Program Files\nodejs;$env:Path"
cd X:\AI-Projects\Chest-Xray-Pneumonia-Detection\webapp\frontend
npm run dev
```

Then open http://127.0.0.1:3000. The only setting is `API_URL` in `frontend/.env.example`
(copy it to `frontend/.env.local` to change it).

### Stopping

Press `Ctrl+C` in each terminal. To check that the ports are free afterwards:

```powershell
netstat -ano | findstr /R ":3000 .*LISTENING :4000 .*LISTENING :8000 .*LISTENING"
```

No output means all three ports are free.

## Running the tests

```powershell
# model service (mock mode, no TensorFlow)
cd webapp\model-service; conda run -n ai-web python -m pytest

# API and website (after adding Node to PATH as above)
cd webapp\api;      npx tsc --noEmit; npx eslint .; npx vitest run
cd webapp\frontend; npx tsc --noEmit; npx eslint .; npx vitest run
```

## Where the numbers come from

- **Predictions** come live from the model service.
- **Thresholds**: each model uses its own decision threshold, tuned on the 553-image validation set
  in `03_evaluation.ipynb` and read from `results/comparison.csv`. The label is Pneumonia when the
  probability is above that threshold. If the CSV (or a model's row) is missing, 0.5 is used and the
  model is marked "uncalibrated threshold".
- **Benchmarks** (macro F1, recalls, ROC-AUC with 95% confidence intervals) are read from
  `results/comparison.csv` and `results/comparison_ci.csv`. They are test-set results at the
  validation-tuned threshold. Nothing is hard-coded in the app; if a file is missing the page says
  "not available yet".

## Limitations

- **One dataset.** All models were trained on one public pediatric chest X-ray dataset. They may not
  generalise to adults, other hospitals, other scanners or other image formats.
- **Not a medical device.** This is an educational demo. It must not be used for diagnosis.
- **Phase 1 resize mismatch (kept on purpose).** The Phase 1 K-Fold models were trained with images
  resized by OpenCV (bilinear), but `03_evaluation.ipynb` evaluates them with Keras'
  `flow_from_directory`, which uses nearest-neighbour resizing by default. The web app serves them with
  **nearest**, like the evaluation notebook, because the thresholds were tuned on that pipeline.
  Changing it would make the stored thresholds and benchmarks no longer match. Interpolation is a
  per-model setting in `model-service/app/registry.py`, so it can follow the notebook if it changes.
- **Thresholds are tuned on the validation set only.** The test set is used once, for the reported
  benchmarks, never to pick a threshold.
- **Benchmark numbers come from `results/comparison.csv`.** They are only as current as the last run of
  `03_evaluation.ipynb`. If a model file is newer than the CSV, its threshold may be out of date until
  the notebook is run again.
- **Grad-CAM only for the Phase 2 fine-tuned model.** The other models show "heatmap not available".
  A heatmap shows where the model looked, not whether it was right.
- **EXIF rotation is ignored**, like in training: images are shown and processed in the pixel order
  stored in the file, so a phone photo with a rotation flag may appear sideways.
