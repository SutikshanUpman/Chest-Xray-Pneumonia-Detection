"use client";

// Home page: upload an X-ray, then run all models or a selection of models.
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import ErrorBox from "@/components/ErrorBox";
import ModelPicker from "@/components/ModelPicker";
import UploadBox from "@/components/UploadBox";
import { createScan, getModels } from "@/lib/api";
import { ApiRequestError, canRetry, friendlyMessage } from "@/lib/errors";
import type { ModelsDto, ScanMode } from "@/lib/types";

export default function HomePage() {
  const router = useRouter();
  const [models, setModels] = useState<ModelsDto | null>(null);
  const [apiDown, setApiDown] = useState(false);
  const [reloads, setReloads] = useState(0); // increased by the Retry button to fetch the models again
  const [file, setFile] = useState<File | null>(null);
  const [picking, setPicking] = useState(false);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState<{ err: unknown; mode: ScanMode; ids: string[] } | null>(null);

  useEffect(() => {
    getModels().then(
      (data) => {
        setModels(data);
        setApiDown(false);
      },
      () => setApiDown(true),
    );
  }, [reloads]);

  function retryModels() {
    setModels(null);
    setApiDown(false);
    setReloads((n) => n + 1);
  }

  async function run(mode: ScanMode, ids: string[]) {
    if (!file) return;
    setRunning(true);
    setFailed(null);
    try {
      const scan = await createScan(file, mode, ids);
      router.push(`/scan/${scan.id}`);
    } catch (err) {
      setFailed({ err, mode, ids });
      setRunning(false);
    }
  }

  const online = models?.serviceOnline ? models : null;
  const ready = file !== null && online !== null && !running;

  return (
    <div>
      {online?.mock && (
        <p role="note" className="mb-6 rounded bg-yellow-400 p-3 font-bold text-black">
          MOCK DATA: the model service is in mock mode. Probabilities and heatmaps are fake.
        </p>
      )}

      <h1 className="text-3xl font-bold">Chest X-ray pneumonia demo</h1>
      <p className="mt-2 text-slate-300">
        Upload one chest X-ray and compare what each trained model predicts.
      </p>

      <section className="mt-8 rounded border border-slate-700 bg-slate-800 p-6">
        <UploadBox onChange={setFile} />

        <div className="mt-6 text-sm text-slate-300">
          {apiDown && <ErrorBox message={friendlyMessage(new ApiRequestError(0, "NETWORK", ""))} onRetry={retryModels} />}
          {models && !models.serviceOnline && (
            <ErrorBox message="Model service offline. Start it and press Retry." onRetry={retryModels} />
          )}
          {!models && !apiDown && <p>Checking which models are available…</p>}
          {online && (
            <p>
              {online.available} of {online.total} models available
            </p>
          )}
        </div>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <button
            onClick={() => run("all", [])}
            disabled={!ready || online.available === 0}
            className="rounded bg-blue-600 px-4 py-3 text-lg font-semibold hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Use all models
          </button>
          <button
            onClick={() => setPicking(true)}
            disabled={!ready}
            className="rounded bg-slate-700 px-4 py-3 text-lg font-semibold hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Comparative analysis
          </button>
        </div>
        {!file && <p className="mt-3 text-sm text-slate-300">Choose an image to enable the buttons.</p>}

        {picking && online && <ModelPicker models={online.models} onRun={(ids) => run("selected", ids)} />}

        {running && (
          <p role="status" className="mt-4">
            Running the models… the first run loads them and can take a while.
          </p>
        )}
        {failed && (
          <div className="mt-4">
            <ErrorBox
              message={friendlyMessage(failed.err)}
              onRetry={canRetry(failed.err) ? () => run(failed.mode, failed.ids) : undefined}
            />
          </div>
        )}
      </section>
    </div>
  );
}
