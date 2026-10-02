"use client";

// History page: past scans, newest first, with Load more and Delete.
import Link from "next/link";
import { useEffect, useState } from "react";
import ErrorBox from "@/components/ErrorBox";
import ScanListItem from "@/components/ScanListItem";
import { deleteScan, listScans } from "@/lib/api";
import { friendlyMessage } from "@/lib/errors";
import type { ScanSummaryDto } from "@/lib/types";

export default function HistoryPage() {
  const [scans, setScans] = useState<ScanSummaryDto[]>([]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [reloads, setReloads] = useState(0);

  useEffect(() => {
    listScans(null).then(
      (page) => {
        setScans(page.items);
        setNextCursor(page.nextCursor);
        setLoaded(true);
      },
      setError,
    );
  }, [reloads]);

  async function loadMore() {
    try {
      const page = await listScans(nextCursor);
      setScans((old) => [...old, ...page.items]);
      setNextCursor(page.nextCursor);
    } catch (err) {
      setError(err);
    }
  }

  async function remove(scan: ScanSummaryDto) {
    if (!window.confirm("Delete this scan? The image and its results are removed.")) return;
    try {
      await deleteScan(scan.id);
      setScans((old) => old.filter((s) => s.id !== scan.id));
    } catch (err) {
      setError(err);
    }
  }

  return (
    <div>
      <h1 className="text-3xl font-bold">History</h1>

      {error !== null && (
        <div className="mt-4">
          <ErrorBox
            message={friendlyMessage(error)}
            onRetry={() => {
              setError(null);
              setReloads((n) => n + 1);
            }}
          />
        </div>
      )}

      {loaded && scans.length === 0 && (
        <p className="mt-6 text-slate-300">
          No scans yet.{" "}
          <Link href="/" className="text-blue-400 hover:underline">
            Upload an X-ray
          </Link>
        </p>
      )}

      <ul className="mt-6 space-y-3">
        {scans.map((scan) => (
          <ScanListItem key={scan.id} scan={scan} onDelete={() => remove(scan)} />
        ))}
      </ul>

      {nextCursor && (
        <button onClick={loadMore} className="mt-6 rounded bg-slate-700 px-4 py-2 hover:bg-slate-600">
          Load more
        </button>
      )}
    </div>
  );
}
