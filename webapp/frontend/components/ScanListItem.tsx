// One row of the history page: thumbnail, date, mode, consensus, MOCK tag, and Open / Delete buttons.
import Link from "next/link";
import type { ScanSummaryDto } from "@/lib/types";

export default function ScanListItem({ scan, onDelete }: { scan: ScanSummaryDto; onDelete: () => void }) {
  const { pneumonia, normal, failed } = scan.consensus;
  const date = new Date(scan.createdAt).toLocaleString();

  return (
    <li className="flex items-center gap-4 rounded border border-slate-700 bg-slate-800 p-3">
      {/* No thumbnail is stored, so thumbUrl is the original image, cropped to a square with object-cover */}
      <img
        src={scan.thumbUrl}
        alt={`Scan from ${date}`}
        loading="lazy"
        style={{ imageOrientation: "none" }}
        className="h-16 w-16 shrink-0 rounded bg-black object-cover"
      />
      <div className="flex-1">
        <p className="font-medium">
          {date}
          {scan.mock && <span className="ml-2 rounded bg-yellow-400 px-2 py-0.5 text-xs font-bold text-black">MOCK</span>}
        </p>
        <p className="text-sm text-slate-300">
          {scan.mode === "all" ? "All models" : "Comparative analysis"} · {pneumonia} pneumonia, {normal} normal
          {failed > 0 && `, ${failed} failed`}
        </p>
      </div>
      <Link href={`/scan/${scan.id}`} className="rounded bg-slate-700 px-3 py-1 hover:bg-slate-600">
        Open
      </Link>
      <button onClick={onDelete} className="rounded bg-slate-700 px-3 py-1 hover:bg-red-800">
        Delete
      </button>
    </li>
  );
}
