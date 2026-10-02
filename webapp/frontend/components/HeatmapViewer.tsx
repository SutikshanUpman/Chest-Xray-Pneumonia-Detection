"use client";

// Shows the uploaded image. If the Phase 2 fine-tuned model produced a Grad-CAM heatmap,
// it can be laid over the image with a checkbox and an opacity slider.
import { useState } from "react";

export default function HeatmapViewer({
  imageUrl,
  heatmapUrl,
  width,
  height,
  mock,
}: {
  imageUrl: string;
  heatmapUrl: string | null;
  width: number;
  height: number;
  mock: boolean;
}) {
  const [show, setShow] = useState(true);
  const [opacity, setOpacity] = useState(0.4); // 0.4 is what the Phase 2 notebook uses for its overlays

  return (
    <figure className="rounded border border-slate-700 bg-slate-800 p-4">
      {/* The aspect ratio keeps the page from jumping while the image loads */}
      <div className="relative bg-black" style={{ aspectRatio: `${width} / ${height}` }}>
        {/* image-orientation: none shows pixels as the model sees them, so the heatmap lines up */}
        <img
          src={imageUrl}
          alt="Uploaded chest X-ray"
          style={{ imageOrientation: "none" }}
          className="absolute inset-0 h-full w-full object-contain"
        />
        {heatmapUrl && show && (
          // plus-lighter adds the heatmap colours to the image, like heat * 0.4 + image in the notebook
          <img
            src={heatmapUrl}
            alt=""
            style={{ mixBlendMode: "plus-lighter", opacity }}
            className="absolute inset-0 h-full w-full object-contain"
          />
        )}
      </div>

      {heatmapUrl ? (
        <div className="mt-3 space-y-2 text-sm">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} />
            Show Grad-CAM heatmap
          </label>
          <label className="flex items-center gap-2">
            Opacity
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={opacity}
              disabled={!show}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="flex-1"
            />
            {Math.round(opacity * 100)}%
          </label>
        </div>
      ) : null}

      <figcaption className="mt-2 text-sm text-slate-300">
        {heatmapUrl
          ? mock
            ? "Heatmap: synthetic (mock), not a real Grad-CAM."
            : "Grad-CAM from the Phase 2 fine-tuned model."
          : "Heatmap not available (only the Phase 2 fine-tuned model makes one)."}
      </figcaption>
    </figure>
  );
}
