"use client";

// File picker with drag and drop, a preview and a Remove button.
// It only checks the file type and size; the server does the real validation (magic bytes).
import { useState } from "react";

const MAX_BYTES = 10 * 1024 * 1024;

export default function UploadBox({ onChange }: { onChange: (file: File | null) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  function choose(picked: File | undefined) {
    if (!picked) return;
    if (picked.type !== "image/jpeg" && picked.type !== "image/png") {
      setProblem("Only JPEG and PNG images are supported.");
      return;
    }
    if (picked.size > MAX_BYTES) {
      setProblem("The image is larger than 10 MB.");
      return;
    }
    setProblem(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(picked);
    setPreviewUrl(URL.createObjectURL(picked));
    onChange(picked);
  }

  function remove() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setProblem(null);
    onChange(null);
  }

  return (
    <div>
      {file && previewUrl ? (
        <div className="flex items-center gap-4">
          {/* image-orientation: none shows the pixels the way the models see them (EXIF rotation is ignored) */}
          <img
            src={previewUrl}
            alt={`Preview of ${file.name}`}
            style={{ imageOrientation: "none" }}
            className="h-40 w-40 rounded border border-slate-700 bg-black object-contain"
          />
          <div>
            <p className="font-medium">{file.name}</p>
            <p className="text-sm text-slate-300">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            <button onClick={remove} className="mt-3 rounded bg-slate-700 px-3 py-1 hover:bg-slate-600">
              Remove
            </button>
          </div>
        </div>
      ) : (
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => {
            e.preventDefault();
            choose(e.dataTransfer.files[0]);
          }}
          className="rounded border-2 border-dashed border-slate-600 p-8 text-center"
        >
          <p className="mb-3">Drag and drop a chest X-ray here (JPEG or PNG, up to 10 MB)</p>
          <label className="inline-block cursor-pointer rounded bg-slate-700 px-4 py-2 hover:bg-slate-600 focus-within:outline focus-within:outline-2 focus-within:outline-blue-400">
            Choose image
            <input
              type="file"
              accept="image/jpeg,image/png"
              className="sr-only"
              onChange={(e) => {
                choose(e.target.files?.[0]);
                e.target.value = ""; // lets the user pick the same file again after Remove
              }}
            />
          </label>
        </div>
      )}
      {problem && (
        <p role="alert" className="mt-3 rounded border border-red-500 bg-red-950 p-3">
          {problem}
        </p>
      )}
    </div>
  );
}
