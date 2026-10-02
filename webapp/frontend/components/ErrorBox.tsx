// A red box that shows an error message (announced to screen readers) with an optional Retry button.
export default function ErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-center justify-between gap-4 rounded border border-red-500 bg-red-950 p-4">
      <p>{message}</p>
      {onRetry && (
        <button onClick={onRetry} className="shrink-0 rounded bg-slate-700 px-3 py-1 hover:bg-slate-600">
          Retry
        </button>
      )}
    </div>
  );
}
