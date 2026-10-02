// Shown when the URL does not match any page.
import Link from "next/link";

export default function NotFound() {
  return (
    <div>
      <h1 className="text-3xl font-bold">Page not found</h1>
      <p className="mt-2">
        <Link href="/" className="text-blue-400 hover:underline">
          Back to a new scan
        </Link>
      </p>
    </div>
  );
}
