// Page frame shared by every page: top navigation, content area, and the disclaimer footer.
import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chest X-ray Pneumonia Demo",
  description: "Educational demo comparing CNN and transfer-learning models on chest X-rays.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-slate-900 text-slate-100">
        <header className="border-b border-slate-700">
          <nav aria-label="Main" className="mx-auto flex max-w-4xl gap-6 px-4 py-4">
            <Link href="/" className="font-semibold hover:text-blue-400">
              New scan
            </Link>
            <Link href="/history" className="hover:text-blue-400">
              History
            </Link>
          </nav>
        </header>
        <main className="mx-auto max-w-4xl px-4 py-8">{children}</main>
        <footer className="mx-auto max-w-4xl px-4 pb-8 text-sm text-slate-300">
          <p className="font-semibold">Educational demo, not a medical device. Not for diagnosis.</p>
          <p>Trained on one public pediatric chest X-ray dataset; may not generalise to other populations or scanners.</p>
        </footer>
      </body>
    </html>
  );
}
