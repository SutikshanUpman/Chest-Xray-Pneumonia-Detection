import { mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { isUuid } from "./validation.js";

export interface StoredFile {
  name: string; // fixed names chosen by the API (original.jpg, heatmap_<modelId>.png), never user input
  data: Buffer;
}

/** Files live in <root>/scans/<uuid>/. The DB stores paths relative to <root>. */
export class Storage {
  readonly root: string;

  constructor(root: string) {
    this.root = path.resolve(root);
  }

  scanDir(id: string): string {
    if (!isUuid(id)) throw new Error(`Invalid scan id: ${id}`);
    return path.join(this.root, "scans", id);
  }

  relativePath(id: string, name: string): string {
    if (!isUuid(id)) throw new Error(`Invalid scan id: ${id}`);
    return path.posix.join("scans", id, name);
  }

  async writeScan(id: string, files: StoredFile[]): Promise<void> {
    const dir = this.scanDir(id);
    await mkdir(dir, { recursive: true });
    try {
      for (const file of files) await writeFile(path.join(dir, path.basename(file.name)), file.data, { flag: "wx" });
    } catch (err) {
      await this.removeScan(id);
      throw err;
    }
  }

  async removeScan(id: string): Promise<void> {
    await rm(this.scanDir(id), { recursive: true, force: true });
  }

  /** Absolute path of a stored file, or null if it would point outside this scan's own folder. */
  resolveScanFile(id: string, relative: string): string | null {
    const dir = this.scanDir(id);
    const abs = path.resolve(this.root, relative);
    return abs.startsWith(dir + path.sep) ? abs : null;
  }
}
