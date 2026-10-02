import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fileURLToPath } from "node:url";

export const SCHEMA_VERSION = 1;
const SCHEMA_PATH = fileURLToPath(new URL("../db/schema.sql", import.meta.url));

/**
 * The only way to open the database. Every connection gets PRAGMA foreign_keys = ON
 * (SQLite's default is OFF, per connection), then the schema is applied if the file is new.
 */
export function openDatabase(file: string): DatabaseSync {
  if (file !== ":memory:") mkdirSync(path.dirname(file), { recursive: true });
  const db = new DatabaseSync(file);
  try {
    db.exec("PRAGMA foreign_keys = ON;");
    const { foreign_keys } = db.prepare("PRAGMA foreign_keys").get() as { foreign_keys: number };
    if (foreign_keys !== 1) throw new Error("Could not enable SQLite foreign keys");
    migrate(db);
    return db;
  } catch (err) {
    db.close();
    throw err;
  }
}

function userVersion(db: DatabaseSync): number {
  return (db.prepare("PRAGMA user_version").get() as { user_version: number }).user_version;
}

function migrate(db: DatabaseSync): void {
  const version = userVersion(db);
  if (version === SCHEMA_VERSION) return;
  if (version !== 0) throw new Error(`Unsupported database schema version ${version} (expected ${SCHEMA_VERSION})`);
  transaction(db, () => db.exec(readFileSync(SCHEMA_PATH, "utf8")));
  if (userVersion(db) !== SCHEMA_VERSION) throw new Error("schema.sql did not set the expected user_version");
}

export function transaction<T>(db: DatabaseSync, fn: () => T): T {
  db.exec("BEGIN IMMEDIATE");
  try {
    const result = fn();
    db.exec("COMMIT");
    return result;
  } catch (err) {
    db.exec("ROLLBACK");
    throw err;
  }
}
