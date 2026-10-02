// Vitest settings: runs the small unit tests in lib/.
import { defineConfig } from "vitest/config";

// Unit tests for pure helpers in lib/ only (no browser, no jsdom).
export default defineConfig({
  test: {
    environment: "node",
    include: ["lib/**/*.test.ts"],
    fileParallelism: false,
  },
});
