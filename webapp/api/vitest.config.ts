import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["test/**/*.test.ts"],
    // One file at a time: training may be using the CPU
    fileParallelism: false,
  },
});
