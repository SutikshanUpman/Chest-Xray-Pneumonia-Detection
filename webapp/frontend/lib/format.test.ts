// Tests for the number formatting functions (pure functions, no browser needed).
import { describe, expect, it } from "vitest";
import { formatInterval, formatMetric, formatMs, formatPercent, formatThreshold } from "./format";

describe("format", () => {
  it("shows probabilities as percentages", () => {
    expect(formatPercent(0.7234)).toBe("72.3%");
    expect(formatPercent(null)).toBe("—");
  });

  it("rounds thresholds to 2 decimals", () => {
    expect(formatThreshold(0.3900000000000002)).toBe("0.39");
    expect(formatThreshold(0.2)).toBe("0.20");
  });

  it("shows benchmark numbers with the 4 decimals of the CSV", () => {
    expect(formatMetric(0.6397)).toBe("0.6397");
    expect(formatMetric(1)).toBe("1.0000");
    expect(formatInterval({ low: 0.2489, high: 0.3645 })).toBe("[0.2489, 0.3645]");
    expect(formatInterval(null)).toBe("—");
  });

  it("shows inference time", () => {
    expect(formatMs(0.2)).toBe("<1 ms");
    expect(formatMs(43.4)).toBe("43 ms");
  });
});
