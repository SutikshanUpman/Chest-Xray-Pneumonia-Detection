// Tests for the error messages (a pure function, no browser needed).
import { describe, expect, it } from "vitest";
import { ApiRequestError, canRetry, friendlyMessage } from "./errors";

const apiError = (code: string) => new ApiRequestError(400, code, "raw server text");

describe("friendlyMessage", () => {
  it("uses our own wording for known codes", () => {
    expect(friendlyMessage(apiError("FILE_TOO_LARGE"))).toBe("The image is larger than 10 MB.");
    expect(friendlyMessage(apiError("UNSUPPORTED_BIT_DEPTH"))).toContain("8-bit JPEG or PNG");
    expect(friendlyMessage(apiError("MODEL_SERVICE_TIMEOUT"))).toContain("first run loads the models");
  });

  it("falls back to a generic message", () => {
    expect(friendlyMessage(apiError("SOMETHING_NEW"))).toBe("Something went wrong. Please try again.");
    expect(friendlyMessage(new Error("boom"))).toBe("Something went wrong. Please try again.");
  });
});

describe("canRetry", () => {
  it("allows retry for service problems but not for a bad image", () => {
    expect(canRetry(apiError("MODEL_SERVICE_DOWN"))).toBe(true);
    expect(canRetry(apiError("NETWORK"))).toBe(true);
    expect(canRetry(apiError("UNSUPPORTED_TYPE"))).toBe(false);
  });
});
