import type { ErrorRequestHandler } from "express";
import multer from "multer";

/** Every error leaves the API as {error: {code, message, details?}}. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details?: unknown,
  ) {
    super(message);
  }
}

export function errorBody(code: string, message: string, details?: unknown) {
  return { error: details === undefined ? { code, message } : { code, message, details } };
}

function fromMulter(err: multer.MulterError): ApiError {
  if (err.code === "LIMIT_FILE_SIZE") return new ApiError(413, "FILE_TOO_LARGE", "The image is larger than 10 MB.");
  return new ApiError(400, "VALIDATION", "Invalid upload.", { reason: err.code, field: err.field });
}

export const errorHandler: ErrorRequestHandler = (err, _req, res, next) => {
  if (res.headersSent) return next(err);
  let apiError: ApiError;
  if (err instanceof ApiError) apiError = err;
  else if (err instanceof multer.MulterError) apiError = fromMulter(err);
  else {
    console.error(err);
    apiError = new ApiError(500, "INTERNAL", "Unexpected server error.");
  }
  res.status(apiError.status).json(errorBody(apiError.code, apiError.message, apiError.details));
};
