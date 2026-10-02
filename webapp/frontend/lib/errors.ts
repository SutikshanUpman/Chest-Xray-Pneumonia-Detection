// Turns API error codes into friendly messages for the user.
// The API always answers errors as { error: { code, message } }; api.ts wraps that in ApiRequestError.

export class ApiRequestError extends Error {
  constructor(
    public status: number, // 0 means the request never reached the API
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

const MESSAGES: Record<string, string> = {
  UNSUPPORTED_TYPE: "Only JPEG and PNG images are supported.",
  UNSUPPORTED_BIT_DEPTH: "This image is 16-bit or floating point. Please use an 8-bit JPEG or PNG.",
  FILE_TOO_LARGE: "The image is larger than 10 MB.",
  IMAGE_TOO_LARGE: "The image has too many pixels. Please use a smaller image.",
  INVALID_IMAGE: "The file could not be read as an image. It may be damaged.",
  EMPTY_FILE: "The file is empty.",
  MODELS_UNAVAILABLE: "One of the chosen models is not trained yet. Reload the page and choose again.",
  MODEL_SERVICE_DOWN: "The model service is not running. Start it and try again.",
  MODEL_SERVICE_TIMEOUT: "The model service took too long. The first run loads the models and can take a while, so try again.",
  MODEL_SERVICE_ERROR: "The model service returned an error. Please try again.",
  NOT_FOUND: "This scan does not exist. It may have been deleted.",
  NETWORK: "Cannot reach the web API. Check that it is running and try again.",
};

const GENERIC = "Something went wrong. Please try again.";

// Codes where trying the same request again can help (the problem is not the image itself)
const RETRYABLE = new Set(["MODEL_SERVICE_DOWN", "MODEL_SERVICE_TIMEOUT", "MODEL_SERVICE_ERROR", "NETWORK"]);

export function friendlyMessage(err: unknown): string {
  if (err instanceof ApiRequestError) return MESSAGES[err.code] ?? GENERIC;
  return GENERIC;
}

export function canRetry(err: unknown): boolean {
  return err instanceof ApiRequestError ? RETRYABLE.has(err.code) : true;
}
