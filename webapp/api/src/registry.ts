/**
 * Model ids from webapp/model-service/app/registry.py (checked by test/registry.test.ts).
 * Kept here so ids can be validated even when the model service is down.
 */
export const MODEL_IDS = [
  "class_weights",
  "oversampling",
  "kfold_ensemble",
  "kfoldc_ensemble",
  "phase2_stage1",
  "phase2_stage2",
] as const;

export type ModelId = (typeof MODEL_IDS)[number];

/** Models that produce a Grad-CAM heatmap (Phase 2 fine-tuned only). */
export const GRADCAM_MODEL_IDS: ReadonlySet<string> = new Set<ModelId>(["phase2_stage2"]);

export function isModelId(value: unknown): value is ModelId {
  return typeof value === "string" && (MODEL_IDS as readonly string[]).includes(value);
}
