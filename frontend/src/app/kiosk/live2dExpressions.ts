/**
 * live2dExpressions — maps go_emotions labels (the 28-class output of
 * SamLowe/roberta-base-go_emotions, ported from jaison-core's emotion_roberta)
 * to parameter *deltas* blended on top of the base FaceState pose.
 *
 * The rig (useLive2DRig.ts) lerps the current deltas toward the target each
 * frame, so emotion changes glide rather than snap. Deltas are unitless and
 * added to the rig's own per-state targets for MouthForm / EyeSmile / brow /
 * eye-open / gaze.
 *
 * Only the params the Haru/Hiyori Cubism 4 sample models expose are touched:
 *   ParamMouthForm   (-1..1, negative = frown, positive = smile)
 *   ParamEyeSmile    (0..1)
 *   ParamBrowLY/RY   (-1..1, up = raised)  — we drive via ParamBrow*
 *   ParamEyeLOpen/R  (0..1, 1 = wide open)
 *   ParamEyeBallX/Y  (-1..1)
 *   ParamCheek       (0..1, blush/warmth)
 *
 * Labels not in the map fall back to NEUTRAL (no deltas). The map collapses
 * the 28 fine-grained labels into ~6 expression families.
 */

export interface ExpressionDeltas {
  /** Added to ParamMouthForm target. */
  mouthForm: number;
  /** Added to ParamEyeSmile target (clamped 0..1). */
  eyeSmile: number;
  /** Added to ParamBrow* target. */
  brow: number;
  /** Multiplier-ish offset for ParamEyeLOpen/R (0..1). */
  eyeOpen: number;
  /** Added to ParamEyeBallX. */
  gazeX: number;
  /** Added to ParamEyeBallY. */
  gazeY: number;
  /** Added to ParamCheek (0..1). */
  cheek: number;
}

const ZERO: ExpressionDeltas = {
  mouthForm: 0,
  eyeSmile: 0,
  brow: 0,
  eyeOpen: 0,
  gazeX: 0,
  gazeY: 0,
  cheek: 0,
};

/** Joy family: warm smile, smiling eyes, slight cheek warmth, raised brow. */
const JOY: ExpressionDeltas = {
  mouthForm: 0.8,
  eyeSmile: 0.7,
  brow: 0.25,
  eyeOpen: 0,
  gazeX: 0,
  gazeY: -0.1,
  cheek: 0.4,
};

/** Sadness family: downturned mouth, drooping brow, looking down. */
const SAD: ExpressionDeltas = {
  mouthForm: -0.7,
  eyeSmile: 0,
  brow: -0.45,
  eyeOpen: -0.2,
  gazeX: 0,
  gazeY: -0.5,
  cheek: 0,
};

/** Anger family: lowered furrowed brow, pressed mouth, slight squint. */
const ANGER: ExpressionDeltas = {
  mouthForm: -0.5,
  eyeSmile: 0,
  brow: -0.7,
  eyeOpen: -0.25,
  gazeX: 0,
  gazeY: -0.15,
  cheek: 0,
};

/** Fear / nerves family: wide eyes, slight frown, looking away. */
const FEAR: ExpressionDeltas = {
  mouthForm: -0.2,
  eyeSmile: 0,
  brow: 0.3,
  eyeOpen: 0.5,
  gazeX: 0.4,
  gazeY: -0.3,
  cheek: 0,
};

/** Surprise family: wide eyes, raised brow, small mouth open (handled in rig). */
const SURPRISE: ExpressionDeltas = {
  mouthForm: 0.1,
  eyeSmile: 0,
  brow: 0.6,
  eyeOpen: 0.7,
  gazeX: 0,
  gazeY: -0.2,
  cheek: 0,
};

/** Curiosity / interest: slight smile, raised brow, eyes forward. */
const CURIOSITY: ExpressionDeltas = {
  mouthForm: 0.3,
  eyeSmile: 0.2,
  brow: 0.35,
  eyeOpen: 0.15,
  gazeX: 0,
  gazeY: -0.05,
  cheek: 0.1,
};

// go_emotions 28 labels → families. Anything unmapped → NEUTRAL (ZERO).
const LABEL_MAP: Record<string, ExpressionDeltas> = {
  // joy family
  joy: JOY,
  amusement: JOY,
  excitement: JOY,
  optimism: JOY,
  gratitude: JOY,
  pride: JOY,
  love: JOY,
  caring: JOY,
  approval: JOY,
  admiration: JOY,
  relief: JOY,
  desire: CURIOSITY,
  // sadness family
  sadness: SAD,
  grief: SAD,
  disappointment: SAD,
  remorse: SAD,
  confusion: { ...SAD, brow: 0.25, eyeOpen: 0.2 },
  nervousness: FEAR,
  // anger family
  anger: ANGER,
  annoyance: ANGER,
  disapproval: ANGER,
  disgust: ANGER,
  // fear family
  fear: FEAR,
  embarrassment: FEAR,
  // surprise family
  surprise: SURPRISE,
  realization: SURPRISE,
  // curiosity family
  curiosity: CURIOSITY,
  // neutral
  neutral: ZERO,
};

/** Resolve a go_emotions label (or demo category) to expression deltas.
 *  Unknown labels collapse to NEUTRAL so the avatar never glitches. */
export function emotionToDeltas(label: string | undefined | null): ExpressionDeltas {
  if (!label) return ZERO;
  const key = label.trim().toLowerCase();
  return LABEL_MAP[key] ?? ZERO;
}

export const NEUTRAL_DELTAS = ZERO;
