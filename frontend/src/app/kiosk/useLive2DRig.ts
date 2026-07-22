"use client";

/**
 * useLive2DRig — the Live2D counterpart to useFaceRig.ts.
 *
 * Same FaceState machine, same asymmetric-EMA amplitude lip-sync
 * (fast attack ~40ms / soft release ~110ms), same blink/nod/look-around and
 * seekAttention overlay — but instead of writing SVG attributes we write
 * Cubism 4 parameters on the model's core each frame after its motion update.
 *
 * The rig runs inside Cubism's model-update lifecycle, reading live state
 * through refs so the effect binds once. Lip-sync amplitude is gated to the
 * "speaking" state so barge-in closes the mouth within ~150ms.
 *
 * Parameters written (Cubism 4 standard IDs; Haru/Hiyori samples expose all):
 *   ParamMouthOpenY  ← amplitude (speaking only)         [LipSync group]
 *   ParamMouthForm   ← smile target + emotion delta       [-1..1]
 *   ParamAngleX/Y/Z  ← head pose                          [-30..30]
 *   ParamBodyAngleX  ← body sway
 *   ParamEyeBallX/Y  ← gaze
 *   ParamEyeLOpen/R  ← blink + emotion eye-open
 *   ParamEyeSmile    ← smile/emotion                      [0..1]
 *   ParamBrowLY/RY   ← brow raise / concern + emotion
 *   ParamBreath       ← idle breathing
 *   ParamCheek        ← emotion warmth                     [0..1]
 *
 * Idle motions are NOT suppressed. They run first, then this rig composes the
 * channels it owns (mouth/gaze/brow/blink). The model's physics still runs.
 */

import { useEffect, useRef } from "react";
import type { RefObject } from "react";
import type { FaceState } from "./faceState";
import type { AmplitudeSource } from "./amplitude";
import type { Live2DModelLike } from "./useLive2DModel";
import { emotionToDeltas, type ExpressionDeltas } from "./live2dExpressions";
import type { FacePosition } from "./useCvSignals";

const SEEK_MS = 2600;

export interface Live2DRigProps {
  state: FaceState;
  amplitude: AmplitudeSource;
  seekAttentionNonce: number;
  emotion: string;
  facePosition: FacePosition | null;
}

// Live2D param IDs (Cubism 4 standard).
const P = {
  mouthOpen: "ParamMouthOpenY",
  mouthForm: "ParamMouthForm",
  angleX: "ParamAngleX",
  angleY: "ParamAngleY",
  angleZ: "ParamAngleZ",
  bodyX: "ParamBodyAngleX",
  eyeBallX: "ParamEyeBallX",
  eyeBallY: "ParamEyeBallY",
  eyeLOpen: "ParamEyeLOpen",
  eyeROpen: "ParamEyeROpen",
  eyeSmile: "ParamEyeSmile",
  browLY: "ParamBrowLY",
  browRY: "ParamBrowRY",
  breath: "ParamBreath",
  cheek: "ParamCheek",
} as const;

export function useLive2DRig(
  modelRef: RefObject<Live2DModelLike | null>,
  ready: boolean,
  props: Live2DRigProps
): void {
  const stateRef = useRef<FaceState>(props.state);
  const ampRef = useRef<AmplitudeSource>(props.amplitude);
  const emotionRef = useRef(props.emotion);
  const facePositionRef = useRef(props.facePosition);
  const seekUntilRef = useRef(0);
  const lastNonceRef = useRef(props.seekAttentionNonce);

  useEffect(() => {
    stateRef.current = props.state;
  }, [props.state]);
  useEffect(() => {
    ampRef.current = props.amplitude;
  }, [props.amplitude]);
  useEffect(() => {
    emotionRef.current = props.emotion;
  }, [props.emotion]);
  useEffect(() => {
    facePositionRef.current = props.facePosition;
  }, [props.facePosition]);
  useEffect(() => {
    if (props.seekAttentionNonce !== lastNonceRef.current) {
      lastNonceRef.current = props.seekAttentionNonce;
      seekUntilRef.current = performance.now() + SEEK_MS;
    }
  }, [props.seekAttentionNonce]);

  useEffect(() => {
    if (!ready) return;

    const model = modelRef.current;
    if (!model) return;
    const internalModel = model.internalModel;
    const core = internalModel.coreModel;

    // Channel state (mirrors useFaceRig.ts).
    const cur = {
      angleX: 0,
      angleY: 0,
      angleZ: 0,
      bodyX: 0,
      brow: 0,
      gazeX: 0,
      gazeY: 0,
      smile: 0.35,
      mouthForm: 0,
      lipShape: 0,
      eyeSmile: 0,
      eyeOpen: 1,
      cheek: 0.2,
      open: 0,
      energy: 0,
    };
    let blinkStart = -1;
    let nextBlink = performance.now() + 1800;
    let nodStart = -1;
    let nextNod = performance.now() + 5000;
    let last = performance.now();
    // Diagnostic: print state + amplitude once per second so we can see WHY
    // the mouth isn't moving (state wrong? amplitude always 0? param write?).
    let nextDiagLog = performance.now();

    const clamp = (v: number, lo: number, hi: number) =>
      Math.max(lo, Math.min(hi, v));
    const writeMouth = () => {
      core.setParameterValueById(P.mouthOpen, clamp(cur.open, 0, 1.1));
      core.setParameterValueById(
        P.mouthForm,
        clamp(cur.mouthForm + cur.lipShape, -1, 1),
      );
    };

    const frame = () => {
      const now = performance.now();
      const dt = Math.min(0.05, Math.max(0, (now - last) / 1000));
      last = now;
      const t = now / 1000;
      const st = stateRef.current;
      const k = (tau: number) => 1 - Math.exp(-dt / tau);

      // ── lip-sync amplitude (asymmetric EMA, gated to speaking) ──
      // Apply a gain curve before smoothing so quiet speech still opens the
      // mouth visibly and loud syllables push it wider. pow(0.6) lifts the
      // mid-range; the *1.15 gain widens the ceiling a touch. Clamped after.
      const ampIn = st === "speaking" ? ampRef.current.read() : 0;
      const rawAmp = Math.min(1.15, Math.pow(ampIn, 0.6) * 1.15);
      cur.open += (rawAmp - cur.open) * k(rawAmp > cur.open ? 0.035 : 0.10);
      cur.energy += (cur.open - cur.energy) * k(0.6);
      // Lip shaping: as the mouth opens, round the lips slightly toward an "O"
      // (negative mouthForm = pursed); as it closes between syllables, let them
      // relax back to the base smile. This makes the motion read as speech
      // rather than a rigid chomp. Tracked separately so it lags the open by a
      // few frames for a natural lip-flip feel.
      const lipTarget = cur.open > 0.15 ? -0.35 * Math.min(1, cur.open) : 0;
      cur.lipShape += (lipTarget - cur.lipShape) * k(0.08);

      // Diagnostic: once per second, dump the values that decide mouth motion.
      if (now >= nextDiagLog) {
        nextDiagLog = now + 1000;
        console.log("[L2D rig]", "state=" + st, "rawAmp=" + rawAmp.toFixed(3), "cur.open=" + cur.open.toFixed(3));
      }

      // ── emotion deltas (blended on top of the base pose) ──
      const e: ExpressionDeltas = emotionToDeltas(emotionRef.current);

      // ── per-state targets ──
      let tAngleX = 0;
      let tAngleY = 0;
      let tAngleZ = 0;
      let tBodyX = 0;
      let tBrow = 0;
      let tGazeX = 0;
      let tGazeY = 0;
      let tSmile = 0.35;
      let tEyeSmile = 0;
      let tEyeOpen = 1;
      let tCheek = 0.2;
      let tMouthFormTarget = 0;

      switch (st) {
        case "attract":
          tSmile = 0.5;
          tEyeSmile = 0.3;
          tAngleZ = 3 * Math.sin(0.31 * t) + 1 * Math.sin(0.83 * t);
          tBodyX = 2 * Math.sin(0.21 * t);
          // Face tracking (below) drives gaze + head angles when a face is
          // detected, overriding the static targets here. When no face is
          // present, the character stays centered with a gentle idle sway.
          break;
        case "connecting":
          tSmile = 0.2;
          tBrow = 0.15;
          tGazeY = -0.15;
          tAngleZ = 2 * Math.sin(1.7 * t);
          break;
        case "listening":
          // Eye contact: face the user directly, eyes locked forward and
          // slightly down (the camera sits below most kiosk screens).
          tAngleX = 0;
          tAngleY = 2;
          tGazeX = 0;
          tGazeY = 0.2;
          tSmile = 0.35;
          tEyeSmile = 0.2;
          if (now > nextNod && nodStart < 0) {
            nodStart = now;
            nextNod = now + 6000 + Math.random() * 3000;
          }
          break;
        case "speaking":
          // Keep eye contact while talking; only a gentle natural sway remains.
          tSmile = 0.3;
          tBrow = 0.1;
          tEyeSmile = 0.15;
          tGazeX = 0.06 * Math.sin(2.1 * t);
          tGazeY = 0.2;
          tAngleX = 0;
          tAngleY = 2 + 3 * cur.open;
          tAngleZ = 1.2 * Math.sin(2 * Math.PI * 1.9 * t) * cur.energy;
          tBodyX = 0.8 * Math.sin(2 * Math.PI * 1.9 * t) * cur.energy;
          break;
        case "thinking":
          tSmile = 0.1;
          tBrow = 0.3;
          tGazeX = 0.4;
          tGazeY = -0.35;
          tAngleZ = 4;
          break;
        case "concerned":
          tBrow = -0.5;
          tSmile = -0.3;
          tCheek = 0.05;
          tEyeOpen = 0.85;
          break;
      }

      // Follow the primary visitor's face from the CV pipeline in EVERY state.
      // A dead zone prevents detector noise around the camera center from
      // making the eyes twitch. When tracking is unavailable, the state-driven
      // targets above provide the fallback pose.
      const trackedFace = facePositionRef.current;
      if (trackedFace) {
        const deadZone = (value: number, radius: number) =>
          Math.abs(value) <= radius
            ? 0
            : Math.sign(value) * (Math.abs(value) - radius) / (1 - radius);
        // getUserMedia frames are unmirrored while the visitor-facing preview
        // is mirrored, so invert X to follow the visitor's physical direction.
        const dx = clamp(deadZone(-(trackedFace.x - 0.5) / 0.5, 0.08), -1, 1);
        const dy = clamp(deadZone((trackedFace.y - 0.5) / 0.5, 0.08), -1, 1);
        tGazeX = dx * 0.7;
        tGazeY = -dy * 0.55;
        tAngleX += dx * 10;
        tAngleY += -dy * 7;
      }

      // ── nod gesture (listening) ──
      if (nodStart >= 0) {
        const p = (now - nodStart) / 550;
        if (p >= 1) nodStart = -1;
        else tAngleY += 8 * Math.sin(Math.PI * p);
      }

      // ── seekAttention overlay ──
      if (now < seekUntilRef.current) {
        const p = 1 - (seekUntilRef.current - now) / SEEK_MS;
        const env = Math.sin(Math.PI * Math.min(1, Math.max(0, p)));
        tAngleY -= 10 * env;
        tBrow = Math.max(tBrow, env);
        tSmile = Math.max(tSmile, 0.6 * env);
        tEyeSmile = Math.max(tEyeSmile, 0.4 * env);
        if (p < 0.05 && blinkStart < 0) blinkStart = now; // attention blink
      }

      // ── apply emotion deltas to the targets (blended in) ──
      tMouthFormTarget = tSmile + e.mouthForm;
      tEyeSmile = Math.max(0, Math.min(1, tEyeSmile + e.eyeSmile));
      tBrow += e.brow;
      tEyeOpen = Math.max(0.1, Math.min(1.3, tEyeOpen + e.eyeOpen));
      tGazeX += e.gazeX;
      tGazeY += e.gazeY;
      tCheek = Math.max(0, Math.min(1, tCheek + e.cheek));

      // ── smoothing toward targets ──
      cur.angleX += (tAngleX - cur.angleX) * k(0.3);
      cur.angleY += (tAngleY - cur.angleY) * k(0.2);
      cur.angleZ += (tAngleZ - cur.angleZ) * k(0.3);
      cur.bodyX += (tBodyX - cur.bodyX) * k(0.4);
      cur.brow += (tBrow - cur.brow) * k(0.15);
      cur.gazeX += (tGazeX - cur.gazeX) * k(0.12);
      cur.gazeY += (tGazeY - cur.gazeY) * k(0.12);
      cur.smile += (tSmile - cur.smile) * k(0.25);
      cur.eyeSmile += (tEyeSmile - cur.eyeSmile) * k(0.25);
      cur.eyeOpen += (tEyeOpen - cur.eyeOpen) * k(0.2);
      cur.cheek += (tCheek - cur.cheek) * k(0.5);
      cur.mouthForm += (tMouthFormTarget - cur.mouthForm) * k(0.25);

      // ── blink keyframes ──
      if (blinkStart < 0 && now >= nextBlink) {
        if (st === "speaking" && cur.open > 0.5) {
          nextBlink = now + 300; // don't blink mid-loud-vowel
        } else {
          blinkStart = now;
          nextBlink = now + (st === "attract" ? 3400 : 2800) + Math.random() * 3200;
        }
      }
      let blink = 0;
      if (blinkStart >= 0) {
        const bt = now - blinkStart;
        if (bt < 70) blink = bt / 70;
        else if (bt < 110) blink = 1;
        else if (bt < 240) blink = 1 - (bt - 110) / 130;
        else {
          blink = 0;
          blinkStart = -1;
        }
      }
      const eyeOpen = Math.max(0, cur.eyeOpen * (1 - blink));

      // ── breath ──
      const breath = (st === "attract" ? 0.6 : 0.4) * (0.5 + 0.5 * Math.sin(2 * Math.PI * 0.22 * t));

      // ── write params (clamped to Live2D ranges) ──
      try {
        writeMouth();
        core.setParameterValueById(P.angleX, clamp(cur.angleX, -30, 30));
        core.setParameterValueById(P.angleY, clamp(cur.angleY, -30, 30));
        core.setParameterValueById(P.angleZ, clamp(cur.angleZ, -30, 30));
        core.setParameterValueById(P.bodyX, clamp(cur.bodyX, -10, 10));
        core.setParameterValueById(P.eyeBallX, clamp(cur.gazeX, -1, 1));
        core.setParameterValueById(P.eyeBallY, clamp(cur.gazeY, -1, 1));
        core.setParameterValueById(P.eyeLOpen, clamp(eyeOpen, 0, 1));
        core.setParameterValueById(P.eyeROpen, clamp(eyeOpen, 0, 1));
        core.setParameterValueById(P.eyeSmile, clamp(cur.eyeSmile, 0, 1));
        core.setParameterValueById(P.browLY, clamp(cur.brow, -1, 1));
        core.setParameterValueById(P.browRY, clamp(cur.brow, -1, 1));
        core.setParameterValueById(P.breath, clamp(breath, 0, 1));
        core.setParameterValueById(P.cheek, clamp(cur.cheek, 0, 1));
      } catch {
        // A model may not expose every standard param; ignore write failures.
      }
    };

    // Motions (including Idle) run before this callback. Reassert the mouth at
    // the final pre-draw stage as well, so expressions/physics can never close
    // it after the audio-driven value has been composed.
    const writeFinalMouth = () => {
      try {
        writeMouth();
      } catch {
        // The model may be in the middle of disposal.
      }
    };
    internalModel.on("afterMotionUpdate", frame);
    internalModel.on("beforeModelUpdate", writeFinalMouth);

    return () => {
      try {
        internalModel.off("afterMotionUpdate", frame);
        internalModel.off("beforeModelUpdate", writeFinalMouth);
      } catch {
        /* ignore */
      }
    };
    // Bind once when the model becomes ready; live values flow through refs.
  }, [ready, modelRef]);
}
