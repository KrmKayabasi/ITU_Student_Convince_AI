"use client";

import type { RefObject } from "react";

/**
 * AmplitudeSource — the face rig's only audio dependency.
 * Production wraps the assistant-output AnalyserNode; demo mode substitutes a
 * synthetic source. `read()` returns a perceptual 0..1 loudness.
 */
export interface AmplitudeSource {
  read(): number;
}

const BUF = new Float32Array(1024);

export function createAnalyserAmplitude(
  ref: RefObject<AnalyserNode | null>
): AmplitudeSource {
  return {
    read() {
      const analyser = ref.current;
      if (!analyser) return 0;
      analyser.getFloatTimeDomainData(BUF);
      let sum = 0;
      for (let i = 0; i < BUF.length; i++) sum += BUF[i] * BUF[i];
      const rms = Math.sqrt(sum / BUF.length);
      return Math.min(1, Math.tanh(rms * 6));
    },
  };
}

export type FakeAmplitudeKind = "speech" | "sine" | "silent";

/**
 * Synthetic amplitude for /kiosk?demo=1.
 * "speech": syllable carrier × random word bursts — convincing babble for
 * tuning the mouth mapping. "sine": steady calibration wave.
 */
export function createFakeAmplitude(kind: FakeAmplitudeKind): AmplitudeSource {
  const t0 = performance.now();
  let gateUntil = 0;
  let gateOpen = true;

  return {
    read() {
      if (kind === "silent") return 0;
      const t = (performance.now() - t0) / 1000;
      if (kind === "sine") {
        return 0.5 + 0.5 * Math.sin(2 * Math.PI * 1.2 * t);
      }
      // "speech": ~4.5 Hz syllables gated into word bursts with pauses.
      const now = performance.now();
      if (now > gateUntil) {
        gateOpen = !gateOpen;
        gateUntil = now + (gateOpen ? 250 + Math.random() * 450 : 120 + Math.random() * 230);
      }
      if (!gateOpen) return 0;
      const syllable = Math.pow(Math.abs(Math.sin(2 * Math.PI * 4.5 * t)), 1.4);
      const noise = 0.08 * Math.random();
      return Math.min(1, 0.85 * syllable + noise);
    },
  };
}
