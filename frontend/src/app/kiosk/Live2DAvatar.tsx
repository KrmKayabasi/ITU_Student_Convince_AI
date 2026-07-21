"use client";

/**
 * Live2DAvatar — the WebGL counterpart to AdvisorFace.tsx.
 *
 * Drop-in swap inside FaceStage: same prop contract (state, amplitude,
 * seekAttentionNonce) plus an `emotion` label that drives blended expression
 * deltas (see live2dExpressions.ts). Renders a Cubism 4 model on a <canvas>
 * via pixi-live2d-display; lip-sync reuses the SAME AmplitudeSource the SVG
 * rig consumes, so both renderers stay in sync with the playback audio.
 *
 * Rendered only on the client (FaceStage loads it via next/dynamic ssr:false).
 */

import { memo } from "react";
import { useLive2DModel } from "./useLive2DModel";
import { useLive2DRig } from "./useLive2DRig";
import type { FaceState } from "./faceState";
import type { AmplitudeSource } from "./amplitude";

export interface Live2DAvatarProps {
  state: FaceState;
  amplitude: AmplitudeSource;
  seekAttentionNonce: number;
  emotion?: string;
  /** Optional model URL override (defaults to the bundled Haru sample). */
  modelUrl?: string;
  className?: string;
}

function Live2DAvatarInner({
  state,
  amplitude,
  seekAttentionNonce,
  emotion = "neutral",
  modelUrl,
  className,
}: Live2DAvatarProps) {
  const { canvasRef, modelRef, tickerRef, ready, error } = useLive2DModel(modelUrl);

  useLive2DRig(modelRef, tickerRef, ready, {
    state,
    amplitude,
    seekAttentionNonce,
    emotion,
  });

  return (
    <div className={className} style={{ position: "relative" }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
        aria-label="Live2D danışman"
        role="img"
      />
      {!ready && !error && (
        <div
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "rgba(242,169,59,0.7)",
            fontSize: 14,
          }}
        >
          model yükleniyor…
        </div>
      )}
      {error && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "rgba(232,126,108,0.9)",
            fontSize: 12,
            padding: 16,
            textAlign: "center",
          }}
        >
          Live2D yüklenemedi: {error}
        </div>
      )}
    </div>
  );
}

export const Live2DAvatar = memo(Live2DAvatarInner);
