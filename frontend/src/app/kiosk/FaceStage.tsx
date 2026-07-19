"use client";

import { memo } from "react";
import { AdvisorFace } from "./AdvisorFace";
import type { FaceState } from "./faceState";
import type { AmplitudeSource } from "./amplitude";

interface FaceStageProps {
  faceState: FaceState;
  amplitude: AmplitudeSource;
  seekAttentionNonce: number;
}

const RING: Record<FaceState, string> = {
  attract: "var(--k-ring-idle)",
  connecting: "var(--k-amber-soft)",
  listening: "var(--k-ring-listening)",
  speaking: "var(--k-ring-speaking)",
  thinking: "var(--k-ring-listening)",
  concerned: "var(--k-ring-concerned)",
};

export const FaceStage = memo(function FaceStage({
  faceState,
  amplitude,
  seekAttentionNonce,
}: FaceStageProps) {
  const ring = RING[faceState];
  return (
    <section className="z-10 row-start-2 flex items-center justify-center">
      <div
        className="relative flex items-center justify-center"
        style={{ width: "min(55dvh, 70vw)", aspectRatio: "420 / 560" }}
      >
        {/* ambient glow behind the face */}
        <div
          aria-hidden
          className="absolute inset-[-18%] rounded-full"
          style={{
            background:
              "radial-gradient(50% 50% at 50% 42%, rgba(242,169,59,0.14) 0%, rgba(29,74,148,0.12) 45%, transparent 75%)",
          }}
        />
        {/* thin state ring */}
        <div
          aria-hidden
          className="absolute inset-[-4%] rounded-full border-2 transition-colors duration-500"
          style={{ borderColor: ring, opacity: 0.55 }}
        />
        {/* seekAttention pulse — remounts (and replays) on each nonce */}
        {seekAttentionNonce > 0 && (
          <div
            key={seekAttentionNonce}
            aria-hidden
            className="k-pulse-ring absolute inset-[-4%] rounded-full border-4"
            style={{ borderColor: "var(--k-amber)" }}
          />
        )}
        <AdvisorFace
          className="relative h-full w-full"
          state={faceState}
          amplitude={amplitude}
          seekAttentionNonce={seekAttentionNonce}
        />
      </div>
    </section>
  );
});
