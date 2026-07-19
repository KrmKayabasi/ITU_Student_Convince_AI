"use client";

import { memo, useState, type RefObject } from "react";

interface WebcamPreviewProps {
  videoRef: RefObject<HTMLVideoElement | null>;
  active: boolean;
}

/**
 * Small self-view feeding the CV pipeline. Collapsing only changes opacity —
 * the <video> must NEVER unmount, because useWebcamStream reads frames from it.
 */
export const WebcamPreview = memo(function WebcamPreview({
  videoRef,
  active,
}: WebcamPreviewProps) {
  const [collapsed, setCollapsed] = useState(false);
  const hidden = collapsed || !active;

  return (
    <div className="absolute bottom-5 right-5 z-30 flex flex-col items-end gap-1">
      <video
        ref={videoRef}
        muted
        playsInline
        className={`w-36 rounded-xl border transition-opacity duration-300 ${
          hidden ? "pointer-events-none opacity-0" : "opacity-80"
        }`}
        style={{ borderColor: "rgba(148,163,189,0.3)", transform: "scaleX(-1)" }}
      />
      {active && (
        <button
          onClick={() => setCollapsed((c) => !c)}
          className="rounded-full px-3 py-1 text-xs font-[550] text-[var(--k-ink-dim)]"
          style={{ background: "rgba(255,255,255,0.07)" }}
        >
          {collapsed ? "kamerayı göster" : "gizle"}
        </button>
      )}
    </div>
  );
});
