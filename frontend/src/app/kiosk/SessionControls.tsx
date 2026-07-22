"use client";

import { memo } from "react";
import type { SessionStatus } from "./useRealtimeSession";

interface SessionControlsProps {
  started: boolean;
  status: SessionStatus;
  errorMessage: string | null;
  onStop: () => void;
  onRetry: () => void;
}

export const SessionControls = memo(function SessionControls({
  started,
  status,
  errorMessage,
  onStop,
  onRetry,
}: SessionControlsProps) {
  if (!started) return <footer className="col-start-1 row-start-4 h-[8dvh]" />;

  return (
    <footer className="z-10 col-start-1 row-start-4 flex flex-col items-center justify-center gap-2 pb-6">
      {status === "error" && (
        <div className="flex items-center gap-3 rounded-lg px-4 py-2 text-sm font-[500]"
          style={{ background: "rgba(243,108,92,0.12)", color: "var(--k-danger)" }}
        >
          <span>Bağlantı sorunu{errorMessage ? `: ${errorMessage}` : ""}</span>
          <button
            onClick={onRetry}
            className="rounded-md px-3 py-1 font-[650]"
            style={{ background: "var(--k-danger)", color: "#fff" }}
          >
            Tekrar dene
          </button>
        </div>
      )}
      <button
        onClick={onStop}
        className="rounded-full border px-8 py-3 text-lg font-[600] text-[var(--k-ink-dim)] transition-colors hover:text-[var(--k-ink)]"
        style={{ borderColor: "rgba(148,163,189,0.35)" }}
      >
        Görüşmeyi Bitir
      </button>
    </footer>
  );
});
