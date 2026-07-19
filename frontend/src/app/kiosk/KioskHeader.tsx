"use client";

import { memo } from "react";
import type { SessionStatus } from "./useRealtimeSession";

interface KioskHeaderProps {
  status: SessionStatus;
  isFocused: boolean;
  started: boolean;
}

const STATUS_TR: Record<SessionStatus, { label: string; color: string }> = {
  idle: { label: "hazır", color: "var(--k-ink-dim)" },
  connecting: { label: "bağlanıyor…", color: "var(--k-amber)" },
  active: { label: "görüşme aktif", color: "var(--k-ok)" },
  error: { label: "bağlantı hatası", color: "var(--k-danger)" },
};

function Pill({ color, children }: { color: string; children: React.ReactNode }) {
  return (
    <span
      className="flex items-center gap-2 rounded-full px-3 py-1 text-sm font-[550]"
      style={{ background: "rgba(255,255,255,0.06)", color: "var(--k-ink)" }}
    >
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: color }}
      />
      {children}
    </span>
  );
}

export const KioskHeader = memo(function KioskHeader({
  status,
  isFocused,
  started,
}: KioskHeaderProps) {
  const s = STATUS_TR[status];
  return (
    <header className="z-10 flex items-center justify-between px-8 py-4">
      <div className="flex items-baseline gap-3">
        <span className="relative text-3xl font-[900] tracking-tight text-[var(--k-ink)]">
          İTÜ
          <span
            aria-hidden
            className="absolute -bottom-1 left-0 h-[3px] w-full rounded-full"
            style={{ background: "var(--k-amber)" }}
          />
        </span>
        <span className="text-lg font-[500] text-[var(--k-ink-dim)]">
          Yapay Zekâ Tercih Danışmanı
        </span>
      </div>
      <div className="flex items-center gap-2">
        {started && status === "active" && (
          <Pill color={isFocused ? "var(--k-ok)" : "var(--k-coral)"}>
            {isFocused ? "seninle" : "buradayım 👋"}
          </Pill>
        )}
        <Pill color={s.color}>{s.label}</Pill>
      </div>
    </header>
  );
});
