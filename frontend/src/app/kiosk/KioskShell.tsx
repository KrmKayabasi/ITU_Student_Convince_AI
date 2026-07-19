"use client";

/** Kiosk chrome: full-viewport grid (never scrolls) + ambient background. */
export function KioskShell({ children }: { children: React.ReactNode }) {
  return (
    <main
      className="relative grid h-dvh w-full select-none grid-rows-[auto_1fr_minmax(20dvh,auto)_auto] overflow-hidden"
      style={{
        background:
          "radial-gradient(120% 90% at 50% 30%, var(--k-bg-2) 0%, var(--k-bg-1) 45%, var(--k-bg-0) 100%)",
      }}
    >
      {/* soft vignette */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(90% 70% at 50% 45%, transparent 55%, rgba(4,8,18,0.55) 100%)",
        }}
      />
      {children}
    </main>
  );
}
