"use client";

import { memo, useEffect, useState } from "react";

const TAGLINES = [
  "Merhaba! Ben Elif — İTÜ tercih danışmanın.",
  "YKS sıralamana en uygun İTÜ bölümünü birlikte bulalım.",
  "Aklındaki soruları sesli sorabilirsin, seni dinliyorum.",
  "Bilgisayar Müh., Yapay Zekâ ve Veri Müh. ve dahası…",
];

/** Idle attract loop: rotating taglines above the CTA. Mounted only when
 *  the kiosk hasn't started a session. */
export const AttractOverlay = memo(function AttractOverlay({
  onStart,
}: {
  onStart: () => void;
}) {
  const [i, setI] = useState(0);
  useEffect(() => {
    const id = window.setInterval(
      () => setI((v) => (v + 1) % TAGLINES.length),
      4200
    );
    return () => window.clearInterval(id);
  }, []);

  return (
    <div className="z-20 col-start-1 row-start-3 row-end-5 flex flex-col items-center justify-center gap-6 px-8 pb-6">
      <p
        key={i}
        className="k-fade-up max-w-3xl text-center text-[clamp(1.3rem,2.4vw,2rem)] font-[550] text-[var(--k-ink)]"
      >
        {TAGLINES[i]}
      </p>
      <button
        onClick={onStart}
        className="k-attract-glow rounded-full px-12 py-5 text-2xl font-[700] text-[#131313] transition-transform active:scale-95"
        style={{
          background:
            "linear-gradient(135deg, var(--k-amber-soft), var(--k-amber))",
        }}
      >
        Konuşmaya Başla
      </button>
      <p className="text-sm font-[450] text-[var(--k-ink-dim)]">
        Mikrofon ve kamera yalnızca görüşme sırasında kullanılır.
      </p>
    </div>
  );
});
