"use client";

import type { FaceState } from "./faceState";
import type { FakeAmplitudeKind } from "./amplitude";
import type { AvatarMode } from "./FaceStage";

const STATES: FaceState[] = [
  "attract",
  "connecting",
  "listening",
  "speaking",
  "thinking",
  "concerned",
];

// Emotion categories the rig maps to Live2D expression deltas (live2dExpressions.ts).
// The first option must stay "neutral" — that is the rest/idle value.
const EMOTIONS = [
  "neutral",
  "joy",
  "sadness",
  "anger",
  "fear",
  "surprise",
  "curiosity",
] as const;

export interface DemoControls {
  faceState: FaceState;
  setFaceState: (s: FaceState) => void;
  ampKind: FakeAmplitudeKind;
  setAmpKind: (k: FakeAmplitudeKind) => void;
  focused: boolean;
  setFocused: (f: boolean) => void;
  fakeSubtitles: boolean;
  setFakeSubtitles: (f: boolean) => void;
  triggerSeek: () => void;
  emotion: string;
  setEmotion: (e: string) => void;
  avatarMode: AvatarMode;
  setAvatarMode: (m: AvatarMode) => void;
}

/** Left control rail for /kiosk?demo=1 — visual QA without any backend. */
export function DemoPanel(c: DemoControls) {
  return (
    <aside
      className="absolute left-4 top-1/2 z-40 flex w-52 -translate-y-1/2 flex-col gap-3 rounded-2xl p-4 text-sm"
      style={{ background: "rgba(10,17,32,0.88)", border: "1px solid rgba(148,163,189,0.25)" }}
    >
      <p className="font-[700] text-[var(--k-amber)]">DEMO MODU</p>

      <div className="flex flex-col gap-1">
        <p className="font-[600] text-[var(--k-ink-dim)]">Avatar</p>
        {(["svg", "live2d"] as AvatarMode[]).map((m) => (
          <label key={m} className="flex items-center gap-2 text-[var(--k-ink)]">
            <input
              type="radio"
              name="avatar-mode"
              checked={c.avatarMode === m}
              onChange={() => c.setAvatarMode(m)}
            />
            {m}
          </label>
        ))}
      </div>

      <div className="flex flex-col gap-1">
        <p className="font-[600] text-[var(--k-ink-dim)]">Yüz durumu</p>
        {STATES.map((s) => (
          <label key={s} className="flex items-center gap-2 text-[var(--k-ink)]">
            <input
              type="radio"
              name="face-state"
              checked={c.faceState === s}
              onChange={() => c.setFaceState(s)}
            />
            {s}
          </label>
        ))}
      </div>

      <div className="flex flex-col gap-1">
        <p className="font-[600] text-[var(--k-ink-dim)]">Duygu (emotion)</p>
        <select
          value={c.emotion}
          onChange={(e) => c.setEmotion(e.target.value)}
          className="rounded-lg bg-[#0b1426] px-2 py-1 text-[var(--k-ink)]"
          style={{ border: "1px solid rgba(148,163,189,0.3)" }}
        >
          {EMOTIONS.map((em) => (
            <option key={em} value={em}>
              {em}
            </option>
          ))}
        </select>
      </div>

      <button
        onClick={c.triggerSeek}
        className="rounded-lg px-3 py-2 font-[650] text-[#131313]"
        style={{ background: "var(--k-amber)" }}
      >
        seekAttention ▶
      </button>

      <div className="flex flex-col gap-1">
        <p className="font-[600] text-[var(--k-ink-dim)]">Ses (lip-sync)</p>
        {(["speech", "sine", "silent"] as FakeAmplitudeKind[]).map((kind) => (
          <label key={kind} className="flex items-center gap-2 text-[var(--k-ink)]">
            <input
              type="radio"
              name="amp-kind"
              checked={c.ampKind === kind}
              onChange={() => c.setAmpKind(kind)}
            />
            {kind}
          </label>
        ))}
      </div>

      <label className="flex items-center gap-2 text-[var(--k-ink)]">
        <input
          type="checkbox"
          checked={c.focused}
          onChange={(e) => c.setFocused(e.target.checked)}
        />
        öğrenci odaklı
      </label>
      <label className="flex items-center gap-2 text-[var(--k-ink)]">
        <input
          type="checkbox"
          checked={c.fakeSubtitles}
          onChange={(e) => c.setFakeSubtitles(e.target.checked)}
        />
        sahte altyazı akışı
      </label>
    </aside>
  );
}
