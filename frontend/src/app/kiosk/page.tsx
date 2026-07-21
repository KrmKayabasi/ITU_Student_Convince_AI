"use client";

import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useSearchParams } from "next/navigation";

import { useRealtimeSession, type TranscriptLine } from "./useRealtimeSession";
import { useWebcamStream } from "./useWebcamStream";
import { useCvSignals } from "./useCvSignals";
import {
  deriveFaceState,
  useThinkingHint,
  type FaceState,
} from "./faceState";
import {
  createAnalyserAmplitude,
  createFakeAmplitude,
  type FakeAmplitudeKind,
} from "./amplitude";
import { type AvatarMode } from "./FaceStage";

import { KioskShell } from "./KioskShell";
import { KioskHeader } from "./KioskHeader";
import { FaceStage } from "./FaceStage";
import { SubtitlePanel } from "./SubtitlePanel";
import { AttractOverlay } from "./AttractOverlay";
import { SessionControls } from "./SessionControls";
import { WebcamPreview } from "./WebcamPreview";
import { DemoPanel } from "./DemoPanel";
import { ProfessorSearchPanel } from "./ProfessorSearchPanel";

function newSessionId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `kiosk-${Date.now()}-${Math.floor(Math.random() * 1e6)}`;
}

/* ── Production kiosk ─────────────────────────────────────────────────────── */

function ProductionKiosk({ avatarMode }: { avatarMode: AvatarMode }) {
  const session = useRealtimeSession();
  const webcam = useWebcamStream();
  const cv = useCvSignals();
  const [started, setStarted] = useState(false);
  const sessionIdRef = useRef("");

  const thinking = useThinkingHint(session.userText, session.assistantSpeaking);

  const start = useCallback(async () => {
    const id = newSessionId();
    sessionIdRef.current = id;
    setStarted(true);
    webcam.start(id);
    cv.start(id);
    await session.connect(id);
  }, [session, webcam, cv]);

  const stop = useCallback(() => {
    session.disconnect();
    webcam.stop();
    cv.stop();
    setStarted(false);
  }, [session, webcam, cv]);

  const amplitude = useMemo(
    () => createAnalyserAmplitude(session.outputAnalyserRef),
    [session.outputAnalyserRef]
  );

  const faceState = deriveFaceState({
    started,
    status: session.status,
    assistantSpeaking: session.assistantSpeaking,
    isFocused: cv.isFocused,
    thinking,
  });

  return (
    <KioskShell>
      <KioskHeader
        status={session.status}
        isFocused={cv.isFocused}
        started={started}
      />
      <FaceStage
        faceState={faceState}
        amplitude={amplitude}
        seekAttentionNonce={session.seekAttentionNonce}
        avatarMode={avatarMode}
        emotion={session.emotion}
      />
      <SubtitlePanel
        assistantText={session.assistantText}
        userText={session.userText}
        history={session.history}
      />
      <ProfessorSearchPanel search={session.professorSearch} />
      {!started && <AttractOverlay onStart={start} />}
      <SessionControls
        started={started}
        status={session.status}
        errorMessage={session.errorMessage}
        onStop={stop}
        onRetry={start}
      />
      <WebcamPreview videoRef={webcam.videoRef} active={started} />
    </KioskShell>
  );
}

/* ── Demo kiosk (/kiosk?demo=1 — no backend) ──────────────────────────────── */

const DEMO_LINES: TranscriptLine[] = [
  { role: "user", text: "Sıralamam 1200 civarı, Bilgisayar tutar mı?" },
  {
    role: "assistant",
    text: "Geçen yılki taban sıralama 1435'ti — senin için gayet ulaşılabilir görünüyor!",
  },
];

const DEMO_STREAM =
  "Harika bir soru! İTÜ Bilgisayar Mühendisliği'nin 2025 taban sıralaması yaklaşık 1435'ti; senin sıralamanla oldukça şanslısın. Peki yazılım mı yapay zekâ mı — hangisi seni daha çok heyecanlandırıyor?";

function DemoKiosk({
  initialState,
  avatarMode: initialAvatarMode,
}: {
  initialState: FaceState;
  avatarMode: AvatarMode;
}) {
  const [faceState, setFaceState] = useState<FaceState>(initialState);
  const [ampKind, setAmpKind] = useState<FakeAmplitudeKind>("speech");
  const [focused, setFocused] = useState(true);
  const [fakeSubtitles, setFakeSubtitles] = useState(false);
  const [nonce, setNonce] = useState(0);
  const [streamedText, setStreamedText] = useState("");
  const [emotion, setEmotion] = useState<string>("neutral");
  const [avatarMode, setAvatarMode] = useState<AvatarMode>(initialAvatarMode);

  const amplitude = useMemo(() => createFakeAmplitude(ampKind), [ampKind]);

  // Fake subtitle stream to test layout stability while "speaking".
  useEffect(() => {
    if (!fakeSubtitles) {
      setStreamedText("");
      return;
    }
    let i = 0;
    const id = window.setInterval(() => {
      i = (i + 3) % (DEMO_STREAM.length + 30);
      setStreamedText(DEMO_STREAM.slice(0, i));
    }, 90);
    return () => window.clearInterval(id);
  }, [fakeSubtitles]);

  const shownState: FaceState =
    !focused && faceState !== "attract" ? "concerned" : faceState;

  return (
    <KioskShell>
      <KioskHeader
        status={faceState === "attract" ? "idle" : "active"}
        isFocused={focused}
        started={faceState !== "attract"}
      />
      <FaceStage
        faceState={shownState}
        amplitude={amplitude}
        seekAttentionNonce={nonce}
        avatarMode={avatarMode}
        emotion={emotion}
      />
      <SubtitlePanel
        assistantText={fakeSubtitles ? streamedText : ""}
        userText=""
        history={fakeSubtitles || faceState === "attract" ? [] : DEMO_LINES}
      />
      {faceState === "attract" && (
        <AttractOverlay onStart={() => setFaceState("listening")} />
      )}
      <SessionControls
        started={faceState !== "attract"}
        status="active"
        errorMessage={null}
        onStop={() => setFaceState("attract")}
        onRetry={() => {}}
      />
      <DemoPanel
        faceState={faceState}
        setFaceState={setFaceState}
        ampKind={ampKind}
        setAmpKind={setAmpKind}
        focused={focused}
        setFocused={setFocused}
        fakeSubtitles={fakeSubtitles}
        setFakeSubtitles={setFakeSubtitles}
        triggerSeek={() => setNonce((n) => n + 1)}
        emotion={emotion}
        setEmotion={setEmotion}
        avatarMode={avatarMode}
        setAvatarMode={setAvatarMode}
      />
    </KioskShell>
  );
}

/* ── Entry (Suspense required for useSearchParams in static builds) ───────── */

const FACE_STATES: FaceState[] = [
  "attract",
  "connecting",
  "listening",
  "speaking",
  "thinking",
  "concerned",
];

function resolveAvatarMode(params: URLSearchParams): AvatarMode {
  // Live2D is the DEFAULT avatar. The SVG "Elif" face is available as an
  // explicit fallback via ?avatar=svg (e.g. for machines without WebGL).
  const fromEnv =
    typeof process !== "undefined" && process.env.NEXT_PUBLIC_AVATAR === "svg"
      ? "svg"
      : "live2d";
  const q = params.get("avatar");
  if (q === "svg") return "svg";
  if (q === "live2d") return "live2d";
  return fromEnv as AvatarMode;
}

function KioskRouter() {
  const params = useSearchParams();
  const demo = params.get("demo") === "1";
  const stateParam = params.get("state") as FaceState | null;
  const initialState =
    stateParam && FACE_STATES.includes(stateParam) ? stateParam : "attract";
  const avatarMode = resolveAvatarMode(params);
  return demo ? (
    <DemoKiosk initialState={initialState} avatarMode={avatarMode} />
  ) : (
    <ProductionKiosk avatarMode={avatarMode} />
  );
}

export default function KioskPage() {
  return (
    <Suspense fallback={<KioskShell>{null}</KioskShell>}>
      <KioskRouter />
    </Suspense>
  );
}
