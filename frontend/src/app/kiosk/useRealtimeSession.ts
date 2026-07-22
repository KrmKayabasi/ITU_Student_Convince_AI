"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getOrchestratorWsUrl } from "./urls";

export type SessionStatus = "idle" | "connecting" | "active" | "error";

export interface TranscriptLine {
  role: "user" | "assistant";
  text: string;
}

export interface ProfessorResult {
  name: string;
  title: string;
  department: string;
  work_areas: string;
  summary: string;
  profile_url: string;
  image_url: string;
}

export interface ProfessorSearchState {
  id: string;
  query: string;
  status: "searching" | "completed" | "cancelled" | "error";
  message?: string;
  results: ProfessorResult[];
  sourceName?: string;
  sourceUrl?: string;
}

export interface RealtimeSession {
  status: SessionStatus;
  errorMessage: string | null;
  userText: string;
  assistantText: string;
  history: TranscriptLine[];
  assistantSpeaking: boolean;
  /** Increments each time the orchestrator asks the avatar to grab attention. */
  seekAttentionNonce: number;
  /** Latest emotion label from the orchestrator (go_emotions, e.g. "joy").
   *  "neutral" when emotion classification is disabled or on turn reset. */
  emotion: string;
  professorSearch: ProfessorSearchState | null;
  /** AnalyserNode on the assistant audio output — drives lip-sync. */
  outputAnalyserRef: React.MutableRefObject<AnalyserNode | null>;
  connect: (sessionId: string) => Promise<void>;
  disconnect: () => void;
}

// Input (mic) sample rate — Silero VAD + Gemini both expect 16 kHz PCM16.
const INPUT_TARGET_RATE = 16000;
// Silero v5 operates on fixed 512-sample (32 ms) frames. We size the capture
// worklet to emit exactly this so we can run inference on each frame directly.
const VAD_FRAME_SAMPLES = 512;
const OUTPUT_RATE = 24000;

// ─────────────────────────────────────────────────────────────────────────────
// Client-side Silero VAD gate.
//
// The browser streams mic audio to Gemini continuously. Without a local gate,
// Gemini's server-side VAD hears *everything* — background chatter triggers
// spurious replies, and the assistant's own TTS echo triggers false barge-in.
// We run Silero VAD (v5 ONNX via onnxruntime-web) per 512-sample frame and
// only forward frames that look like nearby, sustained speech, so:
//   - weak background voices and non-speech become silence (symptom 1), and
//   - during TTS playback we use a STRICTER threshold set so residual speaker
//     echo (which Silero scores ~0.3–0.6) is rejected, but a loud close voice
//     still interrupts (preserves real barge-in — symptom 2).
//
// IMPORTANT (architecture): we do NOT use vad-web's MicVAD. Earlier we let
// MicVAD own the capture (getUserMedia + its own worklet), and it produced
// near-silence on this device (Silero prob=0.00) even though the original
// getUserMedia + pcm16-capture-processor path delivered real signal. So this
// implementation keeps the PROVEN capture path exactly as the original code
// had it, and runs Silero inference directly on the frames that path emits.
// ─────────────────────────────────────────────────────────────────────────────

// Static assets are vendored under /vad-assets (see public/vad-assets/) and
// served by Next.js — no external CDN, no version drift.
const SILERO_MODEL_URL = "/vad-assets/silero_vad_v5.onnx";
const ORT_WASM_BASE_PATH = "/vad-assets/";

// Silero detects any speech, including background conversations. Admission
// therefore also requires near-field energy above both an absolute floor and
// an adaptive ambient-noise floor for several consecutive frames.
const VAD_POS_LISTEN = 0.65;
const VAD_CONTINUE_LISTEN = 0.50;
const VAD_CONFIRM_LISTEN = 3; // 96 ms
const VAD_REDEEM_LISTEN = 16; // 512 ms, tolerates natural pauses
const VAD_MIN_RMS_LISTEN = 0.0125; // about -38 dBFS
const VAD_SNR_LISTEN = 3.16; // +10 dB over the ambient floor

// Playback/barge-in is intentionally stricter to reject residual speaker echo.
const VAD_POS_PLAY = 0.85;
const VAD_CONTINUE_PLAY = 0.70;
const VAD_CONFIRM_PLAY = 4; // 128 ms
const VAD_REDEEM_PLAY = 10; // ~320 ms
const VAD_MIN_RMS_PLAY = 0.025; // about -32 dBFS
const VAD_SNR_PLAY = 5.62; // +15 dB over the ambient floor
const VAD_ECHO_COUPLING = 0.25;
const VAD_ECHO_MARGIN = 0.006;

const VAD_PREROLL = 5; // 160 ms; enough to preserve leading consonants
const VAD_INITIAL_NOISE_RMS = 0.004;
const VAD_NOISE_PROB_MAX = 0.15;
const VAD_NOISE_RISE_ALPHA = 0.01;
const VAD_NOISE_FALL_ALPHA = 0.05;

// Keep stricter echo rejection briefly after browser playback actually drains.
const PLAYBACK_TAIL_MS = 300;

type VadGateState = "silent" | "speaking";

/**
 * Lazily create + cache a Silero v5 ONNX session. We run inference ourselves
 * (instead of using vad-web's MicVAD) so we can feed it frames from the
 * original capture worklet. Stateful: the returned `state` tensor must be fed
 * back into the next call (Silero is an RNN).
 */
async function createSileroVad(ort: typeof import("onnxruntime-web")) {
  ort.env.wasm.wasmPaths = ORT_WASM_BASE_PATH;
  ort.env.logLevel = "error";

  const modelResp = await fetch(SILERO_MODEL_URL);
  if (!modelResp.ok) {
    throw new Error(`Failed to fetch Silero model: ${modelResp.status}`);
  }
  const modelBuffer = await modelResp.arrayBuffer();
  const session = await ort.InferenceSession.create(modelBuffer);

  const inputName = session.inputNames[0]; // "input"
  const stateName = session.inputNames[1]; // "state"
  const srName = session.inputNames[2]; // "sr"
  const outputName = session.outputNames.find((n) =>
    n.toLowerCase().includes("output")
  )!; // "output"
  const stateOutName = session.outputNames.find((n) =>
    n.toLowerCase().includes("state")
  )!; // "stateN"

  const sr = new ort.Tensor("int64", [BigInt(16000)]);
  let state = new ort.Tensor("float32", new Float32Array(2 * 1 * 128), [2, 1, 128]);

  return async function predict(audio: Float32Array): Promise<number> {
    const input = new ort.Tensor("float32", audio, [1, audio.length]);
    const out = await session.run({
      [inputName]: input,
      [stateName]: state,
      [srName]: sr,
    });
    // Feed the new state back in for the next call. The state output is
    // float32; cast past ORT's widened return type.
    state = out[stateOutName] as typeof state;
    const prob = (out[outputName].data as Float32Array)[0];
    return typeof prob === "number" ? prob : 0;
  };
}

export function useRealtimeSession(): RealtimeSession {
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [userText, setUserText] = useState("");
  const [assistantText, setAssistantText] = useState("");
  const [history, setHistory] = useState<TranscriptLine[]>([]);
  const [assistantSpeaking, setAssistantSpeaking] = useState(false);
  const [seekAttentionNonce, setSeekAttentionNonce] = useState(0);
  const [emotion, setEmotion] = useState<string>("neutral");
  const [professorSearch, setProfessorSearch] =
    useState<ProfessorSearchState | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const inputCtxRef = useRef<AudioContext | null>(null);
  const outputCtxRef = useRef<AudioContext | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | null>(null);
  const playbackNodeRef = useRef<AudioWorkletNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);
  const connectionAttemptRef = useRef(0);

  const userBufRef = useRef("");
  const assistantBufRef = useRef("");
  const micTimeRef = useRef(0);

  // ── VAD gate runtime state ──
  const gateStateRef = useRef<VadGateState>("silent");
  const onsetFramesRef = useRef(0);
  const redeemRef = useRef(0);
  const prerollRef = useRef<Int16Array[]>([]);
  const noiseRmsRef = useRef(VAD_INITIAL_NOISE_RMS);
  const assistantPlaybackRef = useRef(false);
  const assistantPlaybackEndedAtRef = useRef(Number.NEGATIVE_INFINITY);
  const suppressAssistantAudioRef = useRef(false);

  const clearConversation = useCallback(() => {
    userBufRef.current = "";
    assistantBufRef.current = "";
    setUserText("");
    setAssistantText("");
    setHistory([]);
  }, []);

  const teardown = useCallback(() => {
    connectionAttemptRef.current += 1;
    try { wsRef.current?.close(); } catch {}
    wsRef.current = null;
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;
    if (captureNodeRef.current) captureNodeRef.current.port.onmessage = null;
    try { captureNodeRef.current?.disconnect(); } catch {}
    if (playbackNodeRef.current) playbackNodeRef.current.port.onmessage = null;
    try { playbackNodeRef.current?.disconnect(); } catch {}
    try { outputAnalyserRef.current?.disconnect(); } catch {}
    captureNodeRef.current = null;
    playbackNodeRef.current = null;
    outputAnalyserRef.current = null;
    inputCtxRef.current?.close().catch(() => {});
    outputCtxRef.current?.close().catch(() => {});
    inputCtxRef.current = null;
    outputCtxRef.current = null;
    // Reset the gate so a fresh connect() starts silent.
    gateStateRef.current = "silent";
    onsetFramesRef.current = 0;
    redeemRef.current = 0;
    prerollRef.current = [];
    noiseRmsRef.current = VAD_INITIAL_NOISE_RMS;
    assistantPlaybackRef.current = false;
    assistantPlaybackEndedAtRef.current = Number.NEGATIVE_INFINITY;
    suppressAssistantAudioRef.current = false;
    micTimeRef.current = 0;
  }, []);

  const disconnect = useCallback(() => {
    teardown();
    clearConversation();
    setStatus("idle");
    setErrorMessage(null);
    setAssistantSpeaking(false);
    setEmotion("neutral");
    setProfessorSearch(null);
  }, [clearConversation, teardown]);

  const handleControl = useCallback((msg: Record<string, unknown>) => {
    switch (msg.type) {
      case "ready":
        setStatus("active");
        break;
      case "transcript": {
        const role = msg.role as "user" | "assistant";
        const text = String(msg.text ?? "");
        if (role === "user") {
          userBufRef.current += text;
          setUserText(userBufRef.current);
        } else {
          assistantBufRef.current += text;
          setAssistantText(assistantBufRef.current);
        }
        break;
      }
      case "interrupt":
        // Barge-in: flush the playback buffer and drop the partial reply.
        playbackNodeRef.current?.port.postMessage({ type: "reset" });
        assistantBufRef.current = "";
        assistantPlaybackRef.current = false;
        assistantPlaybackEndedAtRef.current = performance.now();
        setAssistantSpeaking(false);
        setEmotion("neutral");
        break;
      case "assistant_audio_start":
        // Ordered immediately before the first PCM frame of a new model turn.
        suppressAssistantAudioRef.current = false;
        break;
      case "seekAttention":
        setSeekAttentionNonce((n) => n + 1);
        break;
      case "emotion":
        // Avatar expression: a go_emotions label pushed by the orchestrator
        // (only when ENABLE_EMOTION=true server-side).
        setEmotion(String(msg.emotion ?? "neutral"));
        break;
      case "tool_activity": {
        const id = String(msg.id ?? "");
        const status = String(msg.status ?? "error") as ProfessorSearchState["status"];
        setProfessorSearch((current) => ({
          id,
          query: String(msg.query ?? current?.query ?? ""),
          status,
          message: msg.message ? String(msg.message) : undefined,
          results: current?.id === id ? current.results : [],
          sourceName: current?.id === id ? current.sourceName : undefined,
          sourceUrl: current?.id === id ? current.sourceUrl : undefined,
        }));
        break;
      }
      case "tool_result": {
        const results = Array.isArray(msg.results)
          ? (msg.results as ProfessorResult[])
          : [];
        setProfessorSearch({
          id: String(msg.id ?? ""),
          query: String(msg.query ?? ""),
          status: "completed",
          results,
          sourceName: String(msg.source_name ?? "İTÜ Akademi"),
          sourceUrl: String(msg.source_url ?? ""),
        });
        break;
      }
      case "turn_complete": {
        // Gemini has finished producing this turn, but the AudioWorklet may
        // still have buffered PCM. It will emit playback-drained when audible
        // output has actually finished.
        playbackNodeRef.current?.port.postMessage({ type: "end-of-turn" });
        const u = userBufRef.current.trim();
        const a = assistantBufRef.current.trim();
        setHistory((h) => [
          ...h,
          ...(u ? [{ role: "user" as const, text: u }] : []),
          ...(a ? [{ role: "assistant" as const, text: a }] : []),
        ]);
        userBufRef.current = "";
        assistantBufRef.current = "";
        setUserText("");
        setAssistantText("");
        break;
      }
      case "error":
        playbackNodeRef.current?.port.postMessage({ type: "reset" });
        assistantPlaybackRef.current = false;
        assistantPlaybackEndedAtRef.current = performance.now();
        setAssistantSpeaking(false);
        setErrorMessage(String(msg.message ?? "error"));
        setStatus("error");
        break;
    }
  }, []);

  const connect = useCallback(
    async (sessionId: string) => {
      if (status === "connecting" || status === "active") return;
      clearConversation();
      const attempt = connectionAttemptRef.current + 1;
      connectionAttemptRef.current = attempt;
      const isCurrentAttempt = () => connectionAttemptRef.current === attempt;
      setStatus("connecting");
      setErrorMessage(null);
      setProfessorSearch(null);
      try {
        // ── mic capture -> PCM16 @16k (UNCHANGED from the working original) ──
        // This getUserMedia + pcm16-capture-processor path is proven to deliver
        // signal on this device. The only change vs the original: we size the
        // worklet frames to 512 samples (Silero v5's fixed window) so we can
        // run inference per frame.
        const micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            // AGC can lift distant conversations toward foreground level.
            // A fixed kiosk microphone works better with the explicit RMS gate.
            autoGainControl: false,
          },
          video: false,
        });
        if (!isCurrentAttempt()) {
          micStream.getTracks().forEach((track) => track.stop());
          return;
        }
        micStreamRef.current = micStream;

        const inputCtx = new AudioContext();
        inputCtxRef.current = inputCtx;
        try { await inputCtx.resume(); } catch {}
        await inputCtx.audioWorklet.addModule("/pcm16-capture-processor.js");
        if (!isCurrentAttempt()) {
          micStream.getTracks().forEach((track) => track.stop());
          await inputCtx.close().catch(() => {});
          return;
        }
        const source = inputCtx.createMediaStreamSource(micStream);
        const captureNode = new AudioWorkletNode(inputCtx, "pcm16-capture-processor", {
          processorOptions: {
            targetRate: INPUT_TARGET_RATE,
            frameSamples: VAD_FRAME_SAMPLES,
          },
        });
        captureNodeRef.current = captureNode;
        // Drain and discard capture messages while the recurrent VAD model is
        // loading; MessagePort would otherwise replay stale startup audio.
        captureNode.port.onmessage = () => {};
        source.connect(captureNode);
        captureNode.connect(inputCtx.destination); // keep the node pulled (silent output)

        // ── playback @24k + analyser for lip-sync ───────────────────────────
        // (unchanged — the assistant's audio output path is orthogonal to the
        // mic gate.)
        const outputCtx = new AudioContext({ sampleRate: OUTPUT_RATE });
        outputCtxRef.current = outputCtx;
        try { await outputCtx.resume(); } catch {}
        await outputCtx.audioWorklet.addModule("/audio-output-processor.js");
        if (!isCurrentAttempt()) {
          micStream.getTracks().forEach((track) => track.stop());
          await inputCtx.close().catch(() => {});
          await outputCtx.close().catch(() => {});
          return;
        }
        const playbackNode = new AudioWorkletNode(outputCtx, "audio-output-processor");
        playbackNodeRef.current = playbackNode;
        const analyser = outputCtx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.3;
        outputAnalyserRef.current = analyser;
        playbackNode.connect(outputCtx.destination);
        playbackNode.connect(analyser);
        playbackNode.port.onmessage = (e: MessageEvent) => {
          if (playbackNodeRef.current !== playbackNode) return;
          const type = (e.data as { type?: string } | null)?.type;
          if (type === "playback-start") {
            assistantPlaybackRef.current = true;
            // Do not let a listening-mode segment remain open once speaker
            // output begins; it would bypass the stricter echo onset gate.
            gateStateRef.current = "silent";
            onsetFramesRef.current = 0;
            redeemRef.current = 0;
            prerollRef.current = [];
            setAssistantSpeaking(true);
          } else if (type === "playback-drained" || type === "playback-reset") {
            assistantPlaybackRef.current = false;
            assistantPlaybackEndedAtRef.current = performance.now();
            setAssistantSpeaking(false);
            setEmotion("neutral");
          }
        };

        // Load the recurrent VAD before opening the realtime socket. This keeps
        // the UI in "connecting" while mic frames are intentionally discarded,
        // instead of accepting user speech during a capture blackout.
        const ort = await import("onnxruntime-web");
        console.log("[vad] loading Silero v5 ONNX session...");
        const predict = await createSileroVad(ort);
        if (!isCurrentAttempt()) {
          micStream.getTracks().forEach((track) => track.stop());
          try { captureNode.disconnect(); } catch {}
          try { playbackNode.disconnect(); } catch {}
          await inputCtx.close().catch(() => {});
          await outputCtx.close().catch(() => {});
          return;
        }
        console.log("[vad] session ready; inputCtx.state=" + inputCtx.state);

        // ── WebSocket to orchestrator ───────────────────────────────────────
        // No awaits occur between construction and handler registration, so the
        // orchestrator's immediate {"type":"ready"} handshake cannot be lost.
        const ws = new WebSocket(getOrchestratorWsUrl(sessionId));
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onmessage = (e: MessageEvent) => {
          if (wsRef.current !== ws) return;
          if (typeof e.data === "string") {
            try {
              handleControl(JSON.parse(e.data));
            } catch {
              /* ignore malformed control */
            }
            return;
          }
          // A locally confirmed barge-in has already flushed playback. Drop
          // old-turn PCM until the orchestrator marks the next response.
          if (suppressAssistantAudioRef.current) {
            return;
          }
          // Binary: 24 kHz PCM16 -> Float32 frame for the playback worklet.
          const pcm = new Int16Array(e.data as ArrayBuffer);
          const frame = new Float32Array(pcm.length);
          for (let i = 0; i < pcm.length; i++) frame[i] = pcm[i] / 32768;
          micTimeRef.current += frame.length / OUTPUT_RATE;
          playbackNode.port.postMessage({
            frame,
            micDuration: micTimeRef.current,
          });
        };

        ws.onerror = () => {
          if (wsRef.current !== ws) return;
          setErrorMessage("bağlantı hatası");
          setStatus("error");
          try { ws.close(); } catch {}
        };
        ws.onclose = () => {
          if (wsRef.current !== ws) return;
          playbackNode.port.postMessage({ type: "reset" });
          wsRef.current = null;
          teardown();
          clearConversation();
          setStatus((current) => current === "error" ? current : "idle");
          setAssistantSpeaking(false);
        };

        // PCM16 -> Float32 for Silero, retaining RMS for the near-field gate.
        const decodeFrame = (pcm: Int16Array) => {
          const audio = new Float32Array(pcm.length);
          let sum = 0;
          for (let i = 0; i < pcm.length; i++) {
            const sample = pcm[i] / 32768;
            audio[i] = sample;
            sum += sample * sample;
          }
          return { audio, rms: Math.sqrt(sum / Math.max(1, pcm.length)) };
        };

        const farBuffer = new Float32Array(analyser.fftSize);
        const readFarRms = () => {
          if (!assistantPlaybackRef.current) return 0;
          analyser.getFloatTimeDomainData(farBuffer);
          let sum = 0;
          for (let i = 0; i < farBuffer.length; i++) {
            sum += farBuffer[i] * farBuffer[i];
          }
          return Math.sqrt(sum / farBuffer.length);
        };

        // Diagnostics: one log every ~1 s so the kiosk operator can confirm the
        // gate is live. Watch these in the browser console (F12).
        let sentFrames = 0;
        let sentSpeechFrames = 0;
        let lastLog = 0;

        const sendFrame = (pcm: Int16Array) => {
          if (wsRef.current === ws && ws.readyState === WebSocket.OPEN) {
            ws.send(pcm.buffer);
            sentFrames++;
            return true;
          }
          return false;
        };

        // CRITICAL: Gemini Live's server-side VAD needs a CONTINUOUS audio
        // stream to detect speech boundaries and endpoint turns. If we drop
        // packets during silence, Gemini never sees the trailing silence it
        // needs to commit a turn and the model never responds. So we ALWAYS
        // send a frame — real speech during a speech segment, and this
        // zero-filled buffer during silence. The gate still does its job:
        // background chatter becomes zeros (inaudible to Gemini's VAD) instead
        // of being forwarded as if it were the user talking.
        const SILENT_FRAME = new Int16Array(VAD_FRAME_SAMPLES); // all zeros

        const processCapturedFrame = async (data: ArrayBuffer) => {
          if (wsRef.current !== ws) return;
          const pcm = new Int16Array(data);
          const { audio, rms } = decodeFrame(pcm);

          // Run Silero inference. Stateful (RNN); `predict` keeps the state.
          let prob = 0;
          try {
            prob = await predict(audio);
          } catch (err) {
            console.error("[vad] inference failed", err);
            // On inference failure, send silence so the stream stays continuous.
            sendFrame(SILENT_FRAME);
            return;
          }
          if (wsRef.current !== ws) return;

          // Use actual browser playout, not WebSocket packet arrival. Packets
          // can be queued hundreds of milliseconds before they are audible.
          const now = performance.now();
          const playing =
            assistantPlaybackRef.current ||
            now - assistantPlaybackEndedAtRef.current < PLAYBACK_TAIL_MS;
          const posThr = playing ? VAD_POS_PLAY : VAD_POS_LISTEN;
          const continueThr = playing
            ? VAD_CONTINUE_PLAY
            : VAD_CONTINUE_LISTEN;
          const confirmFrames = playing
            ? VAD_CONFIRM_PLAY
            : VAD_CONFIRM_LISTEN;
          const redeemCap = playing ? VAD_REDEEM_PLAY : VAD_REDEEM_LISTEN;
          const baseRms = playing ? VAD_MIN_RMS_PLAY : VAD_MIN_RMS_LISTEN;
          const snr = playing ? VAD_SNR_PLAY : VAD_SNR_LISTEN;

          // Learn stationary room noise only from confident non-speech while
          // the speaker is quiet. Speech must never raise the ambient floor.
          if (
            gateStateRef.current === "silent" &&
            !playing &&
            prob <= VAD_NOISE_PROB_MAX
          ) {
            const floor = noiseRmsRef.current;
            const alpha =
              rms > floor ? VAD_NOISE_RISE_ALPHA : VAD_NOISE_FALL_ALPHA;
            noiseRmsRef.current = Math.max(
              0.0005,
              Math.min(0.02, floor + (rms - floor) * alpha),
            );
          }

          const farRms = readFarRms();
          const echoFloor = playing
            ? Math.min(
                0.08,
                farRms * VAD_ECHO_COUPLING + VAD_ECHO_MARGIN,
              )
            : 0;
          const minRms = Math.max(
            baseRms,
            noiseRmsRef.current * snr,
            echoFloor,
          );
          const onsetQualified = prob >= posThr && rms >= minRms;

          let sentCurrentThroughPreroll = false;
          let forwardThisFrame = false;

          if (gateStateRef.current === "silent") {
            if (playing) {
              // Never retain speaker echo as barge-in pre-roll. Keep only the
              // consecutive frames that satisfy the strict near-field gate.
              if (onsetQualified) {
                if (onsetFramesRef.current === 0) prerollRef.current = [];
                prerollRef.current.push(pcm);
              } else {
                prerollRef.current = [];
              }
            } else {
              prerollRef.current.push(pcm);
              if (prerollRef.current.length > VAD_PREROLL) {
                prerollRef.current.shift();
              }
            }

            onsetFramesRef.current = onsetQualified
              ? onsetFramesRef.current + 1
              : 0;

            if (onsetFramesRef.current >= confirmFrames) {
              gateStateRef.current = "speaking";
              onsetFramesRef.current = 0;
              redeemRef.current = 0;
              const ring = playing
                ? prerollRef.current.slice(-confirmFrames)
                : prerollRef.current;
              prerollRef.current = [];
              for (const f of ring) {
                if (sendFrame(f)) {
                  sentSpeechFrames++;
                  if (f === pcm) sentCurrentThroughPreroll = true;
                }
              }

              // Confirmed close speech is a local barge-in. Stop old playback
              // immediately rather than waiting for Gemini's round trip.
              if (assistantPlaybackRef.current) {
                playbackNode.port.postMessage({ type: "reset" });
                assistantPlaybackRef.current = false;
                assistantPlaybackEndedAtRef.current = now;
                suppressAssistantAudioRef.current = true;
                setAssistantSpeaking(false);
              }
            }
          } else {
            forwardThisFrame = true;
            const speechContinues =
              prob >= continueThr && rms >= minRms * 0.6;
            if (!speechContinues) {
              redeemRef.current += 1;
              if (redeemRef.current >= redeemCap) {
                gateStateRef.current = "silent";
                onsetFramesRef.current = 0;
                redeemRef.current = 0;
                prerollRef.current = [];
                forwardThisFrame = false;
              }
            } else {
              redeemRef.current = 0;
            }
          }

          // ALWAYS send a frame — real speech audio when the gate is open, a
          // zero-filled silent frame when it's closed. This keeps the uplink
          // stream continuous so Gemini's server-side VAD can endpoint turns
          // (it needs to *see* trailing silence). Dropping packets stalls it.
          if (sentCurrentThroughPreroll) {
            // Already sent once as the final pre-roll frame.
          } else if (forwardThisFrame) {
            if (sendFrame(pcm)) sentSpeechFrames++;
          } else {
            sendFrame(SILENT_FRAME);
          }

          // Diagnostics (~1/s, avoid console spam). `sent` = total frames sent
          // (must keep climbing); `speech` = how many carried real audio.
          if (now - lastLog > 1000) {
            console.log(
              `[vad] frame state=${gateStateRef.current} prob=${prob.toFixed(2)} ` +
                `rms=${rms.toFixed(4)} min=${minRms.toFixed(4)} ` +
                `noise=${noiseRmsRef.current.toFixed(4)} candidate=${onsetFramesRef.current} ` +
                `playing=${playing} sent=${sentFrames} speech=${sentSpeechFrames}`
            );
            lastLog = now;
          }
        };

        // Silero is recurrent: process frames strictly in capture order so two
        // async inferences can never read/update the same RNN state concurrently.
        let vadFrameChain = Promise.resolve();
        captureNode.port.onmessage = (e: MessageEvent) => {
          const data = e.data as ArrayBuffer;
          vadFrameChain = vadFrameChain
            .then(() => processCapturedFrame(data))
            .catch((err) => {
              console.error("[vad] frame processing failed", err);
              sendFrame(SILENT_FRAME);
            });
        };
      } catch (err) {
        if (!isCurrentAttempt()) return;
        setErrorMessage(err instanceof Error ? err.message : String(err));
        setStatus("error");
        teardown();
      }
    },
    [status, clearConversation, handleControl, teardown]
  );

  useEffect(() => () => teardown(), [teardown]);

  return {
    status,
    errorMessage,
    userText,
    assistantText,
    history,
    assistantSpeaking,
    seekAttentionNonce,
    emotion,
    professorSearch,
    outputAnalyserRef,
    connect,
    disconnect,
  };
}
