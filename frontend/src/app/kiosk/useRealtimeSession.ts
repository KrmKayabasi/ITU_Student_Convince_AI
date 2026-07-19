"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getOrchestratorWsUrl } from "./urls";

export type SessionStatus = "idle" | "connecting" | "active" | "error";

export interface TranscriptLine {
  role: "user" | "assistant";
  text: string;
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
  /** AnalyserNode on the assistant audio output — drives lip-sync. */
  outputAnalyserRef: React.MutableRefObject<AnalyserNode | null>;
  connect: (sessionId: string) => Promise<void>;
  disconnect: () => void;
}

const INPUT_TARGET_RATE = 16000;
const OUTPUT_RATE = 24000;

export function useRealtimeSession(): RealtimeSession {
  const [status, setStatus] = useState<SessionStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [userText, setUserText] = useState("");
  const [assistantText, setAssistantText] = useState("");
  const [history, setHistory] = useState<TranscriptLine[]>([]);
  const [assistantSpeaking, setAssistantSpeaking] = useState(false);
  const [seekAttentionNonce, setSeekAttentionNonce] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const micStreamRef = useRef<MediaStream | null>(null);
  const inputCtxRef = useRef<AudioContext | null>(null);
  const outputCtxRef = useRef<AudioContext | null>(null);
  const captureNodeRef = useRef<AudioWorkletNode | null>(null);
  const playbackNodeRef = useRef<AudioWorkletNode | null>(null);
  const outputAnalyserRef = useRef<AnalyserNode | null>(null);

  const userBufRef = useRef("");
  const assistantBufRef = useRef("");
  const micTimeRef = useRef(0);

  const teardown = useCallback(() => {
    try { wsRef.current?.close(); } catch {}
    wsRef.current = null;
    micStreamRef.current?.getTracks().forEach((t) => t.stop());
    micStreamRef.current = null;
    try { captureNodeRef.current?.disconnect(); } catch {}
    try { playbackNodeRef.current?.disconnect(); } catch {}
    try { outputAnalyserRef.current?.disconnect(); } catch {}
    captureNodeRef.current = null;
    playbackNodeRef.current = null;
    outputAnalyserRef.current = null;
    inputCtxRef.current?.close().catch(() => {});
    outputCtxRef.current?.close().catch(() => {});
    inputCtxRef.current = null;
    outputCtxRef.current = null;
  }, []);

  const disconnect = useCallback(() => {
    teardown();
    setStatus("idle");
    setAssistantSpeaking(false);
  }, [teardown]);

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
          setAssistantSpeaking(true);
        }
        break;
      }
      case "interrupt":
        // Barge-in: flush the playback buffer and drop the partial reply.
        playbackNodeRef.current?.port.postMessage({ type: "reset" });
        assistantBufRef.current = "";
        setAssistantSpeaking(false);
        break;
      case "seekAttention":
        setSeekAttentionNonce((n) => n + 1);
        break;
      case "turn_complete": {
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
        setAssistantSpeaking(false);
        break;
      }
      case "error":
        setErrorMessage(String(msg.message ?? "error"));
        setStatus("error");
        break;
    }
  }, []);

  const connect = useCallback(
    async (sessionId: string) => {
      if (status === "connecting" || status === "active") return;
      setStatus("connecting");
      setErrorMessage(null);
      try {
        // ── mic capture -> PCM16 @16k ───────────────────────────────────────
        const micStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
          video: false,
        });
        micStreamRef.current = micStream;

        const inputCtx = new AudioContext();
        inputCtxRef.current = inputCtx;
        await inputCtx.audioWorklet.addModule("/pcm16-capture-processor.js");
        const source = inputCtx.createMediaStreamSource(micStream);
        const captureNode = new AudioWorkletNode(inputCtx, "pcm16-capture-processor", {
          processorOptions: { targetRate: INPUT_TARGET_RATE },
        });
        captureNodeRef.current = captureNode;
        source.connect(captureNode);
        captureNode.connect(inputCtx.destination); // keep the node pulled (silent output)

        // ── playback @24k + analyser for lip-sync ───────────────────────────
        const outputCtx = new AudioContext({ sampleRate: OUTPUT_RATE });
        outputCtxRef.current = outputCtx;
        await outputCtx.audioWorklet.addModule("/audio-output-processor.js");
        const playbackNode = new AudioWorkletNode(outputCtx, "audio-output-processor");
        playbackNodeRef.current = playbackNode;
        const analyser = outputCtx.createAnalyser();
        analyser.fftSize = 1024;
        analyser.smoothingTimeConstant = 0.3;
        outputAnalyserRef.current = analyser;
        playbackNode.connect(outputCtx.destination);
        playbackNode.connect(analyser);

        // ── WebSocket to orchestrator ───────────────────────────────────────
        const ws = new WebSocket(getOrchestratorWsUrl(sessionId));
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        captureNode.port.onmessage = (e: MessageEvent) => {
          // e.data is an ArrayBuffer of Int16 PCM samples.
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(e.data as ArrayBuffer);
          }
        };

        ws.onmessage = (e: MessageEvent) => {
          if (typeof e.data === "string") {
            try {
              handleControl(JSON.parse(e.data));
            } catch {
              /* ignore malformed control */
            }
            return;
          }
          // Binary: 24 kHz PCM16 -> Float32 frame for the playback worklet.
          const pcm = new Int16Array(e.data as ArrayBuffer);
          const frame = new Float32Array(pcm.length);
          for (let i = 0; i < pcm.length; i++) frame[i] = pcm[i] / 32768;
          micTimeRef.current += frame.length / OUTPUT_RATE;
          playbackNodeRef.current?.port.postMessage({
            frame,
            micDuration: micTimeRef.current,
          });
        };

        ws.onerror = () => {
          setErrorMessage("bağlantı hatası");
          setStatus("error");
        };
        ws.onclose = () => {
          if (status !== "error") setStatus("idle");
          setAssistantSpeaking(false);
        };
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : String(err));
        setStatus("error");
        teardown();
      }
    },
    [status, handleControl, teardown]
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
    outputAnalyserRef,
    connect,
    disconnect,
  };
}
