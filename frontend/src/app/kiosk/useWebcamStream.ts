"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getCvWsUrl } from "./urls";

export type WebcamStatus = "idle" | "starting" | "streaming" | "error";

const FRAME_INTERVAL_MS = 100; // ~10 fps to the CV pipeline
const JPEG_QUALITY = 0.6;

/**
 * Captures the webcam, shows it in `videoRef`, and streams JPEG frames to the
 * CV pipeline's /stream/{sessionId} WebSocket (mirrors client/workers.py's
 * cv2.imencode(".jpg") -> ws.send pattern, done in-browser via canvas).
 */
export function useWebcamStream() {
  const [status, setStatus] = useState<WebcamStatus>("idle");
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<number | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const generationRef = useRef(0);

  const stop = useCallback(() => {
    generationRef.current += 1;
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    try { wsRef.current?.close(); } catch {}
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("idle");
  }, []);

  const start = useCallback(async (sessionId: string) => {
    const generation = generationRef.current + 1;
    generationRef.current = generation;
    const current = () => generationRef.current === generation;

    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    if (reconnectTimerRef.current !== null) {
      window.clearTimeout(reconnectTimerRef.current);
    }
    timerRef.current = null;
    reconnectTimerRef.current = null;
    try { wsRef.current?.close(); } catch {}
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setStatus("starting");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (!current()) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play().catch(() => {});
      }
      if (!current()) {
        stream.getTracks().forEach((track) => track.stop());
        if (video?.srcObject === stream) video.srcObject = null;
        return;
      }

      const canvas = canvasRef.current || document.createElement("canvas");
      canvasRef.current = canvas;
      const ctx = canvas.getContext("2d");

      const scheduleReconnect = (connect: () => void) => {
        if (!current() || reconnectTimerRef.current !== null) return;
        reconnectTimerRef.current = window.setTimeout(() => {
          reconnectTimerRef.current = null;
          connect();
        }, 1000);
      };

      const connectUpload = () => {
        if (!current() || streamRef.current !== stream) return;
        let ws: WebSocket;
        try {
          ws = new WebSocket(getCvWsUrl("stream", sessionId));
        } catch {
          setStatus("error");
          scheduleReconnect(connectUpload);
          return;
        }
        ws.binaryType = "arraybuffer";
        wsRef.current = ws;

        ws.onopen = () => {
          if (!current() || wsRef.current !== ws) {
            try { ws.close(); } catch {}
            return;
          }
          setStatus("streaming");
          if (timerRef.current !== null) window.clearInterval(timerRef.current);
          timerRef.current = window.setInterval(() => {
            const v = videoRef.current;
            if (
              !current() ||
              wsRef.current !== ws ||
              !v ||
              !ctx ||
              v.videoWidth === 0 ||
              ws.readyState !== WebSocket.OPEN
            ) return;
            canvas.width = v.videoWidth;
            canvas.height = v.videoHeight;
            ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
            canvas.toBlob(
              (blob) => {
                if (!blob || !current() || wsRef.current !== ws) return;
                blob.arrayBuffer().then((buf) => {
                  if (
                    current() &&
                    wsRef.current === ws &&
                    ws.readyState === WebSocket.OPEN
                  ) ws.send(buf);
                }).catch(() => {});
              },
              "image/jpeg",
              JPEG_QUALITY
            );
          }, FRAME_INTERVAL_MS);
        };
        ws.onerror = () => {
          if (!current() || wsRef.current !== ws) return;
          setStatus("error");
          try { ws.close(); } catch {}
        };
        ws.onclose = () => {
          if (!current() || wsRef.current !== ws) return;
          wsRef.current = null;
          if (timerRef.current !== null) {
            window.clearInterval(timerRef.current);
            timerRef.current = null;
          }
          setStatus("starting");
          scheduleReconnect(connectUpload);
        };
      };

      connectUpload();
    } catch (err) {
      if (!current()) return;
      console.error("[webcam] start failed", err);
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current) videoRef.current.srcObject = null;
      setStatus("error");
    }
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { status, videoRef, start, stop };
}
