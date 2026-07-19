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
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const stop = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    try { wsRef.current?.close(); } catch {}
    wsRef.current = null;
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setStatus("idle");
  }, []);

  const start = useCallback(async (sessionId: string) => {
    setStatus("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play().catch(() => {});
      }

      const canvas = canvasRef.current || document.createElement("canvas");
      canvasRef.current = canvas;
      const ctx = canvas.getContext("2d");

      const ws = new WebSocket(getCvWsUrl("stream", sessionId));
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("streaming");
        timerRef.current = window.setInterval(() => {
          const v = videoRef.current;
          if (!v || !ctx || v.videoWidth === 0 || ws.readyState !== WebSocket.OPEN) return;
          canvas.width = v.videoWidth;
          canvas.height = v.videoHeight;
          ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
          canvas.toBlob(
            (blob) => {
              if (!blob || ws.readyState !== WebSocket.OPEN) return;
              blob.arrayBuffer().then((buf) => {
                if (ws.readyState === WebSocket.OPEN) ws.send(buf);
              });
            },
            "image/jpeg",
            JPEG_QUALITY
          );
        }, FRAME_INTERVAL_MS);
      };
      ws.onerror = () => setStatus("error");
      ws.onclose = () => {
        if (timerRef.current !== null) {
          window.clearInterval(timerRef.current);
          timerRef.current = null;
        }
      };
    } catch (err) {
      console.error("[webcam] start failed", err);
      setStatus("error");
    }
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { status, videoRef, start, stop };
}
