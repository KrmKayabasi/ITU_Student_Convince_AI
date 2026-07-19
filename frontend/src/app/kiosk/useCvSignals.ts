"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getCvWsUrl } from "./urls";

export interface CvProfile {
  scores?: { attention?: number; openness?: number; energy?: number };
  signals?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface CvSignals {
  isFocused: boolean;
  focusTime: number;
  profile: CvProfile | null;
  start: (sessionId: string) => void;
  stop: () => void;
}

/**
 * Browser-side subscription to the CV pipeline's /focus and /profile channels
 * for the same session_id the webcam streams to. Used for low-latency avatar
 * reactions (the orchestrator independently subscribes server-side for LLM
 * injection). Read-only: the CV server pushes; we only consume.
 */
export function useCvSignals(): CvSignals {
  const [isFocused, setIsFocused] = useState(true);
  const [focusTime, setFocusTime] = useState(0);
  const [profile, setProfile] = useState<CvProfile | null>(null);

  const focusWsRef = useRef<WebSocket | null>(null);
  const profileWsRef = useRef<WebSocket | null>(null);
  const closingRef = useRef(false);

  const stop = useCallback(() => {
    closingRef.current = true;
    try { focusWsRef.current?.close(); } catch {}
    try { profileWsRef.current?.close(); } catch {}
    focusWsRef.current = null;
    profileWsRef.current = null;
  }, []);

  const start = useCallback((sessionId: string) => {
    closingRef.current = false;

    const connectFocus = () => {
      const ws = new WebSocket(getCvWsUrl("focus", sessionId));
      focusWsRef.current = ws;
      ws.onmessage = (e) => {
        try {
          const p = JSON.parse(e.data);
          setIsFocused(Boolean(p.is_focused));
          setFocusTime(Number(p.focus_time) || 0);
        } catch {}
      };
      ws.onclose = () => {
        if (!closingRef.current) window.setTimeout(connectFocus, 1000);
      };
      ws.onerror = () => { try { ws.close(); } catch {} };
    };

    const connectProfile = () => {
      const ws = new WebSocket(getCvWsUrl("profile", sessionId));
      profileWsRef.current = ws;
      ws.onmessage = (e) => {
        try {
          setProfile(JSON.parse(e.data));
        } catch {}
      };
      ws.onclose = () => {
        if (!closingRef.current) window.setTimeout(connectProfile, 2000);
      };
      ws.onerror = () => { try { ws.close(); } catch {} };
    };

    connectFocus();
    connectProfile();
  }, []);

  useEffect(() => () => stop(), [stop]);

  return { isFocused, focusTime, profile, start, stop };
}
