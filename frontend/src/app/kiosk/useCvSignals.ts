"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getCvWsUrl } from "./urls";

export interface CvProfile {
  scores?: { attention?: number; openness?: number; energy?: number };
  signals?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface FacePosition {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type CvSessionState = "IDLE" | "CALIBRATING" | "ACTIVE" | "unknown";

export interface CvSignals {
  isFocused: boolean;
  focusTime: number;
  profile: CvProfile | null;
  presenceState: "present" | "absent" | "unknown";
  sessionState: CvSessionState;
  facePosition: FacePosition | null;
  start: (sessionId: string) => void;
  stop: () => void;
}

export function useCvSignals(): CvSignals {
  const [isFocused, setIsFocused] = useState(true);
  const [focusTime, setFocusTime] = useState(0);
  const [profile, setProfile] = useState<CvProfile | null>(null);
  const [presenceState, setPresenceState] =
    useState<CvSignals["presenceState"]>("unknown");
  const [sessionState, setSessionState] = useState<CvSessionState>("unknown");
  const [facePosition, setFacePosition] = useState<FacePosition | null>(null);

  const focusWsRef = useRef<WebSocket | null>(null);
  const profileWsRef = useRef<WebSocket | null>(null);
  const trackingWsRef = useRef<WebSocket | null>(null);
  const generationRef = useRef(0);
  const reconnectTimersRef = useRef<Set<number>>(new Set());

  const stop = useCallback(() => {
    generationRef.current += 1;
    reconnectTimersRef.current.forEach((timer) => window.clearTimeout(timer));
    reconnectTimersRef.current.clear();
    try { focusWsRef.current?.close(); } catch {}
    try { profileWsRef.current?.close(); } catch {}
    try { trackingWsRef.current?.close(); } catch {}
    focusWsRef.current = null;
    profileWsRef.current = null;
    trackingWsRef.current = null;
    setIsFocused(true);
    setFocusTime(0);
    setProfile(null);
    setPresenceState("unknown");
    setSessionState("unknown");
    setFacePosition(null);
  }, []);

  const start = useCallback((sessionId: string) => {
    stop();
    const generation = generationRef.current;
    const current = () => generationRef.current === generation;
    const scheduleReconnect = (connect: () => void, delay: number) => {
      if (!current()) return;
      const timer = window.setTimeout(() => {
        reconnectTimersRef.current.delete(timer);
        if (current()) connect();
      }, delay);
      reconnectTimersRef.current.add(timer);
    };

    const connectFocus = () => {
      if (!current()) return;
      const ws = new WebSocket(getCvWsUrl("focus", sessionId));
      focusWsRef.current = ws;
      ws.onmessage = (e) => {
        if (!current() || focusWsRef.current !== ws) return;
        try {
          const payload = JSON.parse(e.data);
          setIsFocused(Boolean(payload.is_focused));
          setFocusTime(Number(payload.focus_time) || 0);
        } catch {}
      };
      ws.onclose = () => {
        if (!current() || focusWsRef.current !== ws) return;
        focusWsRef.current = null;
        scheduleReconnect(connectFocus, 1000);
      };
      ws.onerror = () => {
        if (!current() || focusWsRef.current !== ws) return;
        try { ws.close(); } catch {}
      };
    };

    const connectProfile = () => {
      if (!current()) return;
      const ws = new WebSocket(getCvWsUrl("profile", sessionId));
      profileWsRef.current = ws;
      ws.onmessage = (e) => {
        if (!current() || profileWsRef.current !== ws) return;
        try { setProfile(JSON.parse(e.data)); } catch {}
      };
      ws.onclose = () => {
        if (!current() || profileWsRef.current !== ws) return;
        profileWsRef.current = null;
        scheduleReconnect(connectProfile, 2000);
      };
      ws.onerror = () => {
        if (!current() || profileWsRef.current !== ws) return;
        try { ws.close(); } catch {}
      };
    };

    const connectTracking = () => {
      if (!current()) return;
      const ws = new WebSocket(getCvWsUrl("tracking", sessionId));
      trackingWsRef.current = ws;
      ws.onmessage = (e) => {
        if (!current() || trackingWsRef.current !== ws) return;
        try {
          const payload = JSON.parse(e.data);
          const state = payload.state;
          setSessionState(
            state === "IDLE" || state === "CALIBRATING" || state === "ACTIVE"
              ? state
              : "unknown"
          );
          const presence = payload.presence_state;
          if (presence !== "present" && presence !== "absent") {
            setPresenceState("unknown");
            setFacePosition(null);
            return;
          }
          setPresenceState(presence);
          if (presence !== "present") {
            setFacePosition(null);
            return;
          }
          const x = payload.face_center_x;
          const y = payload.face_center_y;
          const width = payload.face_bbox_width;
          const height = payload.face_bbox_height;
          if (
            typeof x === "number" && Number.isFinite(x) && x >= 0 && x <= 1 &&
            typeof y === "number" && Number.isFinite(y) && y >= 0 && y <= 1 &&
            typeof width === "number" && Number.isFinite(width) && width > 0 && width <= 1 &&
            typeof height === "number" && Number.isFinite(height) && height > 0 && height <= 1
          ) {
            setFacePosition({ x, y, width, height });
          } else {
            setFacePosition(null);
          }
        } catch {
          if (!current() || trackingWsRef.current !== ws) return;
          setPresenceState("unknown");
          setSessionState("unknown");
          setFacePosition(null);
        }
      };
      ws.onclose = () => {
        if (!current() || trackingWsRef.current !== ws) return;
        trackingWsRef.current = null;
        setPresenceState("unknown");
        setSessionState("unknown");
        setFacePosition(null);
        scheduleReconnect(connectTracking, 1000);
      };
      ws.onerror = () => {
        if (!current() || trackingWsRef.current !== ws) return;
        setPresenceState("unknown");
        setSessionState("unknown");
        setFacePosition(null);
        try { ws.close(); } catch {}
      };
    };

    connectFocus();
    connectProfile();
    connectTracking();
  }, [stop]);

  useEffect(() => () => stop(), [stop]);

  return {
    isFocused,
    focusTime,
    profile,
    presenceState,
    sessionState,
    facePosition,
    start,
    stop,
  };
}
