"use client";

import { useEffect, useRef, useState } from "react";

/** The face's expression states. `seekAttention` is NOT a state — it's a timed
 *  overlay gesture triggered by a nonce, layered on whatever state is active. */
export type FaceState =
  | "attract"
  | "connecting"
  | "listening"
  | "speaking"
  | "thinking"
  | "concerned";

export interface DeriveArgs {
  started: boolean;
  status: "idle" | "connecting" | "active" | "error";
  assistantSpeaking: boolean;
  isFocused: boolean;
  thinking: boolean;
}

export function deriveFaceState(a: DeriveArgs): FaceState {
  if (!a.started) return "attract";
  if (a.status === "connecting") return "connecting";
  if (a.status === "error") return "concerned";
  if (a.status !== "active") return "attract";
  if (a.assistantSpeaking) return "speaking";
  if (!a.isFocused) return "concerned";
  if (a.thinking) return "thinking";
  return "listening";
}

/**
 * "Thinking" hint: true for up to `holdMs` after the user's streaming text
 * clears (turn committed) while the assistant hasn't started speaking yet.
 */
export function useThinkingHint(
  userText: string,
  assistantSpeaking: boolean,
  holdMs = 2500
): boolean {
  const [thinking, setThinking] = useState(false);
  const hadTextRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    if (userText) {
      hadTextRef.current = true;
      return;
    }
    if (hadTextRef.current && assistantSpeaking) {
      // userText currently clears at the model turn boundary. If audio is
      // still playing, consume the edge rather than showing "thinking" after
      // the answer has finished.
      hadTextRef.current = false;
      return;
    }
    if (hadTextRef.current) {
      hadTextRef.current = false;
      setThinking(true);
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
      timerRef.current = window.setTimeout(() => setThinking(false), holdMs);
    }
  }, [userText, assistantSpeaking, holdMs]);

  useEffect(() => {
    if (assistantSpeaking) {
      setThinking(false);
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    }
  }, [assistantSpeaking]);

  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    []
  );

  return thinking;
}
