"use client";

/**
 * useFaceRig — the single-rAF animation engine behind AdvisorFace.
 *
 * Rules (docs/UI_DESIGN.md §4):
 *  - ONE rAF loop computes every channel each frame and writes SVG
 *    `transform`/`opacity` attributes directly on [data-rig] nodes.
 *    React never re-renders the face.
 *  - Pivot-baked transform strings (translate(p) rotate scale translate(-p)).
 *  - Frame-rate-corrected exponential smoothing: k = 1 − exp(−dt/τ).
 *  - Lip-sync: perceptual amplitude with asymmetric EMA (fast attack ~40ms,
 *    soft release ~110ms), gated to 0 outside `speaking` (barge-in closes the
 *    mouth in a few frames).
 *  - Blink = keyframed lid-slab translateY (70ms down / 40ms hold / 130ms up).
 */

import { useEffect, useRef, type RefObject } from "react";
import type { FaceState } from "./faceState";
import type { AmplitudeSource } from "./amplitude";

export interface FaceRigProps {
  state: FaceState;
  amplitude: AmplitudeSource;
  seekAttentionNonce: number;
}

const SEEK_MS = 2600;

function pivot(
  px: number,
  py: number,
  o: { dx?: number; dy?: number; rot?: number; sx?: number; sy?: number }
): string {
  const { dx = 0, dy = 0, rot = 0, sx = 1, sy = 1 } = o;
  return `translate(${px + dx} ${py + dy}) rotate(${rot}) scale(${sx} ${sy}) translate(${-px} ${-py})`;
}

export function useFaceRig(
  rootRef: RefObject<SVGSVGElement | null>,
  props: FaceRigProps
): void {
  const stateRef = useRef<FaceState>(props.state);
  const ampRef = useRef<AmplitudeSource>(props.amplitude);
  const seekUntilRef = useRef(0);
  const lastNonceRef = useRef(props.seekAttentionNonce);

  useEffect(() => {
    stateRef.current = props.state;
  }, [props.state]);
  useEffect(() => {
    ampRef.current = props.amplitude;
  }, [props.amplitude]);
  useEffect(() => {
    if (props.seekAttentionNonce !== lastNonceRef.current) {
      lastNonceRef.current = props.seekAttentionNonce;
      seekUntilRef.current = performance.now() + SEEK_MS;
    }
  }, [props.seekAttentionNonce]);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;

    const nodes = new Map<string, Element>();
    root.querySelectorAll("[data-rig]").forEach((el) => {
      const name = el.getAttribute("data-rig");
      if (name) nodes.set(name, el);
    });
    const setT = (name: string, value: string) =>
      nodes.get(name)?.setAttribute("transform", value);
    const setA = (name: string, attr: string, value: string) =>
      nodes.get(name)?.setAttribute(attr, value);

    // ── mutable channel state (no allocations in the loop) ──
    const cur = {
      headRot: 0,
      headY: 0,
      headScale: 1,
      browRaise: 0,
      browConcern: 0,
      gazeX: 0,
      gazeY: 0,
      smile: 0.35,
      open: 0,
      energy: 0,
      blush: 0.5,
    };

    let blinkStart = -1;
    let nextBlink = performance.now() + 1800;
    let lookX = 0;
    let lookY = 0;
    let lookUntil = 0;
    let nextLook = performance.now() + 4000;
    let nodStart = -1;
    let nextNod = performance.now() + 5000;

    let raf = 0;
    let last = performance.now();
    let running = true;

    const frame = (now: number) => {
      if (!running) return;
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const t = now / 1000;
      const st = stateRef.current;
      const k = (tau: number) => 1 - Math.exp(-dt / tau);

      // ── lip-sync amplitude (asymmetric EMA, gated to speaking) ──
      const rawAmp = st === "speaking" ? ampRef.current.read() : 0;
      cur.open += (rawAmp - cur.open) * k(rawAmp > cur.open ? 0.04 : 0.11);
      cur.energy += (cur.open - cur.energy) * k(0.6);

      // ── per-state targets ──
      let tHeadRot = 0;
      let tHeadY = 0;
      let tScale = 1;
      let tBrowRaise = 0;
      let tConcern = 0;
      let tGazeX = 0;
      let tGazeY = 0;
      let tSmile = 0.35;
      let tBlush = 0.5;

      switch (st) {
        case "attract":
          tSmile = 0.5;
          tHeadRot = 1.2 * Math.sin(0.31 * t) + 0.5 * Math.sin(0.83 * t);
          if (now > nextLook) {
            lookX = Math.random() * 10 - 5;
            lookY = Math.random() * 4 - 2;
            lookUntil = now + 1600;
            nextLook = now + 6000 + Math.random() * 4000;
          }
          if (now < lookUntil) {
            tGazeX = lookX;
            tGazeY = lookY;
          }
          break;
        case "connecting":
          tSmile = 0.2;
          tBrowRaise = 0.2;
          tGazeY = 3;
          tHeadRot = 0.6 * Math.sin(1.7 * t);
          break;
        case "listening":
          tHeadRot = 2.5;
          tSmile = 0.35;
          if (now > nextNod && nodStart < 0) {
            nodStart = now;
            nextNod = now + 6000 + Math.random() * 3000;
          }
          break;
        case "speaking":
          tSmile = 0.3;
          tBrowRaise = 0.15;
          tGazeX = 0.9 * Math.sin(2.1 * t) + 0.5 * Math.sin(3.7 * t);
          tHeadY = 1.5 * cur.open;
          tHeadRot = 0.7 * Math.sin(2 * Math.PI * 1.9 * t) * cur.energy;
          break;
        case "thinking":
          tSmile = 0.1;
          tBrowRaise = 0.3;
          tGazeX = 5;
          tGazeY = -4;
          break;
        case "concerned":
          tConcern = 0.8;
          tSmile = -0.3;
          tBlush = 0.15;
          break;
      }

      // ── nod gesture (listening) ──
      if (nodStart >= 0) {
        const p = (now - nodStart) / 550;
        if (p >= 1) nodStart = -1;
        else tHeadY += 4 * Math.sin(Math.PI * p);
      }

      // ── seekAttention overlay (layered on any state) ──
      if (now < seekUntilRef.current) {
        const p = 1 - (seekUntilRef.current - now) / SEEK_MS;
        const e = Math.sin(Math.PI * Math.min(1, Math.max(0, p)));
        tScale = 1 + 0.06 * e;
        tHeadY -= 8 * e;
        tBrowRaise = Math.max(tBrowRaise, e);
        tSmile = Math.max(tSmile, 0.6 * e);
        if (p < 0.05 && blinkStart < 0) blinkStart = now; // attention blink
      }

      // ── smoothing toward targets ──
      cur.headRot += (tHeadRot - cur.headRot) * k(0.3);
      cur.headY += (tHeadY - cur.headY) * k(0.2);
      cur.headScale += (tScale - cur.headScale) * k(0.2);
      cur.browRaise += (tBrowRaise - cur.browRaise) * k(0.15);
      cur.browConcern += (tConcern - cur.browConcern) * k(0.25);
      cur.gazeX += (tGazeX - cur.gazeX) * k(0.12);
      cur.gazeY += (tGazeY - cur.gazeY) * k(0.12);
      cur.smile += (tSmile - cur.smile) * k(0.25);
      cur.blush += (tBlush - cur.blush) * k(0.5);

      // ── blink keyframes ──
      if (blinkStart < 0 && now >= nextBlink) {
        if (st === "speaking" && cur.open > 0.5) {
          nextBlink = now + 300; // don't blink mid-loud-vowel
        } else {
          blinkStart = now;
          nextBlink =
            now + (st === "attract" ? 3400 : 2800) + Math.random() * 3200;
        }
      }
      let blink = 0;
      if (blinkStart >= 0) {
        const bt = now - blinkStart;
        if (bt < 70) blink = bt / 70;
        else if (bt < 110) blink = 1;
        else if (bt < 240) blink = 1 - (bt - 110) / 130;
        else {
          blink = 0;
          blinkStart = -1;
        }
      }

      // ── derived values ──
      const smilePos = Math.max(0, cur.smile);
      const breathY =
        (st === "attract" ? 1.9 : 1.2) * Math.sin(2 * Math.PI * 0.22 * t);
      const lidY = blink * 24;
      const mouthSy = 0.08 + 0.92 * cur.open;
      const mouthSx =
        (1 - 0.16 * cur.open * cur.open) * (1 + 0.05 * smilePos);
      const seamY = 341 - 8 * cur.smile + 4 * cur.open;
      const strandRot = 1.2 * Math.sin(0.9 * t);
      const earringRot = -cur.headRot * 1.8 + 1.5 * Math.sin(1.3 * t);

      // ── attribute writes (~22) ──
      setT("breath", `translate(0 ${breathY})`);
      setT(
        "head",
        pivot(210, 415, {
          dy: cur.headY,
          rot: cur.headRot,
          sx: cur.headScale,
          sy: cur.headScale,
        })
      );
      setT("gaze-l", `translate(${cur.gazeX} ${cur.gazeY})`);
      setT("gaze-r", `translate(${-cur.gazeX} ${cur.gazeY})`); // mirrored wrapper
      setT("lid-l", `translate(0 ${lidY})`);
      setT("lid-r", `translate(0 ${lidY})`);
      setT("lash-l", `translate(0 ${lidY})`);
      setT("lash-r", `translate(0 ${lidY})`);
      setT("lowerlid-l", `translate(0 ${-2 * smilePos})`);
      setT("lowerlid-r", `translate(0 ${-2 * smilePos})`);
      setT(
        "brow-l",
        pivot(188, 224, { dy: -6 * cur.browRaise, rot: 7 * cur.browConcern })
      );
      setT(
        "brow-r",
        pivot(188, 224, { dy: -6 * cur.browRaise, rot: 7 * cur.browConcern })
      );
      setT("mouth", pivot(210, 340, { sx: mouthSx, dy: -2 * smilePos }));
      setT("mouth-open", pivot(210, 341, { sy: mouthSy }));
      setT("lip-lower", `translate(0 ${11 * cur.open})`);
      setT("lip-upper", `translate(0 ${-2 * cur.open})`);
      setA("mouth-seam", "d", `M 179 340 Q 210 ${seamY} 241 340`);
      setA("blush", "opacity", cur.blush.toFixed(3));
      setA("crease-l", "opacity", (smilePos * 0.55).toFixed(3));
      setA("crease-r", "opacity", (smilePos * 0.55).toFixed(3));
      setT("strand-l", pivot(120, 268, { rot: strandRot }));
      setT("strand-r", pivot(300, 268, { rot: -strandRot }));
      setT("earring-l", pivot(119, 281, { rot: earringRot }));
      setT("earring-r", pivot(119, 281, { rot: earringRot }));

      raf = requestAnimationFrame(frame);
    };

    const onVisibility = () => {
      if (document.hidden) {
        running = false;
        cancelAnimationFrame(raf);
      } else if (!running) {
        running = true;
        last = performance.now();
        raf = requestAnimationFrame(frame);
      }
    };

    document.addEventListener("visibilitychange", onVisibility);
    raf = requestAnimationFrame(frame);

    return () => {
      running = false;
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVisibility);
    };
    // The rig binds once; live values flow through refs.
  }, [rootRef]);
}
