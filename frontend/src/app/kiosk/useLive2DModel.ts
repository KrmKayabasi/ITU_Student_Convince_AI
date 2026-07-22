"use client";

/**
 * useLive2DModel — mounts a Live2D Cubism 4 model on a <canvas> via
 * pixi-live2d-display (+ pixi.js v6), and exposes the loaded model so the rig
 * (useLive2DRig) can write Live2D parameters each frame.
 *
 * Contract mirrors the SVG path: the caller passes a FaceState-derived state,
 * an AmplitudeSource (the SAME AnalyserNode RMS the SVG rig consumes), and a
 * seekAttentionNonce. The hook is SSR-safe (PIXI is dynamically imported).
 *
 * The Cubism Core script (public/live2d/live2dcubismcore.min.js) must be
 * present on window BEFORE pixi-live2d-display boots. We inject it lazily here
 * so the kiosk route never pays for it unless `?avatar=live2d` is active.
 */

import { useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

const CUBISM_CORE_URL = "/live2d/live2dcubismcore.min.js";
export const DEFAULT_MODEL_URL = "/live2d/models/Haru/haru_greeter_t03.model3.json";

// Declared inline to avoid a @types dependency on pixi-live2d-display (which
// references pixi v6 types that clash with strict TS settings). We only touch
// a tiny slice of the API.
export interface Live2DModelLike {
  // pixi-live2d-display exposes the internal model for direct param writes.
  internalModel: {
    on(event: "afterMotionUpdate" | "beforeModelUpdate", listener: () => void): void;
    off(event: "afterMotionUpdate" | "beforeModelUpdate", listener: () => void): void;
    coreModel: {
      setParameterValueById(id: string, value: number): void;
      getParameterValueById(id: string): number;
    };
  };
  // PIXI.Container-ish: we use anchor + scale + position.
  anchor: { set(x: number, y: number): void };
  scale: { set(s: number): void };
  position: { set(x: number, y: number): void };
  width: number;
  height: number;
  update: (dt: number) => void;
  destroy(options?: {
    children?: boolean;
    texture?: boolean;
    baseTexture?: boolean;
  }): void;
}

/** Minimal PIXI ticker shape used by the shared application clock. */
export interface TickerLike {
  add(
    fn: (dt: number) => void,
    context?: unknown,
    priority?: number,
  ): void;
  remove(
    fn: (dt: number) => void,
    context?: unknown,
  ): void;
}

/** Minimal PIXI.Application shape we touch (avoids importing pixi types). */
export interface PixiApp {
  renderer: { resize: (w: number, h: number) => void };
  stage: { addChild: (child: unknown) => void };
  ticker: TickerLike;
  destroy: (
    removeView?: boolean,
    stageOptions?: {
      children?: boolean;
      texture?: boolean;
      baseTexture?: boolean;
    },
  ) => void;
}

/** Minimal PIXI namespace shape used to construct the app. */
export interface PixiCtor {
  Application: new (opts: Record<string, unknown>) => PixiApp;
  /** PIXI.Ticker class — passed to Live2DModel.registerTicker(). */
  Ticker: TickerClass;
}

/**
 * The PIXI.Ticker *class*. pixi-live2d-display's registerTicker stores this and
 * later reads `.shared` off it (a static). Passing the whole PIXI namespace
 * instead breaks that — `.shared` would be undefined.
 */
export interface TickerClass {
  shared: TickerLike;
}

/** Minimal pixi-live2d-display/cubism4 module shape. */
export interface Live2DModule {
  Live2DModel: {
    registerTicker: (tickerClass: TickerClass) => void;
    from: (url: string) => Promise<Live2DModelLike>;
  };
}

export interface Live2DModelHandle {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  modelRef: RefObject<Live2DModelLike | null>;
  ready: boolean;
  error: string | null;
}

let _corePromise: Promise<void> | null = null;

/** Inject the Cubism Core script once, idempotently. */
function ensureCubismCore(): Promise<void> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("window unavailable (SSR)"));
  }
  // The global is set by the core script once it parses.
  if ((window as unknown as { Live2DCubismCore?: unknown }).Live2DCubismCore) {
    return Promise.resolve();
  }
  if (_corePromise) return _corePromise;
  _corePromise = new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = CUBISM_CORE_URL;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => {
      _corePromise = null;
      reject(new Error(`failed to load ${CUBISM_CORE_URL}`));
    };
    document.head.appendChild(s);
  });
  return _corePromise;
}

export function useLive2DModel(modelUrl: string = DEFAULT_MODEL_URL): Live2DModelHandle {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const modelRef = useRef<Live2DModelLike | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let app: PixiApp | null = null;
    let resizeObs: ResizeObserver | null = null;

    async function boot() {
      if (!canvasRef.current) return;
      try {
        console.log("[L2D] boot start");
        await ensureCubismCore();
        if (cancelled) return;
        console.log("[L2D] cubism core ready, importing pixi");

        // Dynamic import keeps pixi out of the SSR bundle. pixi.js v6 exports
        // its API as a namespace (NO .default), so we take the module object
        // directly. The pixi v6 + pixi-live2d-display types clash with this
        // repo's strict TS config, so we import untyped and go through minimal
        // local shapes.
        const pixiModule = await import("pixi.js");
        if (cancelled) return;
        const PIXI = (pixiModule.default ?? pixiModule) as unknown as PixiCtor;
        console.log("[L2D] PIXI loaded; Application?", typeof PIXI?.Application, "Ticker?", typeof PIXI?.Ticker, "Ticker.shared?", typeof PIXI?.Ticker?.shared);
        // pixi-live2d-display reads PIXI off `window.PIXI` internally (NOT the
        // argument passed to registerTicker — see cubism4.js: "window.PIXI.Ticker").
        // Without this global, it throws "o.shared is undefined" on first frame.
        (window as unknown as { PIXI?: PixiCtor }).PIXI = PIXI;
        const live2dNs = await import("pixi-live2d-display/cubism4");
        if (cancelled) return;
        const live2dModule = (live2dNs.default ?? live2dNs) as unknown as Live2DModule;
        console.log("[L2D] live2d module loaded; Live2DModel?", typeof live2dModule?.Live2DModel);
        const { Live2DModel } = live2dModule;
        // registerTicker expects the PIXI.Ticker CLASS (it stores it and later
        // reads tickerRef.shared). Passing the whole PIXI namespace makes
        // tickerRef.shared undefined → "Cannot read properties of undefined
        // (reading 'add')" inside the library's set autoUpdate (cubism4.js:4858).
        Live2DModel.registerTicker(PIXI.Ticker);

        const canvas = canvasRef.current;
        // sharedTicker: true so the app, the Live2D model updates, and our rig's
        // per-frame param writes all run on the single PIXI.Ticker.shared —
        // pixi-live2d-display is hardwired to that ticker (cubism4.js:4854).
        const appAny = new PIXI.Application({
          view: canvas,
          autoStart: true,
          backgroundAlpha: 0,
          antialias: true,
          resolution: Math.min(window.devicePixelRatio || 1, 2),
          autoDensity: true,
          sharedTicker: true,
          width: canvas.clientWidth || 400,
          height: canvas.clientHeight || 560,
        }) as PixiApp;
        console.log("[L2D] app created; ticker?", typeof appAny?.ticker, "ticker.add?", typeof appAny?.ticker?.add);
        app = appAny;
        console.log("[L2D] loading model:", modelUrl);
        const model = (await Live2DModel.from(modelUrl)) as Live2DModelLike;
        console.log("[L2D] model loaded OK:", (model as unknown as { internalModel?: { coreModel?: unknown } })?.internalModel ? "has coreModel" : "NO coreModel");
        if (cancelled) {
          model.destroy({ children: true });
          return;
        }
        modelRef.current = model;
        appAny.stage.addChild(model);

        const fit = () => {
          if (!canvasRef.current) return;
          const w = canvasRef.current.clientWidth;
          const h = canvasRef.current.clientHeight;
          appAny.renderer.resize(w, h);
          // Diagnostic: dump every size we can read so the fit math is grounded
          // in real numbers, not guesses.
          const mAny = model as unknown as {
            width: number; height: number;
            internalModel?: { originalWidth?: number; originalHeight?: number; width?: number; height?: number };
          };
          console.log("[L2D] fit: canvas", w, "x", h,
            "| model.width/height", mAny.width, mAny.height,
            "| internal.original", mAny.internalModel?.originalWidth, mAny.internalModel?.originalHeight,
            "| internal.w/h", mAny.internalModel?.width, mAny.internalModel?.height);
          // Face-focused framing: zoom in so the face + upper torso fill the
          // frame (like the SVG Elif portrait), cropping the lower body. We
          // scale the model so its native height is ~1.72x the canvas height,
          // then anchor near the face and shift up so the head sits centered.
          const modelH = mAny.internalModel?.originalHeight || mAny.height || h;
          const scale = (h * 1.72) / modelH;
          model.scale.set(scale);
          // Anchor at (0.5, ~0.32) — roughly the face center on most Cubism
          // models (head occupies the upper third). This is the pivot the
          // position offset is measured from.
          model.anchor.set(0.5, 0.32);
          // Keep the tighter upper-body crop while leaving enough headroom for
          // the full face.
          model.position.set(w / 2, h * 0.47);
        };
        fit();
        resizeObs = new ResizeObserver(fit);
        resizeObs.observe(canvasRef.current);

        setReady(true);
        console.log("[L2D] boot complete, ready=true");
      } catch (e) {
        console.error("[L2D] BOOT FAILED — full error:", e);
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
        }
      }
    }

    boot();

    return () => {
      cancelled = true;
      resizeObs?.disconnect();
      try {
        // React owns the canvas, and PIXI may cache textures by URL for the
        // next model instance. Destroy the model/core without removing either.
        app?.destroy(false, { children: true });
      } catch {
        /* ignore */
      }
      app = null;
      modelRef.current = null;
      setReady(false);
    };
  }, [modelUrl]);

  return { canvasRef, modelRef, ready, error };
}
