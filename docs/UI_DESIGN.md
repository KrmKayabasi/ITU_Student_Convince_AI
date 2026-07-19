# UI Design Spec — İTÜ Kiosk "Elif"

> Design source of truth for the kiosk UI (branch `version1`). Implemented in
> `frontend/src/app/kiosk/`. Iterate this document first, then the code.

## 1. Concept

**"Elif" — İTÜ'nün yapay zekâ tercih danışmanı.** A warm, competent young
Turkish woman advisor (late 20s), rendered as an elegant
flat-illustration-with-soft-shading vector portrait. Quality bar: premium brand
illustration, not clipart. She IS the product — the entire kiosk frames her;
subtitles, status and controls are quiet supporting cast.

Personality through visuals:
- **professional** — navy blazer with amber lapel piping, groomed brows
- **warm** — soft smile, Duchenne eye-squint, warm amber key light
- **alive** — breathing, blinking, micro-drift, eye contact, lip-sync

## 2. Art direction

### Palette (scoped to `.kiosk-theme`, see `kiosk.css`)

| Token | Value | Use |
|---|---|---|
| `--k-bg-0/1/2` | `#0a1120` / `#0f1a30` / `#152340` | background depth layers |
| `--k-navy`, `--k-navy-bright` | `#123064` / `#1d4a94` | brand surfaces |
| `--k-amber`, `--k-amber-soft` | `#f2a93b` / `#f7c46a` | primary accent, CTA |
| `--k-coral` | `#ff7a59` | warm secondary accent |
| `--k-ink`, `--k-ink-dim` | `#f5f1e8` / `#94a3bd` | text |
| `--k-ok` | `#3fd6a3` | positive status |
| ring hues | amber=speaking, teal `#4fd8c8`=listening, coral=concerned | face state ring |

Background: static radial ambient glow behind the face (navy → amber-tinged),
soft vignette. Subtle shimmer only in attract mode. **UI copy in Turkish.**
Typography: Satoshi variable — lockup 900 + amber underline, headings 600–700,
subtitles 500.

### The face

- viewBox **420×560**, "news-anchor bust" crop; midline x=210.
- Proportions: eye line at vertical skull center (**y=245**); face ≈1.32:1
  height:width; mouth width ≈ inter-pupil distance (**62 units** at y=340);
  lower lip fuller than upper; chin y≈390; head pivots at neck base (210,415).
- Long dark-brown side-swept hair (part near x=188) with gradient **sheen
  crescents**; warm brown eyes with **limbal ring + double catchlights**;
  **filled tapered brows** (never strokes); terracotta-rose lips (harmonize
  with `--k-coral`); navy blazer + amber piping; small amber earrings.
- All softness via **gradient-to-transparent fills — zero SVG filters** in the
  animated subtree (perf + crispness).

## 3. Layout (100dvh grid, never scrolls; portrait-first, 16:9-safe)

```
┌────────────────────────────────────────────┐ ~8vh
│ İTÜ ▁ Yapay Zekâ Tercih Danışmanı   ●durum │
├────────────────────────────────────────────┤ 1fr
│                 ( Elif )                    │  face = min(55dvh, 70vw)
│           thin state-colored ring           │  + pulse ring on seek
├────────────────────────────────────────────┤ minmax(20vh,auto)  ← reserved,
│  Assistant subtitle (large, ink)            │    text never moves the face
│  user subtitle (smaller, dim)               │
├────────────────────────────────────────────┤ ~10vh
│        [ Konuşmaya Başla / Bitir ]          │  + error banner slot
└────────────────────────────────────────────┘
  floating: webcam preview (bottom-right, collapsible via opacity — NEVER
  unmounted, it feeds the CV pipeline), DemoPanel (demo mode only)
```

## 4. Motion & face states

| State | Look |
|---|---|
| **attract** | kiosk idle: soft smile, scripted look-around every 6–10s, slow blinks, prominent breathing; CTA overlay |
| **connecting** | gaze slightly down, brows +raise, "getting ready" drift |
| **listening** | 2.5° head tilt, locked eye contact, nod when user finishes |
| **speaking** | lip-sync + energy-gated head-bob + livelier eyes |
| **thinking** | gaze up-side, asymmetric brow, ≤2.5s then back to listening |
| **concerned** | student unfocused: inner brows up, smile fades, blush dims |
| **seek overlay** | 2.6s lean-in + brow raise + pulse ring; synced with the orchestrator's verbal nudge (not a state — layered on any state) |

**Lip-sync**: output-analyser RMS → `tanh` perceptual curve → asymmetric EMA
(attack ~40ms / release ~110ms) → jaw-open scaleY on inner mouth (pivot at lip
seam) + lower-lip translateY + `o²` mouth-narrowing (fakes rounded vowels) +
single per-frame seam-curve `d` rewrite. Barge-in must close the mouth ≤150ms.

**Engine rules** (see `useFaceRig.ts`): single rAF loop, pivot-baked transform
strings (no CSS transform-origin), no CSS transitions on face parts, blink =
clipped skin-slab translateY (lash line rides it), frame-rate-corrected
exponential smoothing + additive gesture timelines.

## 5. Quality bar / acceptance

1. The **frozen portrait must look genuinely attractive** before any animation.
2. `/kiosk?demo=1` is the visual-QA harness: all states switchable, fake speech
   amplitude, no backend needed.
3. 60fps on integrated graphics; no long tasks from the rig.
