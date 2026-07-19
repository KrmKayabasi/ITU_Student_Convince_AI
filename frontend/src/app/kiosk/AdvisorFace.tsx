"use client";

/**
 * AdvisorFace — "Elif", İTÜ'nün yapay zekâ tercih danışmanı.
 *
 * Hand-authored layered SVG portrait (viewBox 420×560, midline x=210,
 * eye line y=245, mouth seam y=340, neck pivot (210,415)).
 *
 * Rendering rules (see docs/UI_DESIGN.md):
 *  - React renders this SVG exactly ONCE; all animation happens in
 *    useFaceRig via direct attribute writes on [data-rig] nodes.
 *  - Zero SVG filters; all softness via gradient-to-transparent fills.
 *  - Pivot-baked transforms only (no CSS transform-origin).
 *  - Right eye/brow/ear are mirrored via matrix(-1,0,0,1,420,0) wrappers;
 *    the rig negates gaze X for the mirrored eye.
 */

import { memo, useRef } from "react";
import { useFaceRig } from "./useFaceRig";
import type { FaceState } from "./faceState";
import type { AmplitudeSource } from "./amplitude";

export interface AdvisorFaceProps {
  state: FaceState;
  amplitude: AmplitudeSource;
  seekAttentionNonce: number;
  className?: string;
}

// Face outline reused for base fill + lower-face shading overlay.
const FACE_D = [
  "M 126 196",
  "C 121 236 123 266 131 296",
  "C 139 332 160 370 194 389",
  "Q 210 396 226 389",
  "C 260 370 281 332 289 296",
  "C 297 266 299 236 294 196",
  "C 291 148 262 114 210 112",
  "C 158 114 129 148 126 196 Z",
].join(" ");

// Almond eye outline (left eye; inner corner (149,247), outer (183,243)).
const EYE_D = [
  "M 149 247",
  "C 154 236 162 232 169 232",
  "C 177 232 182 237 183 243",
  "C 179 251 171 255 163 254",
  "C 156 253 151 251 149 247 Z",
].join(" ");

/** One eye assembly. Same markup for both eyes; the right eye lives inside a
 *  mirror wrapper, so rig names differ via the `side` suffix. */
function Eye({ side }: { side: "l" | "r" }) {
  const clipId = `k-eye-clip-${side}`;
  return (
    <g>
      <defs>
        <clipPath id={clipId}>
          <path d={EYE_D} />
        </clipPath>
      </defs>
      <g clipPath={`url(#${clipId})`}>
        {/* sclera + soft upper shadow */}
        <path d={EYE_D} fill="#fbf7f0" />
        <path
          d="M 149 247 C 154 236 162 232 169 232 C 177 232 182 237 183 243 C 178 239 170 237 163 238 C 156 240 151 243 149 247 Z"
          fill="rgba(120,80,60,0.14)"
        />
        {/* iris group — gaze target */}
        <g data-rig={`gaze-${side}`}>
          <circle cx={167} cy={246} r={12.5} fill="url(#k-iris)" stroke="#2a180c" strokeWidth={1.5} />
          <circle cx={167} cy={246} r={5.2} fill="#1c0f08" />
          <circle cx={163.5} cy={242} r={2.4} fill="rgba(255,255,255,0.95)" />
          <circle cx={171} cy={250} r={1.1} fill="rgba(255,255,255,0.4)" />
        </g>
        {/* closing lid slab (skin-toned), parked above; blink = translateY 0→24 */}
        <path
          data-rig={`lid-${side}`}
          d="M 141 196 H 191 V 226 Q 166 238 141 226 Z"
          fill="url(#k-lid)"
        />
      </g>
      {/* upper lash line — rides the blink translateY */}
      <g data-rig={`lash-${side}`}>
        <path
          d="M 148 246 C 153 234 162 229 170 230 C 178 231 183 237 184 243 C 183 240 178 234 170 233 C 163 232 155 237 151 246 Z"
          fill="#241812"
        />
        <path
          d="M 182 237 L 189 232 M 184 240 L 191 237"
          stroke="#241812"
          strokeWidth={1.8}
          strokeLinecap="round"
        />
      </g>
      {/* static eyelid crease */}
      <path
        d="M 150 236 C 158 226 174 225 182 233"
        stroke="rgba(150,90,60,0.35)"
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
      />
      {/* lower lid — squints up with smile */}
      <path
        data-rig={`lowerlid-${side}`}
        d="M 152 252 C 160 258 172 258 181 250"
        stroke="rgba(150,90,60,0.3)"
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
      />
    </g>
  );
}

function Brow({ side }: { side: "l" | "r" }) {
  return (
    <path
      data-rig={`brow-${side}`}
      d="M 146 231 C 152 222 164 216 178 218 C 184 219 188 221 190 224 C 190 226 188 227 186 227 C 176 224 162 226 150 233 C 147 233 145 232 146 231 Z"
      fill="#3c2a1c"
    />
  );
}

function EarWithEarring({ side }: { side: "l" | "r" }) {
  return (
    <g>
      <ellipse cx={121} cy={262} rx={11} ry={17} fill="url(#k-skin)" />
      <path
        d="M 117 254 Q 113 262 118 270"
        stroke="rgba(150,85,55,0.4)"
        strokeWidth={2}
        fill="none"
        strokeLinecap="round"
      />
      <g data-rig={`earring-${side}`}>
        <circle cx={119} cy={281} r={2.6} fill="url(#k-gold)" />
        <path d="M 119 283 C 114 290 114 296 119 299 C 124 296 124 290 119 283 Z" fill="url(#k-gold)" />
      </g>
    </g>
  );
}

const MIRROR = "matrix(-1 0 0 1 420 0)";

function AdvisorFaceInner({
  state,
  amplitude,
  seekAttentionNonce,
  className,
}: AdvisorFaceProps) {
  const rootRef = useRef<SVGSVGElement | null>(null);
  useFaceRig(rootRef, { state, amplitude, seekAttentionNonce });

  return (
    <svg
      ref={rootRef}
      viewBox="0 0 420 560"
      className={className}
      role="img"
      aria-label="Elif — İTÜ yapay zekâ tercih danışmanı"
    >
      <defs>
        <radialGradient id="k-skin" cx="0.48" cy="0.42" r="0.75">
          <stop offset="0%" stopColor="#f9deca" />
          <stop offset="55%" stopColor="#f3cba8" />
          <stop offset="100%" stopColor="#e2ab82" />
        </radialGradient>
        <linearGradient id="k-skin-shade" x1="0" y1="0.5" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(130,75,45,0)" />
          <stop offset="55%" stopColor="rgba(130,75,45,0)" />
          <stop offset="100%" stopColor="rgba(130,75,45,0.13)" />
        </linearGradient>
        <linearGradient id="k-lid" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#f2c8a4" />
          <stop offset="100%" stopColor="#e9b890" />
        </linearGradient>
        <linearGradient id="k-hair-back" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3a2a20" />
          <stop offset="100%" stopColor="#201410" />
        </linearGradient>
        <linearGradient id="k-hair-front" x1="0" y1="0" x2="0.4" y2="1">
          <stop offset="0%" stopColor="#4a3524" />
          <stop offset="100%" stopColor="#2c1f15" />
        </linearGradient>
        <linearGradient id="k-hair-sheen" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="rgba(255,225,190,0)" />
          <stop offset="50%" stopColor="rgba(255,225,190,0.22)" />
          <stop offset="100%" stopColor="rgba(255,225,190,0)" />
        </linearGradient>
        <radialGradient id="k-iris" cx="0.4" cy="0.35" r="0.75">
          <stop offset="0%" stopColor="#9a6a38" />
          <stop offset="60%" stopColor="#68421f" />
          <stop offset="100%" stopColor="#3c2410" />
        </radialGradient>
        <linearGradient id="k-lip-upper" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#a84a58" />
          <stop offset="100%" stopColor="#8d3a48" />
        </linearGradient>
        <linearGradient id="k-lip-lower" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#d16b74" />
          <stop offset="100%" stopColor="#b04f5c" />
        </linearGradient>
        <linearGradient id="k-mouth-inner" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#54191f" />
          <stop offset="100%" stopColor="#7c2b33" />
        </linearGradient>
        <radialGradient id="k-blush" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0%" stopColor="rgba(232,126,108,0.38)" />
          <stop offset="100%" stopColor="rgba(232,126,108,0)" />
        </radialGradient>
        <linearGradient id="k-blazer" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#16294a" />
          <stop offset="100%" stopColor="#0e1c36" />
        </linearGradient>
        <linearGradient id="k-neck-shadow" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(120,60,35,0.42)" />
          <stop offset="100%" stopColor="rgba(120,60,35,0)" />
        </linearGradient>
        <radialGradient id="k-gold" cx="0.35" cy="0.3" r="0.9">
          <stop offset="0%" stopColor="#f7c46a" />
          <stop offset="100%" stopColor="#c98a2b" />
        </radialGradient>
      </defs>

      <g data-rig="breath">
        {/* back hair: solid mass BEHIND the body so no gaps appear beside the
            neck; the neck/blazer and face paint over it */}
        <path
          d="M 210 72 C 128 72 88 134 88 210 C 88 280 98 350 94 400 C 92 442 104 470 126 482 C 156 492 190 488 210 484 C 230 488 264 492 294 482 C 316 470 328 442 326 400 C 322 350 332 280 332 210 C 332 134 292 72 210 72 Z"
          fill="url(#k-hair-back)"
        />
        {/* ── body (static; outside the head so the head tilts on the neck) ── */}
        <g>
          <path d="M 184 368 C 184 398 182 418 172 434 L 248 434 C 238 418 236 398 236 368 Q 210 382 184 368 Z" fill="url(#k-skin)" />
          <path d="M 184 368 Q 210 382 236 368 L 236 398 Q 210 404 184 398 Z" fill="url(#k-neck-shadow)" />
          {/* blazer + blouse + lapels with amber piping */}
          <path d="M 30 560 C 34 496 80 448 148 436 C 168 430 186 428 210 428 C 234 428 252 430 272 436 C 340 448 386 496 390 560 Z" fill="url(#k-blazer)" />
          <path d="M 190 432 L 210 500 L 230 432 Z" fill="#f1e8d8" />
          <path d="M 170 434 C 172 452 180 470 196 484 L 210 494 L 196 434 Z" fill="#1b3156" />
          <path d="M 250 434 C 248 452 240 470 224 484 L 210 494 L 224 434 Z" fill="#1b3156" />
          <path d="M 196 434 L 210 494 M 224 434 L 210 494" stroke="var(--k-amber, #f2a93b)" strokeWidth={2.2} strokeLinecap="round" fill="none" />
        </g>

        {/* ── head (pivot 210,415) ── */}
        <g data-rig="head">
          {/* crown: keeps the scalp attached to the head when it tilts */}
          <path
            d="M 210 72 C 128 72 88 134 88 210 C 88 250 92 280 98 300 C 106 250 110 200 128 164 C 152 118 180 100 210 96 C 240 100 268 118 292 164 C 310 200 314 250 322 300 C 328 280 332 250 332 210 C 332 134 292 72 210 72 Z"
            fill="url(#k-hair-back)"
          />

          {/* ears + earrings (right mirrored) */}
          <EarWithEarring side="l" />
          <g transform={MIRROR}>
            <EarWithEarring side="r" />
          </g>

          {/* face base + lower-face shading */}
          <path d={FACE_D} fill="url(#k-skin)" stroke="rgba(0,0,0,0.08)" strokeWidth={1.5} />
          <path d={FACE_D} fill="url(#k-skin-shade)" />

          {/* forehead sheen + chin shade (cheek depth comes from the blush) */}
          <ellipse cx={210} cy={162} rx={54} ry={15} fill="rgba(255,240,220,0.14)" />
          <ellipse cx={210} cy={372} rx={19} ry={5.5} fill="rgba(150,85,55,0.11)" />

          {/* blush (opacity channel) */}
          <g data-rig="blush" opacity={0.5}>
            <ellipse cx={162} cy={296} rx={21} ry={11} fill="url(#k-blush)" />
            <ellipse cx={258} cy={296} rx={21} ry={11} fill="url(#k-blush)" />
          </g>

          {/* nose — understated: bridge shadow, alae, base shadow, tip light */}
          <path d="M 214 252 C 216 272 218 288 222 298 C 219 301 215 302 212 302 C 214 286 213 268 211 252 Z" fill="rgba(160,95,60,0.18)" />
          <path d="M 197 297 Q 194 303 200 306 M 223 297 Q 226 303 220 306" stroke="rgba(120,60,40,0.45)" strokeWidth={2} fill="none" strokeLinecap="round" />
          <ellipse cx={210} cy={308} rx={12} ry={3.5} fill="rgba(120,60,40,0.15)" />
          <ellipse cx={208} cy={295} rx={5} ry={3} fill="rgba(255,240,225,0.35)" />

          {/* ── mouth group (pivot 210,340) ── */}
          <g data-rig="mouth">
            {/* jaw cavity — authored fully open, rests at scaleY≈0.08 */}
            <g data-rig="mouth-open">
              <path d="M 181 340 Q 210 346 239 340 Q 235 364 210 367 Q 185 364 181 340 Z" fill="url(#k-mouth-inner)" />
              <path d="M 186 341 L 234 341 L 231 351 Q 210 355 189 351 Z" fill="#f8f4ee" />
              <ellipse cx={210} cy={361} rx={13} ry={5.5} fill="#b3565e" />
            </g>
            <g data-rig="lip-upper">
              <path d="M 179 340 C 186 334 196 331 203 334 Q 210 338 217 334 C 224 331 234 334 241 340 C 232 343 221 344.5 210 344.5 C 199 344.5 188 343 179 340 Z" fill="url(#k-lip-upper)" />
            </g>
            <g data-rig="lip-lower">
              <path d="M 182 342 C 191 341.5 200 341.5 210 341.5 C 220 341.5 229 341.5 238 342 C 235 352 226 358.5 210 358.5 C 194 358.5 185 352 182 342 Z" fill="url(#k-lip-lower)" />
              <ellipse cx={210} cy={350} rx={10} ry={3} fill="rgba(255,255,255,0.15)" />
            </g>
            <path data-rig="mouth-seam" d="M 179 340 Q 210 338 241 340" stroke="rgba(80,30,36,0.6)" strokeWidth={2} fill="none" strokeLinecap="round" />
            <path data-rig="crease-l" d="M 176 336 Q 172 340 175 345" stroke="rgba(140,70,55,0.5)" strokeWidth={2} fill="none" strokeLinecap="round" opacity={0} />
            <path data-rig="crease-r" d="M 244 336 Q 248 340 245 345" stroke="rgba(140,70,55,0.5)" strokeWidth={2} fill="none" strokeLinecap="round" opacity={0} />
          </g>

          {/* eyes + brows (right side mirrored; rig negates gaze X for -r) */}
          <Eye side="l" />
          <Brow side="l" />
          <g transform={MIRROR}>
            <Eye side="r" />
            <Brow side="r" />
          </g>

          {/* front hair: side-swept curtains (part near x=188) + sheen + strands */}
          <g>
            <path
              d="M 188 102 C 148 104 118 134 112 184 C 109 220 114 250 126 268 C 134 252 132 228 136 210 C 142 172 162 136 194 120 Z"
              fill="url(#k-hair-front)"
            />
            <path
              d="M 188 102 C 232 96 272 108 292 142 C 306 168 310 200 306 232 C 300 258 296 272 294 278 C 288 252 286 224 278 202 C 266 168 238 134 200 118 Z"
              fill="url(#k-hair-front)"
            />
            <path d="M 130 200 C 136 160 156 130 186 114 C 160 136 146 168 142 206 Z" fill="url(#k-hair-sheen)" />
            <path d="M 292 170 C 280 140 252 118 220 108 C 250 124 272 146 284 178 Z" fill="url(#k-hair-sheen)" />
            <path
              data-rig="strand-l"
              d="M 120 268 C 112 320 114 380 126 430 C 130 440 138 444 142 438 C 130 392 128 330 134 276 Z"
              fill="url(#k-hair-front)"
            />
            <path
              data-rig="strand-r"
              d="M 300 268 C 308 320 306 380 294 430 C 290 440 282 444 278 438 C 290 392 292 330 286 276 Z"
              fill="url(#k-hair-front)"
            />
          </g>
        </g>
      </g>
    </svg>
  );
}

export const AdvisorFace = memo(AdvisorFaceInner);
