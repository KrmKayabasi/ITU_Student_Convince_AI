# -*- coding: utf-8 -*-
"""
Shared metrics formatting for the İTÜ Student Convince AI desktop client.
Extracted from desktop_client.py to keep the GUI module focused on UI concerns.
"""

from __future__ import annotations


def _fmt(x, nd=3):
    """Format a value for display — floats to N decimal places, others as-is."""
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def format_metrics(debug: dict | None, profile: dict | None, focus: dict | None) -> str:
    """Render live CV metrics as a human-readable text block for the metrics panel."""
    lines: list[str] = []
    if debug is not None:
        raw = debug.get("raw", {})
        smoothed = debug.get("smoothed", {})
        live = debug.get("live_score_preview", {})
        emotion_scores = raw.get("emotion_scores") or {}
        top_emotion = ", ".join(
            f"{k}={_fmt(v, 2)}" for k, v in sorted(emotion_scores.items(), key=lambda kv: -kv[1])[:3]
        )
        lines += [
            f"state={debug.get('state')}  face={raw.get('face_present')}",
            f"focused={debug.get('is_focused')}  focus_time={_fmt(debug.get('focus_time'))}",
            f"lean(raw)={_fmt(raw.get('lean'))}  delta_lean={_fmt(smoothed.get('delta_lean'))}  baseline={_fmt(smoothed.get('baseline_lean'))}",
            f"eye_contact(raw)={_fmt(raw.get('eye_contact'))}  avg={_fmt(smoothed.get('avg_eye_contact'))}  head_yaw={_fmt(raw.get('head_yaw_deg'), 1)}",
            f"spine_ratio={_fmt(raw.get('spine_ratio'))}  shoulder_tilt={_fmt(raw.get('shoulder_tilt'))}  arms_crossed={raw.get('arms_crossed')}",
            f"emotion={raw.get('emotion_label')}  ({top_emotion})",
            f"live attn={_fmt(live.get('attention'))}  open={_fmt(live.get('openness'))}  energy={_fmt(live.get('energy'))}",
        ]
    else:
        lines.append("/debug baglantisi bekleniyor...")

    if focus is not None:
        lines.append(f"[/focus] is_focused={focus.get('is_focused')}  focus_time={_fmt(focus.get('focus_time'))}")

    if profile is not None:
        scores = profile.get("scores", {})
        lines.append(
            f"[/profile TEK SEFERLIK] attn={scores.get('attention')} open={scores.get('openness')} energy={scores.get('energy')}"
        )

    return "\n".join(lines)
