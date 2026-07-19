"""
İTÜ Convince AI — realtime orchestrator.

Owns a Google Gemini Live (native-audio) session per kiosk visitor, bridges
audio between the browser and Gemini, and injects Computer-Vision context into
the conversation:

  - one-time CV /profile  -> a Turkish opening hint (seeds the first turn)
  - continuous CV /focus   -> a re-engage nudge when the student looks away

The browser (Next.js frontend) is the kiosk UI; the CV pipeline (backend/
cv_pipeline, :8000) is unchanged. See docs/ARCHITECTURE.md and the plan at
.claude/plans for the full design.
"""
