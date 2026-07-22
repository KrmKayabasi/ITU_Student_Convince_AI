"""
FastAPI router for speaker enrollment and management (speech_backend).

Provides REST endpoints for:
- Enrolling a target speaker from audio
- Verifying audio against enrolled profiles
- Listing, removing, and resetting profiles

Mounted by the speech_backend server.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger(__name__)


def create_speaker_router(speaker_manager) -> APIRouter:
    """
    Create a FastAPI router with speaker management endpoints.

    Args:
        speaker_manager: A SpeakerManager instance (must be initialized).
    """
    router = APIRouter(prefix="/v1/speaker", tags=["speaker"])

    @router.post("/enroll")
    async def enroll_speaker(
        request: Request,
        speaker_id: str = Query("target_user", description="Speaker identifier"),
    ):
        """Enroll a target speaker from raw PCM float32 16kHz mono audio."""
        if not speaker_manager.is_ready:
            raise HTTPException(status_code=503, detail="Speaker manager not initialized")

        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="No audio data provided")

        try:
            audio = np.frombuffer(body, dtype=np.float32)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {exc}")

        if len(audio) < 16000:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short: {len(audio)} samples (minimum 16000)",
            )

        result = await speaker_manager.enroll(speaker_id, audio)
        return result

    @router.post("/verify")
    async def verify_speaker(
        request: Request,
        speaker_id: str = Query("target_user", description="Speaker to verify against"),
    ):
        """Verify audio against an enrolled speaker."""
        if not speaker_manager.is_ready:
            raise HTTPException(status_code=503, detail="Speaker manager not initialized")

        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="No audio data provided")

        try:
            audio = np.frombuffer(body, dtype=np.float32)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid audio data: {exc}")

        is_match, score, matched_id = await speaker_manager.verify(audio, speaker_id)

        return {
            "match": is_match,
            "speaker_id": matched_id,
            "score": round(score, 4),
            "threshold": speaker_manager.verify_threshold,
        }

    @router.get("/profiles")
    async def list_profiles():
        """List all enrolled speaker profiles."""
        if not speaker_manager.is_ready:
            raise HTTPException(status_code=503, detail="Speaker manager not initialized")
        profiles = speaker_manager.get_profiles()
        return {
            "profiles": profiles,
            "target_speaker_id": speaker_manager.target_speaker_id,
        }

    @router.delete("/profiles/{speaker_id}")
    async def delete_profile(speaker_id: str):
        """Remove a specific speaker profile."""
        if not speaker_manager.is_ready:
            raise HTTPException(status_code=503, detail="Speaker manager not initialized")
        deleted = speaker_manager.remove_profile(speaker_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Speaker '{speaker_id}' not found")
        return {"status": "deleted", "speaker_id": speaker_id}

    @router.post("/reset")
    async def reset_profiles():
        """Remove all enrolled speaker profiles."""
        if not speaker_manager.is_ready:
            raise HTTPException(status_code=503, detail="Speaker manager not initialized")
        count = speaker_manager.reset_all()
        return {"status": "reset", "deleted_count": count}

    @router.get("/health")
    async def speaker_health():
        """Health check for the speaker subsystem."""
        return {
            "status": "ok" if speaker_manager.is_ready else "not_initialized",
            "enabled": speaker_manager.enabled,
            "bargein_enabled": speaker_manager.bargein_enabled,
            "target_speaker_id": speaker_manager.target_speaker_id,
            "profiles_count": (
                speaker_manager._db.speaker_count if speaker_manager._db else 0
            ),
        }

    return router
