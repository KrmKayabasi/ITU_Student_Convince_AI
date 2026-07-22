"""
Speaker embedding engine using sherpa-onnx with TitaNet / CAM++ / ResNet models.

Provides:
- Embedding extraction from audio samples
- Speaker enrollment and management (add, remove, search, verify)
- Profile persistence (save/load to JSON)

Integrated from target_speaker_pipeline/lib/speaker_engine.py
"""

from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import sherpa_onnx


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

def _resolve_model_path(model_path: str = "") -> str:
    """
    Resolve the speaker embedding model path.

    Checks (in order):
    1. Explicitly provided path (if not empty)
    2. SPEAKER_MODEL_PATH environment variable (unconditional — Docker sets this)
    3. models/nemo_en_titanet_small.onnx relative to project root
    4. models/nemo_en_titanet_small.onnx relative to cwd
    """
    if model_path:
        return model_path

    env_path = os.environ.get("SPEAKER_MODEL_PATH", "")
    if env_path:
        return env_path

    # Try relative to this file's location (backend/speaker/ -> ../../)
    candidate = Path(__file__).parent.parent.parent / "models" / "nemo_en_titanet_small.onnx"
    if candidate.exists():
        return str(candidate)

    # Try relative to cwd
    cwd_candidate = Path("models") / "nemo_en_titanet_small.onnx"
    if cwd_candidate.exists():
        return str(cwd_candidate)

    # Return the project-relative default even if it doesn't exist yet
    return str(candidate)


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

class SpeakerEmbeddingEngine:
    """
    Wraps sherpa-onnx SpeakerEmbeddingExtractor for extracting
    speaker-discriminative embeddings from audio samples.

    Supports: TitaNet (192-dim), ResNet34-LM (256-dim), CAM++ (512-dim)
    """

    def __init__(
        self,
        model_path: str = "",
        num_threads: int = 4,
        provider: str = "cpu",
        debug: bool = False,
    ):
        resolved = _resolve_model_path(model_path)
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=resolved,
            num_threads=num_threads,
            debug=debug,
            provider=provider,
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.model_path = resolved

    @property
    def dim(self) -> int:
        return self._extractor.dim

    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extract a speaker embedding from an audio waveform.

        Args:
            audio: 1-D float32 array of audio samples in [-1, 1].
            sample_rate: Audio sample rate (must be 16000 for TitaNet).

        Returns:
            L2-normalized 1-D float32 embedding vector.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()

        stream = self._extractor.create_stream()
        stream.accept_waveform(float(sample_rate), audio.tolist())
        stream.input_finished()

        embedding = np.array(self._extractor.compute(stream), dtype=np.float32)

        # L2 normalize
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-8:
            embedding /= norm

        return embedding

    def extract_batch(
        self,
        audio_list: list[np.ndarray],
        sample_rate: int = 16000,
    ) -> list[np.ndarray]:
        """Extract embeddings from a list of audio samples."""
        return [self.extract(a, sample_rate) for a in audio_list]


# ---------------------------------------------------------------------------
# Speaker profile storage
# ---------------------------------------------------------------------------

class SpeakerProfile:
    """Serializable speaker profile with embedding history and metadata."""

    def __init__(
        self,
        speaker_id: str,
        embeddings: list[np.ndarray] | None = None,
        metadata: dict | None = None,
    ):
        self.speaker_id = speaker_id
        self.embeddings: list[np.ndarray] = embeddings or []
        self.metadata: dict = metadata or {}
        self.created_at: float = time.time()
        self.updated_at: float = time.time()

    @property
    def centroid(self) -> np.ndarray | None:
        """L2-normalized mean embedding."""
        if not self.embeddings:
            return None
        mean = np.mean(self.embeddings, axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean /= norm
        return mean

    @property
    def quality(self) -> float:
        """
        Profile quality score (0-1).
        Based on: number of embeddings, embedding consistency.
        """
        if len(self.embeddings) < 3:
            return 0.3 * len(self.embeddings)

        centroid = self.centroid
        if centroid is None:
            return 0.0

        similarities = [
            float(np.dot(e / (np.linalg.norm(e) + 1e-8), centroid))
            for e in self.embeddings
        ]
        mean_sim = float(np.mean(similarities))
        count_factor = min(1.0, len(self.embeddings) / 20.0)
        return mean_sim * 0.8 + count_factor * 0.2

    def add_embedding(self, embedding: np.ndarray, max_stored: int = 50) -> None:
        """Add an embedding, maintaining rolling window."""
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm
        self.embeddings.append(emb)
        if len(self.embeddings) > max_stored:
            self.embeddings = self.embeddings[-max_stored:]
        self.updated_at = time.time()

    def to_dict(self) -> dict:
        return {
            "speaker_id": self.speaker_id,
            "embeddings": [e.tolist() for e in self.embeddings],
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SpeakerProfile":
        profile = cls(
            speaker_id=data["speaker_id"],
            embeddings=[np.array(e, dtype=np.float32) for e in data.get("embeddings", [])],
            metadata=data.get("metadata", {}),
        )
        profile.created_at = data.get("created_at", time.time())
        profile.updated_at = data.get("updated_at", time.time())
        return profile


# ---------------------------------------------------------------------------
# Speaker database (persistent)
# ---------------------------------------------------------------------------

class SpeakerDatabase:
    """
    Persistent speaker profile database backed by JSON files.

    Thread-safe for concurrent read/write from verification and enrollment paths.
    """

    def __init__(self, storage_dir: str = "profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: dict[str, SpeakerProfile] = {}
        self._manager: sherpa_onnx.SpeakerEmbeddingManager | None = None
        self._lock = threading.Lock()
        self._load_all()

    # -- manager -------------------------------------------------------

    def set_manager(self, manager: sherpa_onnx.SpeakerEmbeddingManager) -> None:
        """Bind a sherpa-onnx SpeakerEmbeddingManager for fast verification."""
        with self._lock:
            self._manager = manager
            # Sync existing profiles into the manager
            for speaker_id, profile in self._profiles.items():
                if profile.centroid is not None:
                    self._manager.add(speaker_id, profile.centroid.tolist())

    @property
    def manager(self) -> sherpa_onnx.SpeakerEmbeddingManager | None:
        return self._manager

    # -- CRUD ----------------------------------------------------------

    def get(self, speaker_id: str) -> SpeakerProfile | None:
        with self._lock:
            return self._profiles.get(speaker_id)

    def get_all(self) -> dict[str, SpeakerProfile]:
        with self._lock:
            return dict(self._profiles)

    def add_or_update(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        metadata: dict | None = None,
        max_stored: int = 50,
    ) -> SpeakerProfile:
        with self._lock:
            if speaker_id in self._profiles:
                profile = self._profiles[speaker_id]
                profile.add_embedding(embedding, max_stored=max_stored)
                if metadata:
                    profile.metadata.update(metadata)
            else:
                profile = SpeakerProfile(
                    speaker_id=speaker_id,
                    embeddings=[embedding],
                    metadata=metadata or {},
                )
                self._profiles[speaker_id] = profile

            # Update the manager's centroid
            if self._manager is not None and profile.centroid is not None:
                # Remove old entry and re-add with updated centroid
                if speaker_id in self._manager:
                    self._manager.remove(speaker_id)
                self._manager.add(speaker_id, profile.centroid.tolist())

            self._save(speaker_id, profile)
            return profile

    def remove(self, speaker_id: str) -> bool:
        """Remove a speaker profile. Returns True if it existed."""
        with self._lock:
            if speaker_id not in self._profiles:
                return False
            del self._profiles[speaker_id]
            if self._manager is not None and speaker_id in self._manager:
                self._manager.remove(speaker_id)
            self._delete_file(speaker_id)
            return True

    def verify(
        self,
        speaker_id: str,
        embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[bool, float]:
        """
        Verify if an embedding matches a specific speaker.

        Returns (is_match, confidence_score).
        """
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm

        with self._lock:
            if self._manager is not None:
                score = self._manager.score(speaker_id, emb.tolist())
                return score >= threshold, score

            # Fallback: manual cosine similarity
            profile = self._profiles.get(speaker_id)
            if profile is None or profile.centroid is None:
                return False, 0.0

            score = float(np.dot(emb, profile.centroid))
            return score >= threshold, score

    def search(
        self,
        embedding: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[str | None, float]:
        """
        Find the best-matching speaker for an embedding.

        Returns (speaker_id or None, best_score).
        """
        emb = np.asarray(embedding, dtype=np.float32)
        norm = float(np.linalg.norm(emb))
        if norm > 1e-8:
            emb = emb / norm

        with self._lock:
            if self._manager is not None:
                result = self._manager.search(emb.tolist(), threshold)
                if result:
                    score = self._manager.score(result, emb.tolist())
                    return result, score
                return None, 0.0

            # Fallback
            best_id, best_score = None, 0.0
            for sid, profile in self._profiles.items():
                if profile.centroid is not None:
                    score = float(np.dot(emb, profile.centroid))
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_id = sid
            return best_id, best_score

    @property
    def speaker_count(self) -> int:
        with self._lock:
            return len(self._profiles)

    @property
    def all_speaker_ids(self) -> list[str]:
        with self._lock:
            return list(self._profiles.keys())

    # -- persistence ---------------------------------------------------

    def _profile_path(self, speaker_id: str) -> Path:
        safe_name = "".join(c for c in speaker_id if c.isalnum() or c in "_-")
        return self.storage_dir / f"{safe_name}.json"

    def _save(self, speaker_id: str, profile: SpeakerProfile) -> None:
        path = self._profile_path(speaker_id)
        tmp_path = path.with_suffix(".tmp")
        data = profile.to_dict()
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        tmp_path.replace(path)  # atomic rename

    def _load_all(self) -> None:
        for path in self.storage_dir.glob("*.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                profile = SpeakerProfile.from_dict(data)
                self._profiles[profile.speaker_id] = profile
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"[SpeakerDB] Warning: skipping corrupt profile {path}: {exc}")

    def _delete_file(self, speaker_id: str) -> None:
        path = self._profile_path(speaker_id)
        if path.exists():
            path.unlink()

    def flush(self) -> None:
        """Force-save all profiles to disk."""
        with self._lock:
            for sid, profile in self._profiles.items():
                self._save(sid, profile)
