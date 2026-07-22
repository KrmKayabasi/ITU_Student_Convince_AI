"""Tests for configuration loading and env-var overrides."""

from __future__ import annotations

import os
import pytest


class TestConfigDefaults:
    """Verify config values are within reasonable ranges."""

    def test_calibration_seconds_positive(self):
        from backend.cv_pipeline import config
        assert config.CALIBRATION_SECONDS > 0

    def test_no_face_timeout_positive(self):
        from backend.cv_pipeline import config
        assert config.NO_FACE_TIMEOUT_SECONDS > 0

    def test_session_gc_timeout_large_enough(self):
        from backend.cv_pipeline import config
        assert config.SESSION_GC_TIMEOUT_SECONDS >= 60, \
            "GC timeout should be at least 60 seconds"

    def test_profile_min_samples_reasonable(self):
        from backend.cv_pipeline import config
        assert 10 <= config.PROFILE_MIN_SAMPLES <= 100

    def test_focus_eye_contact_threshold_between_zero_and_one(self):
        from backend.cv_pipeline import config
        assert 0.0 < config.FOCUS_EYE_CONTACT_THRESHOLD < 1.0

    def test_tracking_defaults_are_positive_and_five_hz(self):
        from backend.cv_pipeline import config
        assert config.TRACKING_EMIT_INTERVAL_SECONDS == 0.2
        assert config.TRACKING_STALE_AFTER_SECONDS > 0

    def test_ema_alpha_between_zero_and_one(self):
        from backend.cv_pipeline import config
        assert 0.0 < config.EMA_ALPHA < 1.0

    def test_max_num_faces_positive(self):
        from backend.cv_pipeline import config
        assert config.MAX_NUM_FACES > 0

    def test_max_num_poses_positive(self):
        from backend.cv_pipeline import config
        assert config.MAX_NUM_POSES > 0

    def test_emotion_infer_hz_positive(self):
        from backend.cv_pipeline import config
        assert config.EMOTION_INFER_HZ > 0

    def test_primary_person_strategy_known(self):
        from backend.cv_pipeline import config
        assert config.PRIMARY_PERSON_STRATEGY in ("largest_bbox",)

    def test_arms_visibility_threshold_between_zero_and_one(self):
        from backend.cv_pipeline import config
        assert 0.0 < config.ARMS_VISIBILITY_THRESHOLD <= 1.0

    def test_arms_crossed_distance_ratio_positive(self):
        from backend.cv_pipeline import config
        assert 0.0 < config.ARMS_CROSSED_DISTANCE_RATIO < 2.0

    def test_spine_upright_threshold_between_zero_and_one(self):
        from backend.cv_pipeline import config
        assert 0.0 < config.SPINE_UPRIGHT_THRESHOLD < 1.0


class TestConfigEnvOverride:
    """Verify that env variables can override config values."""

    def test_float_override(self, monkeypatch):
        monkeypatch.setenv("CALIBRATION_SECONDS", "7.5")
        # Force reimport to pick up env var
        import importlib
        import backend.cv_pipeline.config as cfg
        importlib.reload(cfg)
        assert cfg.CALIBRATION_SECONDS == 7.5
        # Restore
        monkeypatch.delenv("CALIBRATION_SECONDS", raising=False)
        importlib.reload(cfg)

    def test_int_override(self, monkeypatch):
        monkeypatch.setenv("PROFILE_MIN_SAMPLES", "60")
        import importlib
        import backend.cv_pipeline.config as cfg
        importlib.reload(cfg)
        assert cfg.PROFILE_MIN_SAMPLES == 60
        monkeypatch.delenv("PROFILE_MIN_SAMPLES", raising=False)
        importlib.reload(cfg)

    def test_str_override(self, monkeypatch):
        monkeypatch.setenv("PRIMARY_PERSON_STRATEGY", "test_strategy")
        import importlib
        import backend.cv_pipeline.config as cfg
        importlib.reload(cfg)
        assert cfg.PRIMARY_PERSON_STRATEGY == "test_strategy"
        monkeypatch.delenv("PRIMARY_PERSON_STRATEGY", raising=False)
        importlib.reload(cfg)


class TestSpeechConfig:
    """Verify speech backend config values."""

    def test_sample_rate_is_16000(self):
        from backend.speech_backend import config
        assert config.SAMPLE_RATE == 16000

    def test_channels_is_mono(self):
        from backend.speech_backend import config
        assert config.CHANNELS == 1

    def test_silence_duration_reasonable(self):
        from backend.speech_backend import config
        assert 0.1 <= config.SILENCE_DURATION <= 3.0

    def test_server_port_positive(self):
        from backend.speech_backend import config
        assert config.SERVER_PORT > 0
