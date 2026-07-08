"""Esikler kod degistirmeden env ile ayarlanabiliyor (T12)."""
import importlib

from backend.cv_pipeline import config as config_module


def test_calibration_seconds_overridable_via_env(monkeypatch):
    monkeypatch.setenv("CALIBRATION_SECONDS", "9.5")
    importlib.reload(config_module)
    try:
        assert config_module.CALIBRATION_SECONDS == 9.5
    finally:
        monkeypatch.delenv("CALIBRATION_SECONDS", raising=False)
        importlib.reload(config_module)
        assert config_module.CALIBRATION_SECONDS == 3.0
