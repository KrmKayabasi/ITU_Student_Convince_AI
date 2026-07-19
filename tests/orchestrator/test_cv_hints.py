"""Opening-hint formatting from a CV /profile payload (pure, no deps)."""

import cv_hints


def test_generic_opener_when_no_profile():
    hint = cv_hints.build_opening_hint(None)
    assert "selamla" in hint.lower()
    assert "öğrenci" in hint.lower()


def test_open_and_energetic_profile():
    profile = {
        "scores": {"attention": 0.8, "openness": 0.8, "energy": 0.8},
        "signals": {"emotion": {"dominant": "happy"}},
    }
    hint = cv_hints.build_opening_hint(profile)
    assert "açık ve rahat" in hint
    assert "enerjik" in hint
    assert "mutlu" in hint
    # Warm/energetic tone for a high openness+energy student.
    assert "enerjik ve samimi" in hint


def test_distracted_reserved_profile_uses_curiosity_opener():
    profile = {
        "scores": {"attention": 0.3, "openness": 0.3, "energy": 0.3},
        "signals": {"arms_crossed": {"value": True, "valid": True}},
    }
    hint = cv_hints.build_opening_hint(profile)
    assert "dikkati biraz dağınık" in hint
    assert "çekingen" in hint
    assert "kolları kapalı" in hint
    assert "merak uyandıran" in hint


def test_hint_never_reveals_observation_instruction():
    profile = {"scores": {"attention": 0.5, "openness": 0.5, "energy": 0.5}, "signals": {}}
    hint = cv_hints.build_opening_hint(profile)
    # The model is told not to state the observed traits aloud.
    assert "asla açıkça söyleme" in hint


def test_malformed_profile_does_not_crash():
    hint = cv_hints.build_opening_hint({"scores": {"attention": "oops"}, "signals": None})
    assert isinstance(hint, str) and hint
