import os
import numpy as np
import pytest
from unmute.process_recording import diarize_user_audio, StepEvents, AudioAndText

def test_diarize_user_audio():
    sample_rate = 24000
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    user_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    samples_per_step = 960
    num_steps = int((sample_rate * duration) / samples_per_step)
    
    processed_events = []
    for i in range(num_steps):
        samples_since_start = i * samples_per_step
        processed_events.append(
            StepEvents(
                samples_since_start=samples_since_start,
                received=AudioAndText(audio=None, text=None),
                emitted=AudioAndText(audio=None, text=None)
            )
        )

    try:
        updated_events = diarize_user_audio(user_audio, processed_events)
        assert len(updated_events) == num_steps
    except Exception as e:
        # If the model gated terms are not accepted or download failed, skip/fail gracefully
        if "gated" in str(e).lower() or "forbidden" in str(e).lower() or "unauthorized" in str(e).lower():
            pytest.skip(f"Gated model access issue: {e}")
        else:
            raise e
