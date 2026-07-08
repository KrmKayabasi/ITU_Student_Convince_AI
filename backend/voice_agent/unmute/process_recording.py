"""Process a .msgpack recording for displaying on the project page.

The recording is done by recorder.py, which just records the JSON messages sent back
and forth between the client and the server.

This script converts the recording into a time-aligned format that's easier for
visualization.

It can also extract the audio from the recording.
"""

import argparse
import base64
import json
import logging
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import torchaudio
if not hasattr(torchaudio, "AudioMetaData"):
    class DummyAudioMetaData:
        pass
    torchaudio.AudioMetaData = DummyAudioMetaData

import soundfile as sf
import torch

def _patched_torchaudio_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True):
    start = frame_offset
    stop = None if num_frames == -1 else (start + num_frames)
    data, samplerate = sf.read(filepath, start=start, stop=stop, dtype='float32', always_2d=True)
    tensor = torch.from_numpy(data)
    if channels_first:
        tensor = tensor.t()
    return tensor, samplerate

torchaudio.load = _patched_torchaudio_load

import torch
try:
    _original_load = torch.load
    def _patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)
    torch.load = _patched_load
except Exception:
    pass

import msgpack
import numpy as np
import sphn
from pydantic import BaseModel

from unmute import openai_realtime_api_events as ora
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.recorder import RecorderEvent
from unmute.tts.text_to_speech import prepare_text_for_tts

# Note this is not the ASR/TTS frame size; for that, see SAMPLES_PER_FRAME.
# It's the number of samples we get from the user per step.
SAMPLES_PER_STEP = 960
SAMPLES_PER_WAVEFORM = 240

logger = logging.getLogger(__name__)


class AudioFrame(BaseModel):
    amplitude_rms: list[float]
    n_samples: int
    created_at_samples: int
    speaker_id: int | None = None

    def split(self, n_samples_start: int) -> tuple["AudioFrame", "AudioFrame"]:
        assert 0 < n_samples_start < self.n_samples, (
            f"{self.n_samples=}, {n_samples_start=}"
        )

        fraction = n_samples_start / self.n_samples
        amplitude_index_float = fraction * len(self.amplitude_rms)
        amplitude_index = int(amplitude_index_float)
        if amplitude_index_float - amplitude_index > 0.1:
            logger.warning(
                "Amplitude RMS split unevenly, "
                f"{fraction=}, {amplitude_index_float=} {len(self.amplitude_rms)}."
            )

        return (
            AudioFrame(
                amplitude_rms=self.amplitude_rms[:amplitude_index],
                n_samples=n_samples_start,
                created_at_samples=self.created_at_samples,
                speaker_id=self.speaker_id,
            ),
            AudioFrame(
                amplitude_rms=self.amplitude_rms[amplitude_index:],
                n_samples=self.n_samples - n_samples_start,
                created_at_samples=self.created_at_samples,
                speaker_id=self.speaker_id,
            ),
        )


class TextFrame(BaseModel):
    text: str
    created_at_samples: int
    # Using 0 as "unknown" is a bit hacky but it makes addition simpler
    duration_samples: int = 0
    speaker_id: int | None = None



class AudioAndText(BaseModel):
    audio: AudioFrame | None = None
    text: TextFrame | None = None


class StepEvents(BaseModel):
    samples_since_start: int
    received: AudioAndText
    emitted: AudioAndText = AudioAndText(audio=None, text=None)
    other_events: list[ora.Event] = []


def get_audio_volume_rms(arr: np.ndarray) -> list[float]:
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / np.iinfo(np.int16).max

    if len(arr) % SAMPLES_PER_WAVEFORM != 0:
        raise ValueError(
            f"Array length {len(arr)} is not a multiple of SAMPLES_PER_WAVEFORM ({SAMPLES_PER_WAVEFORM})"
        )

    rms_list = []
    for i in range(0, len(arr), SAMPLES_PER_WAVEFORM):
        chunk = arr[i : i + SAMPLES_PER_WAVEFORM]
        rms = np.sqrt(np.mean(chunk**2))
        rms_list.append(rms)
    return rms_list


def round_to_multiple(value: float, multiple: int) -> int:
    """Round `value` to the nearest multiple of `multiple`."""
    return round(value / multiple) * multiple


def with_samples_since_start(
    recorder_events: list[RecorderEvent],
) -> Iterable[tuple[int, RecorderEvent]]:
    pass
    """Yield (timestamp_samples, recorder_event) pairs from the recorder events."""
    stream_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    samples_since_start = -SAMPLES_PER_STEP
    for recorder_event in recorder_events:
        ora_event = recorder_event.data

        if isinstance(ora_event, ora.InputAudioBufferAppend):
            audio_data = stream_reader.append_bytes(base64.b64decode(ora_event.audio))
            if not audio_data.size:
                logger.warning(
                    f"At {samples_since_start=}, received empty audio data. Skipping."
                )
                continue

            n = len(audio_data)
            if n != SAMPLES_PER_STEP:
                # Related to Opus. Seems to only happen at the beginning
                logger.warning(
                    f"At {samples_since_start=}, received audio data with {n} samples, "
                    f"expected {SAMPLES_PER_STEP} samples. Skipping."
                )
                continue

            samples_since_start += n
            yield samples_since_start, recorder_event
        else:
            yield samples_since_start, recorder_event


def process_events(recorder_events: list[RecorderEvent]) -> list[StepEvents]:
    step_events: dict[int, StepEvents] = {}
    # other_events for a given timestamp might be created before we've created the
    # corresponding step event, so collect them in a separate dict and then merge them
    other_events: defaultdict[int, list[ora.Event]] = defaultdict(list)

    # There are actually two levels of buffering, so use two queues
    tts_server_audio_queued: deque[AudioFrame] = deque()
    tts_client_audio_queued: deque[AudioFrame] = deque()

    tts_text_ready: deque[TextFrame] = deque()

    client_opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    server_opus_reader = sphn.OpusStreamReader(SAMPLE_RATE)

    for samples_since_start, recorder_event in with_samples_since_start(
        recorder_events
    ):
        recorder_event = deepcopy(recorder_event)
        ora_event = recorder_event.data

        if isinstance(ora_event, ora.InputAudioBufferAppend):
            # Received audio from the client
            audio_data = client_opus_reader.append_bytes(
                base64.b64decode(ora_event.audio)
            )
            if not audio_data.size:
                continue

            assert samples_since_start not in step_events

            n = len(audio_data)

            step_events[samples_since_start] = StepEvents(
                samples_since_start=samples_since_start,
                received=AudioAndText(
                    audio=AudioFrame(
                        amplitude_rms=get_audio_volume_rms(audio_data),
                        n_samples=n,
                        # For received audio, the creation time is the same as the
                        # receive time.
                        # We add SAMPLES_PER_STEP as a hack so that the waveform
                        # visualization of the received audio doesn't show parts of the
                        # audio as being created but not received yet (because the step
                        # is shown as multiple rectangles since amplitude_rms is a list)
                        created_at_samples=samples_since_start + SAMPLES_PER_STEP,
                    ),
                ),
                emitted=AudioAndText(audio=None, text=None),
            )

            if tts_client_audio_queued:
                audio = tts_client_audio_queued.popleft()
                if audio.n_samples == n:
                    step_events[samples_since_start].emitted.audio = audio
                elif audio.n_samples > n:
                    head, tail = audio.split(n)
                    step_events[samples_since_start].emitted.audio = head
                    tts_client_audio_queued.appendleft(tail)
                else:
                    raise RuntimeError(
                        "Unexpected: output audio frame size is not "
                        "a multiple of the input frame size. "
                        f"{n=}, {audio.n_samples=}"
                    )
        elif isinstance(ora_event, ora.UnmuteResponseAudioDeltaReady):
            tts_server_audio_queued.append(
                AudioFrame(
                    amplitude_rms=[],  # Will be set later
                    n_samples=ora_event.number_of_samples,
                    created_at_samples=samples_since_start,
                )
            )
        elif isinstance(ora_event, ora.ResponseAudioDelta):
            # The server emitted TTS audio that it queued up previously

            audio_data = server_opus_reader.append_bytes(
                base64.b64decode(ora_event.delta)
            )
            assert audio_data.size > 0, "Received empty audio delta"

            if not tts_server_audio_queued:
                # Not sure why this happens, maybe some off-by one? Something related
                # to Opus?
                logger.warning(
                    f"Received TTS audio delta at timestamp {samples_since_start} "
                    "but no audio frame was queued on the server side."
                )
                continue

            # Move from the server-side queue to the client-side queue
            assert tts_server_audio_queued[0].n_samples == len(audio_data)
            audio_frame = tts_server_audio_queued.popleft()
            audio_frame.amplitude_rms = get_audio_volume_rms(audio_data)
            tts_client_audio_queued.append(audio_frame)
        elif isinstance(ora_event, ora.UnmuteResponseTextDeltaReady):
            tts_text_ready.append(
                TextFrame(
                    text=ora_event.delta,
                    created_at_samples=samples_since_start,
                    duration_samples=0,  # We don't know yet
                )
            )
            other_events[samples_since_start].append(ora_event)
        elif isinstance(ora_event, ora.ResponseTextDelta):
            assert tts_text_ready
            prepared_text = tts_text_ready.popleft()
            assert ora_event.delta == prepare_text_for_tts(prepared_text.text), (
                f"Expected TTS text delta to be '{prepared_text.text}', "
                f"but got '{ora_event.delta}'"
            )
            step_events[samples_since_start].emitted.text = prepared_text
        elif isinstance(ora_event, ora.ConversationItemInputAudioTranscriptionDelta):
            # The STT transcribed something in the past, so we need to compute the
            # timestamp and retroactively add it to the existing step event
            ts_in_question = round_to_multiple(
                ora_event.start_time * SAMPLE_RATE, SAMPLES_PER_STEP
            )
            assert step_events[ts_in_question].received.text is None

            step_events[ts_in_question].received.text = TextFrame(
                text=ora_event.delta,
                created_at_samples=samples_since_start,
                duration_samples=0,  # We don't know
            )
        elif isinstance(ora_event, ora.ResponseCreated):
            # There might be text that the TTS queued up before but it got interrupted
            # before it could be emitted, so remove that text when we start generating
            # a new response.
            tts_text_ready.clear()
            other_events[samples_since_start].append(ora_event)
        else:
            ignored_event_types = [ora.UnmuteAdditionalOutputs]
            if not isinstance(ora_event, tuple(ignored_event_types)):
                other_events[samples_since_start].append(ora_event)

    # Merge other_events into step_events
    for samples_since_start, step_event in step_events.items():
        step_event.other_events = other_events[samples_since_start]

    step_events_list = list(step_events.values())
    step_events_list.sort(key=lambda x: x.samples_since_start)

    # Sanity checks
    samples_per_step = (
        step_events_list[1].samples_since_start
        - step_events_list[0].samples_since_start
    )
    for i, step_event in enumerate(step_events_list):
        assert step_event.samples_since_start == i * samples_per_step

    assert step_events_list[0].samples_since_start == 0

    return step_events_list


def slice_processed_events(
    processed_events: list[StepEvents], start_samples: int
) -> list[StepEvents]:
    filtered = [
        # Copy because we'll be modifying the events later
        deepcopy(event)
        for event in processed_events
        if start_samples <= event.samples_since_start
    ]

    # Fix the timestamps
    for event in filtered:
        event.samples_since_start -= start_samples
        if event.received.audio:
            event.received.audio.created_at_samples -= start_samples
        if event.received.text:
            event.received.text.created_at_samples -= start_samples
        if event.emitted.audio:
            event.emitted.audio.created_at_samples -= start_samples
        if event.emitted.text:
            event.emitted.text.created_at_samples -= start_samples

    return filtered


def extract_audios(
    recorder_events: list[RecorderEvent],
) -> np.ndarray:
    """Return a 2d NumPy array containing user and assistant audio.

    User audio is on the first channel, assistant audio is on the second channel.
    They are time-aligned and trimmed
    """
    user_pcm_chunks = []
    user_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    user_n_samples = 0

    assistant_pcm_chunks = []
    assistant_reader = sphn.OpusStreamReader(SAMPLE_RATE)
    assistant_n_samples = 0

    for e in recorder_events:
        if isinstance(e.data, ora.InputAudioBufferAppend):
            pcm = user_reader.append_bytes(base64.b64decode(e.data.audio))
            user_pcm_chunks.append(pcm)
            user_n_samples += len(pcm)

        elif isinstance(e.data, ora.ResponseAudioDelta):
            pcm = assistant_reader.append_bytes(base64.b64decode(e.data.delta))
            assistant_pcm_chunks.append(pcm)
            assistant_n_samples += len(pcm)

        # The assistant is not emitting audio all the time, so add silence so that the
        # lengths match
        if user_n_samples > assistant_n_samples:
            assistant_pcm_chunks.append(
                np.zeros(user_n_samples - assistant_n_samples, dtype=np.float32)
            )
            assistant_n_samples = user_n_samples

    user_audio = np.concatenate(user_pcm_chunks)
    assistant_audio = np.concatenate(assistant_pcm_chunks)
    length = max(len(user_audio), len(assistant_audio))

    def pad(audio: np.ndarray):
        """Pad the audio to the given length with zeros."""
        if len(audio) < length:
            return np.pad(audio, (0, length - len(audio)), mode="constant")
        return audio

    return np.array([pad(user_audio), pad(assistant_audio)])


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)

    duration = audio.size / src_rate
    dst_size = max(1, int(round(duration * dst_rate)))
    src_x = np.linspace(0.0, duration, audio.size, endpoint=False)
    dst_x = np.linspace(0.0, duration, dst_size, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def diarize_user_audio(user_audio: np.ndarray, processed_events: list[StepEvents]) -> list[StepEvents]:
    import torch
    import soundfile as sf
    import tempfile
    import os
    import contextlib
    import time

    logger.info("Initializing DiariZen speaker diarisation...")

    # Choose device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Using CUDA device for diarisation with TF32 enabled")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device for diarisation")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device for diarisation")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        from diarizen.pipelines.inference import DiariZenPipeline
    except ImportError as e:
        logger.error(
            "diarizen is not installed. Please make sure to install it using "
            "`pip install -e ./diarizen_src` and `pip install -e ./diarizen_src/pyannote-audio`"
        )
        raise e

    try:
        diar_pipeline = DiariZenPipeline.from_pretrained(
            "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
        )
        diar_pipeline = diar_pipeline.to(device)
    except Exception as e:
        logger.error(
            "Failed to load DiariZen model. If you haven't accepted the model terms "
            "at https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2, please do so. "
            "Also ensure HF_TOKEN or HUGGING_FACE_HUB_TOKEN is set in your environment."
        )
        raise e

    # Resample user audio from 24000 to 16000
    user_audio_16k = resample_linear(user_audio, SAMPLE_RATE, 16000)

    # Save to a temporary file
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_path = tmp_file.name
        sf.write(tmp_path, user_audio_16k, 16000)

        # Context manager for mixed precision / inference mode
        @contextlib.contextmanager
        def get_inference_context(dev: torch.device):
            if dev.type == "cuda":
                context = torch.autocast("cuda", dtype=torch.float16)
            elif dev.type == "mps":
                try:
                    context = torch.autocast("mps", dtype=torch.float16)
                except Exception:
                    context = contextlib.nullcontext()
            else:
                context = contextlib.nullcontext()

            with torch.inference_mode():
                with context:
                    yield

        logger.info("Running DiariZen speaker diarisation on user audio...")
        t0 = time.perf_counter()
        with get_inference_context(device):
            diar_results = diar_pipeline(tmp_path)
        logger.info(f"Diarisation finished in {time.perf_counter() - t0:.2f}s")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")

    # Build speaker map to map speaker labels to clean integers (0, 1, 2, 3)
    speaker_map = {}
    def get_speaker_id(label: str) -> int:
        if label not in speaker_map:
            speaker_map[label] = len(speaker_map)
        return speaker_map[label]

    # Convert results to speaker turns list
    speaker_turns = []
    for turn, _, speaker in diar_results.itertracks(yield_label=True):
        speaker_turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker_id": get_speaker_id(speaker),
            "speaker_label": speaker
        })

    logger.info(f"Detected {len(speaker_map)} user speaker(s): {list(speaker_map.keys())}")

    # Annotate processed events
    for event in processed_events:
        t = event.samples_since_start / SAMPLE_RATE

        # Check active speaker for this frame
        active_speaker = None
        for turn in speaker_turns:
            if turn["start"] <= t < turn["end"]:
                active_speaker = turn["speaker_id"]
                break

        if event.received.audio:
            event.received.audio.speaker_id = active_speaker

        if event.received.text:
            text_start_sec = t
            text_speaker = None
            for turn in speaker_turns:
                if turn["start"] <= text_start_sec < turn["end"]:
                    text_speaker = turn["speaker_id"]
                    break
            if text_speaker is None:
                text_speaker = active_speaker
            event.received.text.speaker_id = text_speaker

    return processed_events



def main(
    input_path: Path,
    output_path: Path,
    audio_output_path: Path | None,
    discard_first_assistant_message: bool = False,
    diarize: bool = False,
):
    with input_path.open("rb") as f:
        events_raw = msgpack.load(f)
        recorder_events = [RecorderEvent(**e) for e in events_raw]

    processed = process_events(recorder_events)

    if diarize:
        try:
            audio_data = extract_audios(recorder_events)
            user_audio = audio_data[0]
            processed = diarize_user_audio(user_audio, processed)
        except Exception:
            logger.exception("Failed to diarize user audio")

    slice_from_sample = 0
    if discard_first_assistant_message:
        user_speech_start = None
        for e in processed:
            if e.received.text is not None:
                user_speech_start = e
                break

        assert user_speech_start is not None, "No user speech found in the recording."

        padding_samples = SAMPLE_RATE * 0.2  # A bit arbitrary here
        slice_from_sample = user_speech_start.samples_since_start - int(padding_samples)

    if slice_from_sample > 0:
        processed = slice_processed_events(processed, slice_from_sample)

    with open(output_path, "w") as f:
        json.dump([e.model_dump() for e in processed], f, indent=2)
        len_sec = len(processed) * SAMPLES_PER_STEP / SAMPLE_RATE
        print(
            f"Saved processed recording with {len(processed)} steps ({len_sec:.1f}s) "
            f"to {output_path}"
        )

    if audio_output_path is not None:
        audio = extract_audios(recorder_events)
        audio = np.mean(audio, axis=0)  # Combine channels into one
        audio = audio[slice_from_sample:]

        sphn.write_opus(audio_output_path, audio, SAMPLE_RATE)
        print(
            f"Saved {len(audio) / SAMPLE_RATE:.1f}s of user and assistant audio "
            f"to {audio_output_path}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path", type=Path, help="The .msgpack file of the raw recording"
    )
    parser.add_argument(
        "output_path", type=Path, help="The path to which the output JSON will be saved"
    )
    parser.add_argument(
        "--discard-first-assistant-message",
        action="store_true",
    )
    parser.add_argument(
        "--audio-output-path",
        type=Path,
        help="Save the combined audio to this path. Supports .ogg and .wav.",
    )
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Run DiariZen speaker diarisation on user audio.",
    )
    args = parser.parse_args()

    main(
        args.input_path,
        args.output_path,
        args.audio_output_path,
        args.discard_first_assistant_message,
        args.diarize,
    )

