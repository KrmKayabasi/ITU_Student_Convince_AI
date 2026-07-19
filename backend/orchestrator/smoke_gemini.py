"""
P1 live smoke test for the Gemini Live bridge.

Requires a real key:  GOOGLE_API_KEY=... python backend/orchestrator/smoke_gemini.py

It opens a native-audio session with affective dialog + proactive audio
(v1alpha), asks for a short Turkish greeting via injected context, and verifies
that (a) the v1alpha flags are accepted, (b) a Turkish voice speaks, and (c) we
receive 24 kHz PCM16 audio. The received audio is written to smoke_out.wav so
you can listen and confirm Turkish + voice quality.
"""

from __future__ import annotations

import asyncio
import os
import sys
import wave

sys.path.insert(0, os.path.dirname(__file__))

import config  # noqa: E402
from gemini_live_bridge import GeminiLiveBridge  # noqa: E402


async def main() -> int:
    if not config.GOOGLE_API_KEY:
        print("GOOGLE_API_KEY is not set — cannot run the live smoke test.")
        return 2

    bridge = GeminiLiveBridge(
        api_key=config.GOOGLE_API_KEY,
        model=config.GEMINI_LIVE_MODEL,
        voice=config.GEMINI_VOICE,
        instructions=config.SYSTEM_INSTRUCTION,
        api_version=config.GEMINI_API_VERSION,
        enable_affective_dialog=config.ENABLE_AFFECTIVE_DIALOG,
        enable_proactive_audio=config.ENABLE_PROACTIVE_AUDIO,
    )

    print(f"Connecting: model={config.GEMINI_LIVE_MODEL} voice={config.GEMINI_VOICE} "
          f"api={config.GEMINI_API_VERSION} affective={config.ENABLE_AFFECTIVE_DIALOG} "
          f"proactive={config.ENABLE_PROACTIVE_AUDIO}")
    await bridge.connect()
    print("Connected. Seeding a Turkish greeting request...")

    # Prompt an opening turn (turn_complete=True so the model responds now).
    await bridge.inject_context(
        "Bir öğrenci tanıtım standına yeni geldi. Onu çok kısa, sıcak ve "
        "enerjik bir cümleyle Türkçe selamla ve nasıl yardımcı olabileceğini sor.",
        turn_complete=True,
    )

    audio = bytearray()
    transcript_parts: list[str] = []
    try:
        async with asyncio.timeout(30):
            async for ev in bridge.receive():
                if ev["type"] == "audio":
                    audio.extend(ev["pcm16"])
                elif ev["type"] == "output_transcript":
                    transcript_parts.append(ev["text"])
                    print("  [assistant text]", ev["text"])
                elif ev["type"] == "turn_complete":
                    break
    except (asyncio.TimeoutError, TimeoutError):
        print("  (timed out waiting for turn_complete — using what we got)")
    finally:
        await bridge.aclose()

    print(f"Received {len(audio)} PCM16 bytes "
          f"(~{len(audio)/2/config.OUTPUT_SAMPLE_RATE:.2f}s @ {config.OUTPUT_SAMPLE_RATE}Hz)")
    if transcript_parts:
        print("Assistant said:", "".join(transcript_parts).strip())

    if audio:
        out = os.path.join(os.path.dirname(__file__), "smoke_out.wav")
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(config.OUTPUT_SAMPLE_RATE)
            w.writeframes(bytes(audio))
        print(f"Wrote {out} — play it to confirm Turkish speech.")
        return 0

    print("No audio received — check model/voice/flags.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
