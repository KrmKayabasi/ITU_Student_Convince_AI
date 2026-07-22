import asyncio
import os
import sys
from functools import partial
from logging import getLogger
import numpy as np
import httpx
import sherpa_onnx

from fastrtc import (
    AdditionalOutputs,
    AsyncStreamHandler,
    CloseStream,
    audio_to_float32,
    wait_for_item,
)
from pydantic import BaseModel
import unmute.openai_realtime_api_events as ora
from unmute.kyutai_constants import SAMPLE_RATE
from unmute.quest_manager import Quest, QuestManager

logger = getLogger(__name__)

HandlerOutput = (
    tuple[int, np.ndarray] | AdditionalOutputs | ora.ServerEvent | CloseStream
)

class GradioUpdate(BaseModel):
    chat_history: list[dict[str, str]]
    debug_dict: dict
    debug_plot_data: list[dict]

class UnmuteHandler(AsyncStreamHandler):
    def __init__(self) -> None:
        super().__init__(
            input_sample_rate=SAMPLE_RATE,  # 24000
            output_frame_size=480,
            output_sample_rate=SAMPLE_RATE,  # 24000
        )
        self.n_samples_received = 0
        self.output_queue: asyncio.Queue[HandlerOutput] = asyncio.Queue()
        self.quest_manager = QuestManager()
        self.recorder = None
        
        self.user_audio_buffer = []
        self.is_speech_started = False
        self.is_bot_speaking = False
        self.bot_interrupted = False
        self.chat_history_local = []
        
        # Configure cascaded server URL
        # Running in Docker: http://host.docker.internal:8002
        # Running natively: http://localhost:8002
        self.cascaded_url = os.environ.get("CASCADED_SERVER_URL", "http://host.docker.internal:8002" if os.path.exists("/.dockerenv") else "http://localhost:8002")
        logger.info(f"UnmuteHandler: Connecting to cascaded server at {self.cascaded_url}")
        
        # Load Silero VAD (24000 Hz, matching unmute sample rate!)
        vad_model_path = "/Users/baydogan/Documents/ComputerScience/Projects/Turkish_Speech_to_Speech/cascaded_architecture/silero_vad.onnx"
        if not os.path.exists(vad_model_path):
            # Fallback to current directory or parents
            for p in ["silero_vad.onnx", "../silero_vad.onnx", "../../silero_vad.onnx", "../../../silero_vad.onnx"]:
                if os.path.exists(p):
                    vad_model_path = p
                    break
        
        logger.info(f"UnmuteHandler: Loading Silero VAD model from {vad_model_path}")
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = vad_model_path
        vad_config.sample_rate = SAMPLE_RATE # 24000
        vad_config.silero_vad.threshold = 0.45
        vad_config.silero_vad.min_silence_duration = 0.8
        vad_config.silero_vad.min_speech_duration = 0.3
        self.vad_detector = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

    async def cleanup(self):
        pass

    async def start_up(self):
        # Reset the conversation state in the cascaded server on startup
        logger.info("UnmuteHandler: Resetting conversation state on cascaded server...")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{self.cascaded_url}/reset")
        except Exception as e:
            logger.error(f"UnmuteHandler: Failed to reset cascaded server: {e}")

    async def update_session(self, session: ora.SessionConfig):
        # Allow the client to configure instructions/voice if needed
        logger.info(f"UnmuteHandler: Session updated (voice={session.voice})")

    def resample_audio(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        if orig_sr == target_sr:
            return audio
        num_samples = int(len(audio) * target_sr / orig_sr)
        return np.interp(
            np.linspace(0, len(audio), num_samples, endpoint=False),
            np.arange(len(audio)),
            audio
        ).astype(np.float32)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        sr = frame[0]
        assert sr == self.input_sample_rate
        assert frame[1].shape[0] == 1  # Mono
        array = frame[1][0]
        self.n_samples_received += array.shape[0]
        
        float_audio = audio_to_float32(array)
        
        # Stream into Silero VAD
        self.vad_detector.accept_waveform(float_audio)
        
        if self.vad_detector.is_speech_detected():
            if not self.is_speech_started:
                self.is_speech_started = True
                logger.info("Local VAD: Speech detected!")
                await self.output_queue.put(ora.InputAudioBufferSpeechStarted())
                
            if self.is_bot_speaking:
                logger.info("Local VAD: User interrupted the assistant!")
                await self.interrupt_bot()
                
            self.user_audio_buffer.append(float_audio)
        else:
            if self.is_speech_started:
                if not self.vad_detector.empty():
                    segment = self.vad_detector.front
                    self.vad_detector.pop()
                    
                    if self.user_audio_buffer:
                        full_audio = np.concatenate(self.user_audio_buffer)
                    else:
                        full_audio = np.array(segment.samples, dtype=np.float32)
                    
                    self.user_audio_buffer = []
                    self.is_speech_started = False
                    logger.info(f"Local VAD: User stopped speaking ({len(full_audio)/SAMPLE_RATE:.2f}s). Generating response...")
                    await self.output_queue.put(ora.InputAudioBufferSpeechStopped())
                    
                    # Spawn the response task in the background
                    await self._generate_response(full_audio)
                else:
                    self.user_audio_buffer.append(float_audio)

    async def _generate_response(self, audio_data: np.ndarray):
        await self.output_queue.put(
            ora.ResponseCreated(
                response=ora.Response(
                    status="in_progress",
                    voice="turkish_dfki",
                    chat_history=self.chat_history_local,
                )
            )
        )
        quest = Quest.from_run_step("llm", partial(self._generate_response_task, audio_data))
        await self.quest_manager.add(quest)

    async def _generate_response_task(self, audio_data: np.ndarray):
        self.is_bot_speaking = True
        self.bot_interrupted = False
        
        # 1. Downsample audio to 16000 Hz for Whisper/Gemma
        audio_16k = self.resample_audio(audio_data, SAMPLE_RATE, 16000)
        
        # 2. POST to cascaded speech-to-speech server
        try:
            logger.info("UnmuteHandler: Streaming audio to cascaded server...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.cascaded_url}/chat_stream",
                    content=audio_16k.tobytes()
                ) as response:
                    if response.status_code != 200:
                        logger.error(f"UnmuteHandler: Cascaded server returned {response.status_code}")
                        return
                    
                    server_sr = int(response.headers.get("X-Sample-Rate", "24000"))
                    
                    # Yield synthesized float32 PCM blocks (each float is 4 bytes)
                    # Use chunk size matching ~40ms frames for smooth playback
                    async for chunk in response.iter_bytes(chunk_size=960 * 4):
                        if self.bot_interrupted:
                            logger.info("UnmuteHandler: Streaming aborted due to interruption.")
                            break
                        
                        audio_chunk = np.frombuffer(chunk, dtype=np.float32)
                        if len(audio_chunk) > 0:
                            if server_sr != SAMPLE_RATE:
                                audio_chunk = self.resample_audio(audio_chunk, server_sr, SAMPLE_RATE)
                            
                            # Send output block to FastRTC queue
                            await self.output_queue.put((SAMPLE_RATE, audio_chunk))
                            
            # 3. Retrieve texts of this turn from `/last_turn`
            if not self.bot_interrupted:
                logger.info("UnmuteHandler: Fetching texts of the turn...")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    txt_res = await client.get(f"{self.cascaded_url}/last_turn")
                    if txt_res.status_code == 200:
                        data = txt_res.json()
                        user_text = data.get("user", "").strip()
                        bot_text = data.get("assistant", "").strip()
                        
                        logger.info(f"UnmuteHandler Turn: User='{user_text}' | Assistant='{bot_text}'")
                        
                        if user_text:
                            await self.output_queue.put(
                                ora.ConversationItemInputAudioTranscriptionDelta(
                                    delta=user_text,
                                    start_time=0.0
                                )
                            )
                            self.chat_history_local.append({"role": "user", "content": user_text})
                            
                        if bot_text:
                            await self.output_queue.put(ora.ResponseTextDelta(delta=bot_text))
                            await self.output_queue.put(ora.ResponseTextDone(text=bot_text))
                            self.chat_history_local.append({"role": "assistant", "content": bot_text})
                            
            await self.output_queue.put(ora.ResponseAudioDone())
            
        except Exception as e:
            logger.error(f"UnmuteHandler Error: {e}")
        finally:
            self.is_bot_speaking = False

    async def interrupt_bot(self):
        self.bot_interrupted = True
        self.is_bot_speaking = False
        
        # Clear out queue
        self.output_queue = asyncio.Queue()
        
        # Emit interruption event to WebSocket
        await self.output_queue.put(ora.UnmuteInterruptedByVAD())
        
        # Cancel any active quests/generation tasks
        await self.quest_manager.remove("llm")
        logger.info("UnmuteHandler: Assistant execution interrupted and cleared.")

    async def emit(self) -> HandlerOutput | None:
        item = await wait_for_item(self.output_queue)
        if item is not None:
            return item
        return None

    def copy(self):
        return UnmuteHandler()

    async def __aenter__(self) -> None:
        await self.quest_manager.__aenter__()

    async def __aexit__(self, *exc: Any) -> None:
        return await self.quest_manager.__aexit__(*exc)
