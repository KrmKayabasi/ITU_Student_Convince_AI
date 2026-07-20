import numpy as np
import sounddevice as sd
import time
import queue
import scipy.io.wavfile
import io
import os
import urllib.request
import ssl
import hashlib
import sherpa_onnx
import select
import sys
from pywebrtc_audio import AudioProcessor

# Known SHA-256 hash for silero_vad.onnx to verify model integrity.
_SILERO_VAD_SHA256 = "9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6"

class AudioHandler:
    def __init__(self, sample_rate=16000, channels=1, energy_threshold=1.5, silence_duration=0.8, min_speech_duration=0.3, interruption_mode="both"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.interruption_mode = interruption_mode
        self.audio_queue = queue.Queue()
        self.baseline_energy = 0.005  # Default low baseline

        # Lazy initialization of WebRTC Audio Processor for thread safety
        self.ap = None

        # Initialize Silero VAD via sherpa-onnx (ONNX Runtime, 100% offline)
        model_path = "silero_vad.onnx"
        if not os.path.exists(model_path):
            print("[VAD] silero_vad.onnx model dosyası indiriliyor...", flush=True)
            url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(model_path, 'wb') as out_file:
                data = response.read()
                out_file.write(data)
            # Verify integrity of the downloaded model
            actual_hash = hashlib.sha256(data).hexdigest()
            expected_hash = _SILERO_VAD_SHA256
            if actual_hash != expected_hash:
                print(f"[VAD WARNING] Model hash mismatch — expected {expected_hash[:16]}..., got {actual_hash[:16]}...", flush=True)
            print("[VAD] Model dosyası başarıyla indirildi.", flush=True)
            
        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = model_path
        config.sample_rate = self.sample_rate
        config.silero_vad.threshold = 0.65  # Raised for noisy environments
        config.silero_vad.min_silence_duration = self.silence_duration
        config.silero_vad.min_speech_duration = self.min_speech_duration
        
        # 60 seconds circular buffer size for standard VAD (prevents overflow on long questions)
        self.vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=60)

        # Playback VAD (used when assistant is speaking, set higher to ignore residual echo)
        config_playback = sherpa_onnx.VadModelConfig()
        config_playback.silero_vad.model = model_path
        config_playback.sample_rate = self.sample_rate
        config_playback.silero_vad.threshold = 0.85  # Raised for noisy environments
        config_playback.silero_vad.min_silence_duration = self.silence_duration
        config_playback.silero_vad.min_speech_duration = self.min_speech_duration
        self.vad_playback = sherpa_onnx.VoiceActivityDetector(config_playback, buffer_size_in_seconds=60)

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for sounddevice input stream (applies WebRTC Noise Suppression and AGC)."""
        near_samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        far_samples = np.zeros(frames, dtype=np.float32)
        clean_samples = self.ap.process(near_samples, far_samples)
        self.audio_queue.put(clean_samples.copy())

    def calibrate(self, duration=1.5):
        """Calibrate the background noise to set a dynamic energy threshold."""
        try:
            input_device = sd.query_devices(kind='input')
            device_name = input_device.get('name', 'Bilinmeyen Cihaz')
            print(f"[Audio] Aktif Giriş Cihazı: {device_name}", flush=True)
        except Exception as e:
            print(f"[Audio Warning] Giriş cihazı sorgulanamadı: {e}", flush=True)

        print(f"[Audio] Ortam gürültüsü kalibre ediliyor ({duration} saniye)...", flush=True)
        q = queue.Queue()
        
        def calibration_callback(indata, frames, time_info, status):
            q.put(indata.copy())

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=calibration_callback
        )
        
        energies = []
        with stream:
            start_time = time.time()
            while time.time() - start_time < duration:
                try:
                    chunk = q.get(timeout=0.1)
                    rms = np.sqrt(np.mean(chunk**2))
                    energies.append(rms)
                except queue.Empty:
                    continue
        
        if energies:
            self.baseline_energy = np.mean(energies)
            self.baseline_energy = max(self.baseline_energy, 0.0001)
        
        print(f"[Audio] Kalibrasyon tamamlandı. Eşik Değeri: {self.baseline_energy * self.energy_threshold:.6f}", flush=True)

    def listen_until_silence(self):
        """
        Listens to the microphone stream.
        Uses Silero VAD (via sherpa-onnx) to detect speech segments and returns
        the active speech waveform once the user stops speaking.
        Prevents voice onset clipping by maintaining a 400ms pre-roll buffer.
        """
        from collections import deque
        pre_roll = deque(maxlen=25)  # 25 frames * 20ms = 500ms pre-roll buffer (wider for noisy envs)
        
        self.vad.reset()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Create fresh AudioProcessor for clean filter state and thread safety
        self.ap = AudioProcessor(
            sample_rate=self.sample_rate,
            noise_suppression=True,
            echo_cancellation=True,
            auto_gain_control=True,
            stream_delay_ms=50
        )

        # Set blocksize=320 for consistent 20ms frames matching WebRTC boundaries
        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=320
        )

        print("\n[Dinleniyor] Konuşmaya başlayın (Eller Serbest)...", flush=True)
        
        with stream:
            while True:
                try:
                    chunk_data = self.audio_queue.get(timeout=0.1)
                    if isinstance(chunk_data, tuple):
                        chunk, _, _ = chunk_data
                    else:
                        chunk = chunk_data
                except queue.Empty:
                    continue

                samples = chunk.flatten().astype(np.float32)
                self.vad.accept_waveform(samples)

                if self.vad.is_speech_detected():
                    print("\r[Konuşma Algılandı] Konuşun...                                ", end="", flush=True)
                else:
                    # Buffer pre-speech frames only when no speech is active
                    pre_roll.append(samples)
                    print("\r[Dinleniyor] Sessizlik / Ortam Gürültüsü Dinleniyor...     ", end="", flush=True)

                if not self.vad.empty():
                    segment = self.vad.front
                    detected_samples = np.array(segment.samples, dtype=np.float32)
                    self.vad.pop()
                    
                    # Prepend pre-roll buffer to prevent cutting off the start of speech
                    if len(pre_roll) > 0:
                        pre_roll_samples = np.concatenate(list(pre_roll))
                        audio_data = np.concatenate([pre_roll_samples, detected_samples])
                        print(f"\n[Konuşma Bitti] İşleniyor (Ön-ses tamponu {len(pre_roll_samples)/16000:.2f}s eklendi)...", flush=True)
                    else:
                        audio_data = detected_samples
                        print("\n[Konuşma Bitti] İşleniyor...", flush=True)
                        
                    return audio_data

    def listen_push_to_talk(self):
        """
        Records audio from the microphone when the user presses Enter, 
        and stops when they press Enter again (Manual Push-to-Talk).
        """
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Create fresh AudioProcessor
        self.ap = AudioProcessor(
            sample_rate=self.sample_rate,
            noise_suppression=True,
            echo_cancellation=True,
            auto_gain_control=True,
            stream_delay_ms=50
        )

        input("\n[Hazır] Konuşmayı başlatmak için [ENTER] tuşuna basın...")
        
        recorded_chunks = []
        
        def record_callback(indata, frames, time_info, status):
            near_samples = indata[:, 0].astype(np.float32)
            far_samples = np.zeros(frames, dtype=np.float32)
            clean_samples = self.ap.process(near_samples, far_samples)
            recorded_chunks.append(clean_samples.reshape(-1, 1))

        stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=record_callback
        )
        
        print("[Kaydediliyor] Konuşun... Bitirmek ve yanıt almak için [ENTER] tuşuna basın.", end="", flush=True)
        
        with stream:
            input()  # Block until the user presses Enter again
            
        print("\n[Konuşma Bitti] İşleniyor...", flush=True)
        
        if not recorded_chunks:
            return np.zeros(0, dtype=np.float32)
            
        audio_data = np.concatenate(recorded_chunks, axis=0).flatten()
        return audio_data

    def play_audio(self, audio_data, sample_rate=16000):
        """
        Plays audio and monitors for user interruption using an acoustic envelope gate.
        Uses WebRTC Noise Suppression and AGC to keep the signal clean, and prevents
        self-interruption by comparing mic energy against a 400ms sliding window
        of speaker playback energy.
        
        Returns:
            (interrupted, user_audio_data)
        """
        # Resample incoming audio if its sample rate doesn't match the stream's sample rate (16kHz)
        if sample_rate != self.sample_rate and len(audio_data) > 0:
            duration = len(audio_data) / sample_rate
            dst_size = max(1, int(round(duration * self.sample_rate)))
            src_x = np.linspace(0.0, duration, len(audio_data), endpoint=False)
            dst_x = np.linspace(0.0, duration, dst_size, endpoint=False)
            audio_data = np.interp(dst_x, src_x, audio_data).astype(np.float32)

        playback_idx = 0
        audio_len = len(audio_data)
        mute_playback = False

        # Enable WebRTC Acoustic Echo Cancellation (AEC), Noise Suppression (NS), and Automatic Gain Control (AGC)
        self.ap = AudioProcessor(
            sample_rate=self.sample_rate,
            noise_suppression=True,
            echo_cancellation=True,
            auto_gain_control=True,
            stream_delay_ms=0
        )

        # Clear VAD states and queue
        self.vad.reset()
        self.vad_playback.reset()
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        def duplex_callback(indata, outdata, frames, time_info, status):
            nonlocal playback_idx
            # 1. Fill outdata with playback samples (or silence if muted)
            far_samples = np.zeros(frames, dtype=np.float32)
            if not (mute_playback or playback_idx >= audio_len):
                remainder = audio_len - playback_idx
                if remainder >= frames:
                    far_samples[:] = audio_data[playback_idx : playback_idx + frames]
                    playback_idx += frames
                else:
                    far_samples[:remainder] = audio_data[playback_idx:]
                    playback_idx += remainder

            outdata[:, 0] = far_samples

            # 2. Get microphone samples
            near_samples = np.ascontiguousarray(indata[:, 0], dtype=np.float32)

            # 3. Apply WebRTC Noise Suppression & AGC
            clean_samples = self.ap.process(near_samples, far_samples)

            # Calculate RMS energy of cleaned microphone signal and raw playback signal
            rms_clean = float(np.sqrt(np.mean(clean_samples**2)))
            rms_far = float(np.sqrt(np.mean(far_samples**2)))

            self.audio_queue.put((clean_samples.copy(), rms_clean, rms_far))

        stream = sd.Stream(
            samplerate=self.sample_rate,
            channels=(self.channels, 1),
            callback=duplex_callback,
            blocksize=320
        )

        # Query actual hardware latency and set WebRTC stream delay dynamically.
        in_lat, out_lat = stream.latency
        total_latency_ms = int((in_lat + out_lat) * 1000)
        self.ap.stream_delay_ms = total_latency_ms
        print(f"\n[AEC] Donanımsal Akış Gecikmesi Hizalandı: {total_latency_ms} ms", flush=True)

        interrupted = False
        consecutive_speech_frames = 0
        start_time = time.time()

        # Sliding window history of speaker (far-end) RMS values
        # 20 frames of 320 samples at 16kHz corresponds to ~400ms of history,
        # which safely covers physical loopback delay and room reverberation.
        far_history = []
        max_history_len = 20

        # Calibration parameters
        calibration_near_values = []
        calibration_far_values = []
        alpha = 0.25  # Safe default coupling ratio
        calibrated = False

        with stream:
            while playback_idx < audio_len:
                # A. Keyboard Interruption Check
                if self.interruption_mode in ("both", "key_only"):
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
                    if rlist:
                        sys.stdin.readline()
                        print("\n[Klavye] Enter tuşuna basıldı! Asistan susturuluyor...", flush=True)
                        mute_playback = True
                        interrupted = True
                        self.vad.reset()
                        break

                try:
                    chunk_data = self.audio_queue.get(timeout=0.05)
                    if isinstance(chunk_data, tuple):
                        chunk, rms_clean, rms_far = chunk_data
                    else:
                        chunk = chunk_data
                        rms_clean = float(np.sqrt(np.mean(chunk**2)))
                        rms_far = 0.0
                except queue.Empty:
                    continue

                samples = chunk.flatten().astype(np.float32)
                elapsed = time.time() - start_time

                # Track speaker energy history
                far_history.append(rms_far)
                if len(far_history) > max_history_len:
                    far_history.pop(0)

                max_far_energy = max(far_history) if far_history else 0.0

                # B. Voice Interruption Check
                if self.interruption_mode in ("both", "vad_only"):

                    if elapsed <= 0.8:
                        # CALIBRATION PHASE (first 800ms):
                        # Store energies to perform peak-to-peak calibration later
                        calibration_near_values.append(rms_clean)
                        calibration_far_values.append(rms_far)
                        consecutive_speech_frames = 0
                    else:
                        # DETECTION PHASE:
                        if not calibrated:
                            if calibration_near_values and calibration_far_values:
                                max_near = max(calibration_near_values)
                                max_far = max(calibration_far_values)
                                if max_far > 0.015:
                                    # Ratio of peak energies with 50% safety margin, capped at 0.60
                                    alpha = min(0.60, (max_near / max_far) * 1.5)
                                else:
                                    alpha = 0.25
                            else:
                                alpha = 0.25
                            calibrated = True
                            print(f"[AEC] Eko Eşik Kat sayısı Kalibre Edildi: {alpha:.3f}", flush=True)

                        # Acoustic Envelope Gate: Only evaluate VAD if mic energy exceeds
                        # the expected echo ceiling of the speaker in the last 400ms.
                        expected_echo_ceiling = alpha * max_far_energy
                        gate_open = rms_clean > (expected_echo_ceiling + 0.015)  # Tighter gate for noisy envs

                        if gate_open:
                            self.vad_playback.accept_waveform(samples)
                            if self.vad_playback.is_speech_detected():
                                consecutive_speech_frames += 1
                            else:
                                consecutive_speech_frames = 0
                        else:
                            consecutive_speech_frames = 0

                    # Require 6 consecutive active frames (~192ms) to confirm user speech in noisy environments
                    if consecutive_speech_frames >= 6:
                        print("\n[VAD] Söz kesme doğrulandı! Asistan susturuluyor...", flush=True)
                        mute_playback = True
                        interrupted = True
                        self.vad.reset()
                        self.vad.accept_waveform(samples)
                        break

            # 2. Recording continuation phase (if interrupted)
            if interrupted:
                print("[VAD] Konuşmanın devamı dinleniyor...", flush=True)
                start_wait = time.time()
                max_speech_limit = 15.0
                while time.time() - start_wait < max_speech_limit:
                    try:
                        chunk_data = self.audio_queue.get(timeout=0.1)
                        if isinstance(chunk_data, tuple):
                            chunk, _, _ = chunk_data
                        else:
                            chunk = chunk_data
                    except queue.Empty:
                        continue

                    samples = chunk.flatten().astype(np.float32)
                    self.vad.accept_waveform(samples)

                    if self.vad.is_speech_detected():
                        max_speech_limit = min(25.0, max_speech_limit + 0.1)

                    if not self.vad.empty():
                        segment = self.vad.front
                        user_audio = np.array(segment.samples, dtype=np.float32)
                        self.vad.pop()
                        return True, user_audio

                print("[VAD] Konuşma tamamlanamadı veya limit aşıldı. Dinlemeye geçiliyor.", flush=True)
                return False, None

        # Playback completed naturally without interruption
        return False, None

    def close(self):
        """Release audio resources: clear queues and reset VAD state."""
        # Drain audio queue to unblock any waiting consumers
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        # Reset VAD detectors to release internal buffers
        try:
            self.vad.reset()
            self.vad_playback.reset()
        except Exception:
            pass
        # Release WebRTC audio processor if active
        self.ap = None

