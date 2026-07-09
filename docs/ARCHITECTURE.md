# System Architecture - ITU Student Convince AI

This document provides a technical overview of the components, communication flows, and design decisions of the ITU Student Convince AI system.

---

## 🏛️ Component Overview

The system is designed around a **decoupled, low-latency native architecture** composed of three primary domains:

```
┌──────────────────────────────────────────────────────────┐
│                    PyQt6 Client Panel                    │
│   (Captures camera frame stream & local mic audio,       │
│    runs DiariZen and plays back Gemma 12B audio response)│
└───────────┬──────────────────────────────────┬───────────┘
            │ 1. Video Frames                  │ 3. Audio & Text Stream
            │ (JPEG via WebSocket)             │ (HTTP POST /chat_stream)
            ▼                                  ▼
┌──────────────────────────┐         ┌─────────────────────┐
│        FastAPI           │         │   Gemma 12B Host    │
│   CV Description Server  │         │    Speech Server    │
│  (MediaPipe & ONNX Eng.) │         │ (Whisper + VITS)    │
└──────────────────────────┘         └─────────────────────┘
```

### 1. Computer Vision (CV) Description Pipeline (`backend/cv_pipeline/`)
*   **Role**: Handles high-frequency real-time video processing to extract behavioral biometric features.
*   **Technologies**: FastAPI, Uvicorn, MediaPipe (Face & Pose Landmarkers), ONNX Runtime (Emotion classification model).
*   **Design**:
    *   **FrameSlot (Drop-Stale)**: Frame ingestion uses a single-slot buffer. If video frames arrive faster than the model processing thread runs, older frames are dropped immediately rather than queued, ensuring zero-latency real-time response.
    *   **Stateless Worker Threads**: Processing of landmarker frames is offloaded to worker threads so Uvicorn's async event loop remains non-blocked and highly responsive.

### 2. PyQt6 Desktop Client Panel (`client/`)
*   **Role**: The primary client and orchestrator.
*   **Technologies**: PyQt6, sounddevice, sherpa-onnx, diarizen, OpenCV, websockets.
*   **Design**:
    *   **Left Panel**: Captures video from the local webcam using OpenCV, compresses frames to JPEG, and streams them over a local WebSocket to the CV pipeline server. Displays bounding boxes and numerical metrics (focus time, posture baseline deviations, and active emotion confidence).
    *   **Right Panel**: Native chat bubble history displaying the text log of the conversation. Features distinct rounded chat bubble colors corresponding to different diarised user speakers.
    *   **Local Audio Capture & VAD**: Uses `sounddevice` to capture microphone input at 16000 Hz. Runs a local **Silero VAD** model (`sherpa-onnx`) to identify speech boundaries automatically and trigger voice submissions.
    *   **Offline Speaker Diarisation**: Once speech ends, the client runs the **DiariZen Pipeline** (utilizing WavLM and PyAnote audio) inside a background thread to identify the specific speaker ID (Speaker 0, Speaker 1, etc.).
    *   **Direct Server Playout**: Streams audio bytes from the host-native Gemma 12B server and plays them back using a background output stream at 24000 Hz.

### 3. Cascaded Speech Server (`Turkish_Speech_to_Speech`)
*   **Role**: Natural voice conversation interface.
*   **Technologies**: Transformers, MLX (Apple Silicon) / PyTorch CUDA, Piper VITS (Sherpa-onnx), OpenAI Whisper.
*   **Components**:
    *   **Automatic Speech Recognition (ASR)**: Whisper-tiny transcribes user speech.
    *   **Large Language Model (LLM)**: Gemma 4 12B generates replies.
    *   **Text-to-Speech (TTS)**: Piper VITS synthesizes voice output.

---

## 📡 Communication Protocols

1.  **Ingestion Stream**: Ingests webcam frames at `/stream/{session_id}` (binary JPEG over WebSocket).
2.  **Profile Stream**: Broadcasts a single, rich behavioral assessment JSON over `/profile/{session_id}` once calibration and scoring thresholds are reached.
3.  **Focus Stream**: Periodically pushes engagement metrics over `/focus/{session_id}` (every ~2.5 seconds).
4.  **Speech Stream**: PyQt6 client posts raw float32 PCM audio bytes to `/chat_stream` (HTTP POST) and streams back synthesized float32 PCM response audio bytes.
