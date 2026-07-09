# System Architecture - ITU Student Convince AI

This document provides a technical overview of the components, communication flows, and design decisions of the ITU Student Convince AI system.

---

## 🏛️ Component Overview

The system is designed around a **high-performance client-server architecture** separating lightweight, low-overhead native processing on the client device from heavy LLM/ASR compute on the remote server:

```
┌──────────────────────────────────────────────────────────┐
│             PyQt6 Client Device (e.g. Mac)               │
│   (Webcam frame capture & Local Image Processing,        │
│    Rust-optimized Silero VAD, Local Speaker Diarisation) │
└───────────┬──────────────────────────────────┬───────────┘
            │ 1. Video Frames                  │ 3. Audio & Text Stream
            │ (JPEG via WebSocket)             │ (HTTP POST /chat_stream)
            ▼                                  ▼
┌──────────────────────────┐         ┌─────────────────────┐
│    FastAPI CV Container  │         │   Dual NVIDIA H200  │
│   (Local Client Device)  │         │  Speech Server (8002)│
└──────────────────────────┘         └─────────────────────┘
```

### 1. Computer Vision (CV) Description Pipeline (`backend/cv_pipeline/`)
*   **Role**: Handles real-time video processing natively on the local client device to minimize latency and bandwidth.
*   **Technologies**: FastAPI, Uvicorn, MediaPipe (Face & Pose Landmarkers), ONNX Runtime (Emotion classification).
*   **Design**:
    *   Runs as a container directly on the local client device. MediaPipe and ONNX use compiled C/C++ backends under Python to bypass the Python Global Interpreter Lock (GIL) overhead.

### 2. PyQt6 Desktop Client Panel (`client/`)
*   **Role**: Client-side recording, VAD boundaries, local speaker diarisation, and user interface.
*   **Technologies**: PyQt6, sounddevice, sherpa-onnx, diarizen.
*   **Rust-Optimized Native Performance**:
    *   **Silero VAD**: Uses `sherpa-onnx`'s Voice Activity Detector which compiles directly into a native C++ runtime (with Rust/C bindings), running silence checks with virtually zero Python GIL or interpreter loop overhead.
    *   **PortAudio Engine**: Audio capture and playout streams run under native compiled PortAudio threads via `sounddevice` to enforce hardware-aligned buffer sizes (16kHz input / 24kHz output).
    *   **Local Diarisation**: The client runs speaker diarisation (`diarizen`) locally on the user's machine to offload speaker-identity processing from the remote GPU server.

### 3. Dedicated Cascaded Speech Server (`backend/speech_backend/`)
*   **Role**: High-power speech synthesis and language models.
*   **Technologies**: Transformers, PyTorch, CUDA, Docker.
*   **Design**:
    *   Deployed on a remote server equipped with **2 x NVIDIA H200 GPUs** (sharing 282 GB HBM3e memory).
    *   Loads unquantized **Gemma 4 12B** (in full `bfloat16` precision) and automatically distributes layers across both H200 cards using PyTorch `device_map="auto"`.
    *   Loads **Whisper Large v3 Turbo** and **Piper VITS** on CUDA for sub-100ms transcription and synthesis.

---

## 📡 Communication Protocols

1.  **Ingestion Stream**: Ingests webcam frames at `/stream/{session_id}` (binary JPEG over WebSocket).
2.  **Profile Stream**: Broadcasts a single, rich behavioral assessment JSON over `/profile/{session_id}` once calibration and scoring thresholds are reached.
3.  **Focus Stream**: Periodically pushes engagement metrics over `/focus/{session_id}` (every ~2.5 seconds).
4.  **Speech Stream**: PyQt6 client posts raw float32 PCM audio bytes to `/chat_stream` (HTTP POST) and streams back synthesized float32 PCM response audio bytes.
