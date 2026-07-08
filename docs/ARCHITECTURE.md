# System Architecture - ITU Student Convince AI

This document provides a technical overview of the components, communication flows, and design decisions of the ITU Student Convince AI system.

---

## 🏛️ Component Overview

The system is designed around a **decoupled, multi-service architecture** composed of three primary domains:

```
┌──────────────────────────────────────────────────────────┐
│                    PyQt6 Client Panel                    │
│   (Captures camera frame stream & embeds Web UI View)   │
└───────────┬──────────────────────────────────▲───────────┘
            │ 1. Video Frames                  │ 4. Scoring Metrics
            │ (JPEG via WebSocket)             │ (WebSocket /focus)
            ▼                                  │
┌──────────────────────────┐         ┌─────────┴───────────┐
│        FastAPI           │         │      FastAPI        │
│   CV Description Server  │         │ Voice Backend (8001)│
│  (MediaPipe & ONNX Eng.) │         │ (Moshi TTS / Gemma) │
└──────────────────────────┘         └─────────────────────┘
```

### 1. Computer Vision (CV) Description Pipeline (`backend/cv_pipeline/`)
*   **Role**: Handles high-frequency real-time video processing to extract behavioral biometric features.
*   **Technologies**: FastAPI, Uvicorn, MediaPipe (Face & Pose Landmarkers), ONNX Runtime (Emotion classification model).
*   **Design**:
    *   **FrameSlot (Drop-Stale)**: Frame ingestion uses a single-slot buffer. If video frames arrive faster than the model processing thread runs, older frames are dropped immediately rather than queued, ensuring zero-latency real-time response.
    *   **Stateless Worker Threads**: Processing of landmarker frames is offloaded to worker threads so Uvicorn's async event loop remains non-blocked and highly responsive.

### 2. PyQt6 Desktop Client Panel (`client/`)
*   **Role**: The primary kiosk client.
*   **Technologies**: PyQt6, QtWebEngineWidgets, OpenCV, websockets.
*   **Design**:
    *   **Left Panel**: Captures video from the local webcam using OpenCV, compresses frames to JPEG, and streams them over a local WebSocket to the CV pipeline server. Displays bounding boxes and numerical metrics (focus time, posture baseline deviations, and active emotion confidence).
    *   **Right Panel**: Embeds a Chromium-based Web engine (`QWebEngineView`) pointing to the voice agent Next.js frontend UI (`http://localhost:3000`), displaying the animated AI avatar.
    *   **Service Manager**: Runs as background `QProcess` tasks to launch and monitor the status of the Voice backend stack (LLM, STT, TTS, Backend, Frontend).

### 3. Voice Conversational Agent Stack (`backend/voice_agent/` & `frontend/`)
*   **Role**: Natural voice conversation interface.
*   **Technologies**: Moshi, Gemma 4 LLM, Next.js.
*   **Components**:
    *   **Speech-to-Text (STT)**: Locally hosts Whisper/Moshi STT on port 8090.
    *   **Text-to-Speech (TTS)**: Locally hosts Moshi TTS on port 8089.
    *   **Large Language Model (LLM)**: Locally hosts Gemma 4 E2B on vLLM (port 8091).
    *   **Frontend UI**: Interactive web dashboard running on port 3000.

---

## 📡 Communication Protocols

1.  **Ingestion Stream**: Ingests webcam frames at `/stream/{session_id}` (binary JPEG over WebSocket).
2.  **Profile Stream**: Broadcasts a single, rich behavioral assessment JSON over `/profile/{session_id}` once calibration and scoring thresholds are reached.
3.  **Focus Stream**: Periodically pushes engagement metrics over `/focus/{session_id}` (every ~2.5 seconds).
