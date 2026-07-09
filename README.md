# İTÜ AI Tercih Danışmanı - Student Convince AI

Welcome to the **İTÜ AI Preference Advisor**. This application is an interactive desktop assistant designed to help you prepare for advisor interviews. It combines real-time **Computer Vision** (tracking your camera feed for focus, eye contact, and posture) with a **Gemma 12B Voice Assistant** that hears you, understands who is speaking using speaker diarisation, and speaks back in Turkish.

This guide explains how to set up and run the entire application, command by command, in simple terms.

---

## 🛠️ Step-by-Step Launch Instructions

Make sure **Docker Desktop** is open and running on your computer before starting.

### Step 1: Set Up Python Dependencies
Open your **Terminal** app (search for "Terminal" on your Mac) and run the following commands to create your local environment and install all packages automatically:

```bash
# 1. Navigate to the project directory
cd /Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI

# 2. Run the automated installer script
./scripts/setup_all.sh
```
*What this does:* This runs our installer script which sets up Python, installs libraries to read your camera feed, load the GUI window, and configure the audio processing models.

---

### Step 2: Start the Gemma 12B Speech Server
The voice assistant (integrated directly inside this repository under `backend/speech_backend/`) runs natively on your machine to access GPU acceleration (making it answer you instantly). Open a **new terminal tab or window** and run:

```bash
# 1. Navigate to the project folder
cd /Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI

# 2. Start the speech-to-speech server
./scripts/start_cascaded_speech_server.sh
```
*What to expect:* You will see messages showing that **Whisper Large v3 Turbo**, **Gemma 12B**, and the Turkish Voice Synthesizer are loading. Once finished, it will say `All models loaded and ready!` on port `8002`. Keep this terminal open in the background.

---

### Step 3: Start the Desktop Application
Open a **third terminal window** and run the main desktop application:

```bash
# 1. Navigate to the project folder
cd /Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI

# 2. Launch the desktop GUI
uv run python client/desktop_client.py
```
*What to expect:* A dark-themed application window will open:
*   **Left Panel**: Shows your webcam feed.
*   **Right Panel**: Shows the chat history with speech control buttons.
*   **Status Label**: Will say `Status: Idle` once the speaker diarisation model is loaded.

---

### Step 4: Turn on the Camera Scoring Pipeline (Docker)
Inside the left panel of the Desktop Application GUI, click the green **"Start CV Pipeline"** button.
*   *What this does:* This automatically starts a Docker container in the background to analyze your posture, attention level, and facial expressions from your camera stream.
*   *Verification:* The status indicator under the button will turn green and read `CV_PIPELINE: RUNNING`. You will see live scoring metrics (like focus level and emotion probabilities) appearing in the text box below your camera preview.

---

## 🎙️ How to Talk with the AI

1.  **Auto-Talk (VAD) Mode** (Default):
    *   Simply start speaking into your microphone.
    *   The app will say `Status: Listening...`.
    *   When you stop speaking, it will automatically say `Status: Diarizing speech...` to identify who was speaking, and then send it to the advisor model.
    *   The advisor's Turkish voice response will play through your speakers, and its reply will print as a chat bubble on the screen.
2.  **Speaker Diarisation Colors**:
    *   If multiple people speak in the room, the app uses its offline AI model to label the user bubbles with the exact speaker ID.
    *   **Speaker 0** turns appear in **Blue**, **Speaker 1** turns in **Teal**, and **Speaker 2** in **Purple**.
3.  **Manual Control**:
    *   If you want to control when it records manually, uncheck **Auto-Talk (VAD)**.
    *   Click and hold the **Hold to Talk** button while you speak, and release it when you are finished.
4.  **Interrupting**:
    *   If the AI is speaking and you want it to stop, click the orange **"Interrupt Playout"** button, or simply speak over it (in Auto-Talk mode).

---

## 📖 Further Reading
For detailed software architecture specifications, developer guides, and test instructions, refer to:
*   **[Installation Guide (docs/SETUP.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/SETUP.md)**
*   **[System Architecture (docs/ARCHITECTURE.md)](file:///Users/baydogan/Documents/ComputerScience/Projects/ITU_Student_Convince_AI/docs/ARCHITECTURE.md)**
