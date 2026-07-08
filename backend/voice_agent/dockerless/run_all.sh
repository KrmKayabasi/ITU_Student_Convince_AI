#!/bin/bash

# Clean up function to terminate all child processes on exit (Ctrl+C)
cleanup() {
    echo ""
    echo "Stopping all Unmute services..."
    # Disable trap to prevent recursive loop
    trap - SIGINT SIGTERM EXIT
    # Terminate all background processes spawned by this script
    pkill -P $$ 2>/dev/null || true
    echo "All services stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Change to workspace root directory
cd "$(dirname "$0")/.."

echo "=========================================================="
echo " Starting all Unmute services concurrently (Dockerless)..."
echo "=========================================================="
echo ""

# Create a logs directory
mkdir -p logs

# Start each service in the background and redirect output to log files
echo "1. Starting LLM service (logging to logs/llm.log)..."
./dockerless/start_llm.sh > logs/llm.log 2>&1 &

echo "2. Starting Speech-to-Text (STT) service (logging to logs/stt.log)..."
./dockerless/start_stt.sh > logs/stt.log 2>&1 &

echo "3. Starting Text-to-Speech (TTS) service (logging to logs/tts.log)..."
./dockerless/start_tts.sh > logs/tts.log 2>&1 &

echo "4. Starting Backend service (logging to logs/backend.log)..."
./dockerless/start_backend.sh > logs/backend.log 2>&1 &

echo "5. Starting Frontend service (logging to logs/frontend.log)..."
./dockerless/start_frontend.sh > logs/frontend.log 2>&1 &

echo ""
echo "=========================================================="
echo " All services started in background!"
echo " Logs are saved in the 'logs/' directory."
echo " Unmute website will be accessible at http://localhost:3000"
echo " Press Ctrl+C to stop all services."
echo "=========================================================="
echo ""

# Keep running and tail the logs so the output is shown in terminal
tail -f logs/frontend.log logs/backend.log logs/stt.log logs/tts.log logs/llm.log
