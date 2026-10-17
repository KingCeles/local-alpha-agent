#!/bin/bash
cd ~/local-alpha-agent

# 1. Start Ollama in background if it is not already running
if ! pgrep -x "ollama" > /dev/null; then
    echo "[*] Starting Ollama GPU server..."
    nohup ollama serve > /dev/null 2>&1 &
    sleep 2
else
    echo "[+] Ollama GPU server is already running."
fi

# 2. Activate virtual environment
source venv/bin/activate

# 3. Launch Streamlit (headless so Windows can open it in standalone App Mode)
echo "[*] Launching Financial RAG Terminal on port 8501..."
streamlit run rag_dashboard.py --server.headless true --server.port 8501
