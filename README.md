# A 100% local, privacy-first equity research and financial RAG (Retrieval-Augmented Generation) terminal built for consumer GPUs (NVIDIA RTX 3060). 

## Features
- **Full Financial Statement Ingestion**: Automatically pulls Annual & Quarterly Income Statements, Balance Sheets, and Cash Flow Statements via `yfinance` and formats them into compact CSV context blocks.
- **Deterministic 5-Year DCF Valuation**: Computes CAPM WACC, Present Value of Free Cash Flows, Terminal Value, and Intrinsic Value per share with interactive Streamlit sensitivity sliders.
- **Hybrid RAG Architecture**: Pins core annual accounting statements inside a 12k token context window while indexing chunked quarterly statements and live Yahoo Finance / Bloomberg RSS news into an in-memory **ChromaDB** vector store using `nomic-embed-text`.
- **Local GPU Inference**: Powered by **Ollama** (`qwen2.5:7b` + `nomic-embed-text`) with streaming citation support.
- **One-Click Windows WSL App Launcher**: Includes `start_terminal.sh` and `Alpha_RAG_Terminal.bat` to launch the dashboard as a standalone desktop window and automatically free GPU VRAM on exit.

## Quick Start
1. Pull required Ollama models:

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

2. Create virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Launch the Streamlit RAG Terminal:

```bash
streamlit run rag_dashboard.py
```
