#!/bin/bash
# NIFTY TERMINAL — Quick Setup & Launch Script
# Run this script to install dependencies and start the terminal

set -e
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         NIFTY TERMINAL v2 — SETUP & LAUNCH              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+ first."
    exit 1
fi
echo "✅ Python $(python3 --version)"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✅ Dependencies installed"

# Check for Anthropic API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo ""
    echo "⚠️  ANTHROPIC_API_KEY not set."
    echo "   AI news sentiment will use heuristic fallback."
    echo "   To enable Claude AI: export ANTHROPIC_API_KEY=your-key-here"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  STARTING BACKEND API on http://localhost:8000           ║"
echo "║  FRONTEND at:  open index.html in browser               ║"
echo "║  API Docs at:  http://localhost:8000/docs               ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  HOW TO USE:"
echo "  1. This script starts the FastAPI backend"
echo "  2. Open index.html in your browser"
echo "  3. The API URL field shows http://localhost:8000"
echo "  4. Prices refresh every 60 seconds automatically"
echo "  5. News refreshes every 5 minutes with AI sentiment"
echo ""
echo "  API ENDPOINTS:"
echo "    GET /api/summary    — all data in one call"
echo "    GET /api/prices     — live NIFTY 50 prices"
echo "    GET /api/news       — live news + AI sentiment"
echo "    GET /api/signals    — ML buy/hold/sell signals"
echo "    GET /api/chart/{sym}?period=1d — OHLCV chart data"
echo "    GET /api/sector_performance   — sector returns"
echo "    GET /api/ablation   — K-regime ablation results"
echo "    GET /api/portfolio  — portfolio backtest metrics"
echo "    POST /api/refresh   — force data refresh"
echo "    GET /health         — health check"
echo ""

# Start server
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload --log-level info