"""
Start the ICT Bot dashboard server.

Usage:
    python run_server.py
    python run_server.py --port 8080
    python run_server.py --reload       # auto-reload on code changes (dev mode)
"""

import argparse
import uvicorn

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args()

    print(f"\n  ICT Trading Bot — Dashboard")
    print(f"  Open http://localhost:{args.port} in your browser\n")

    uvicorn.run(
        "api.server:app",
        host   = args.host,
        port   = args.port,
        reload = args.reload,
    )
