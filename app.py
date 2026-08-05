"""
TeenCode Translator — entry point.

Usage:
    python app.py
    python app.py --share          # public Gradio link (optional, for remote demo)
    python app.py --port 7860

Env:
    MODEL_ID   Hugging Face model id (default: sonleuid1/TeenCode-Translator-BARTpho)
"""

from __future__ import annotations

import argparse

from visualization.gui import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="TeenCode Translator Gradio app")
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (useful if recording on another device)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
