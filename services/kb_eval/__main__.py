"""Entry point for running kbEval server: python -m services.kb_eval [options]."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="kbEval kernel evaluation server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8456, help="Bind port")
    parser.add_argument("--devices", default="cuda:0", help="Comma-separated CUDA devices")
    parser.add_argument("--max-critical-time", type=int, default=120, help="GPU lock timeout (s)")
    parser.add_argument("--max-timeout", type=int, default=600, help="Subprocess timeout (s)")
    parser.add_argument("--code-type", default="triton", help="Default code type")
    parser.add_argument("--data-root", default=None, help="Data root directory")

    args = parser.parse_args()

    from services.kb_eval.server import configure

    devices = [d.strip() for d in args.devices.split(",")]
    configure(
        devices=devices,
        max_critical_time=args.max_critical_time,
        max_timeout=args.max_timeout,
        code_type=args.code_type,
        data_root=args.data_root,
        port=args.port,
    )

    import uvicorn

    uvicorn.run(
        "services.kb_eval.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
