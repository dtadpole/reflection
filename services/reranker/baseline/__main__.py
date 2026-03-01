"""Entry point: python -m services.reranker.baseline [options]."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Reranker server (vLLM backend)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=42983, help="Bind port")
    parser.add_argument(
        "--vllm-url", default="http://localhost:42984", help="vLLM backend URL"
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-32B", help="Model name"
    )

    args = parser.parse_args()

    from services.reranker.baseline.server import configure

    configure(
        vllm_url=args.vllm_url,
        model_name=args.model,
        port=args.port,
    )

    import uvicorn

    uvicorn.run(
        "services.reranker.baseline.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
