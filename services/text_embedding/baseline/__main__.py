"""Entry point: python -m services.text_embedding.baseline [options]."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Text embedding server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=42982, help="Bind port")
    parser.add_argument(
        "--model", default="Qwen/Qwen3-Embedding-8B", help="Model name"
    )
    parser.add_argument("--dimension", type=int, default=4096, help="Embedding dim")
    parser.add_argument("--max-batch-size", type=int, default=64, help="Max batch size")
    parser.add_argument(
        "--max-seq-length", type=int, default=8192, help="Max sequence length"
    )
    parser.add_argument("--device", default="cuda:0", help="Device")

    args = parser.parse_args()

    from services.text_embedding.baseline.server import configure

    configure(
        model_name=args.model,
        dimension=args.dimension,
        max_batch_size=args.max_batch_size,
        max_seq_length=args.max_seq_length,
        device=args.device,
        port=args.port,
    )

    import uvicorn

    uvicorn.run(
        "services.text_embedding.baseline.server:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
