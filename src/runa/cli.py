from __future__ import annotations

import argparse

from .generation import generate_text
from .loading import load_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a text continuation with a Runa open-weights model."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--weights")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle = load_bundle(
        args.model_dir,
        weights_path=args.weights,
        device=args.device,
    )
    print(f"Model: {bundle.model_dir.name}")
    print(f"Device: {bundle.device}")
    print(f"Weights: {bundle.weights_path}")
    print()
    print(
        generate_text(
            bundle,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()

