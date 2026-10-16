from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer

from .configuration import RunaConfig
from .modeling import RunaLM


@dataclass(frozen=True)
class RunaBundle:
    model: RunaLM
    tokenizer: Tokenizer
    config: RunaConfig
    model_dir: Path
    weights_path: Path
    device: torch.device

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


def _select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_weights(
    model_dir: Path,
    configured_name: str,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path is not None:
        candidate = Path(explicit_path).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    candidates = [
        model_dir / configured_name,
        model_dir.parent.parent / "release-assets" / configured_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    locations = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not find {configured_name}. Checked:\n{locations}\n"
        "Download the matching SafeTensors file from GitHub Releases."
    )


def load_bundle(
    model_dir: str | Path,
    *,
    weights_path: str | Path | None = None,
    device: str = "auto",
) -> RunaBundle:
    model_path = Path(model_dir).expanduser().resolve()
    config_path = model_path / "config.json"
    tokenizer_path = model_path / "tokenizer.json"

    with config_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    config = RunaConfig.from_dict(metadata)
    resolved_weights = _resolve_weights(
        model_path,
        metadata["weight_file"],
        weights_path,
    )

    model = RunaLM(config)
    state = load_file(str(resolved_weights), device="cpu")
    result = model.load_state_dict(state, strict=False)

    expected_missing = {"lm_head.weight"} if config.tie_embeddings else set()
    if set(result.missing_keys) != expected_missing:
        raise RuntimeError(f"Unexpected missing tensors: {result.missing_keys}")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected tensors: {result.unexpected_keys}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected_count = metadata.get("parameter_count")
    if expected_count is not None and parameter_count != expected_count:
        raise RuntimeError(
            f"Parameter count mismatch: expected {expected_count}, got {parameter_count}"
        )

    selected_device = _select_device(device)
    model.requires_grad_(False)
    model.eval()
    model.to(selected_device)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    if tokenizer.get_vocab_size(with_added_tokens=True) != config.vocab_size:
        raise RuntimeError("Tokenizer vocabulary size does not match the model")
    expected_tokenizer_hash = metadata.get("tokenizer_sha256")
    if expected_tokenizer_hash and _sha256(tokenizer_path) != expected_tokenizer_hash:
        raise RuntimeError("Tokenizer SHA-256 does not match config.json")

    return RunaBundle(
        model=model,
        tokenizer=tokenizer,
        config=config,
        model_dir=model_path,
        weights_path=resolved_weights,
        device=selected_device,
    )
