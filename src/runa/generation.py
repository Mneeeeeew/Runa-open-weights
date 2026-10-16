from __future__ import annotations

import torch

from .loading import RunaBundle


@torch.inference_mode()
def generate_text(
    bundle: RunaBundle,
    *,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    seed: int = 42,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")

    prompt_ids = bundle.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must produce at least one token")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokens = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=bundle.device,
    )

    for _ in range(max_new_tokens):
        context = tokens[:, -bundle.config.max_seq_len :]
        next_logits = bundle.model(context)[:, -1, :].float()
        if not torch.isfinite(next_logits).all().item():
            raise RuntimeError("Model produced NaN or infinite logits")

        if temperature == 0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            k = min(top_k, bundle.config.vocab_size)
            if k > 0:
                top_values, top_ids = torch.topk(next_logits, k=k)
                probabilities = torch.softmax(top_values, dim=-1)
                choice = torch.multinomial(probabilities, num_samples=1)
                next_token = top_ids.gather(1, choice)
            else:
                probabilities = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)

    return bundle.decode(tokens[0].tolist())

