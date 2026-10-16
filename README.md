# Runa Open Weights

Runa is a small family of English causal language models released for research, education, and local experimentation. This repository contains inference code, model configuration, tokenizer files, and links to the published SafeTensors weights for two model sizes:

| Model | Parameters | Training tokens | Context | WikiText-2 loss | WikiText-2 perplexity |
| --- | ---: | ---: | ---: | ---: | ---: |
| Runa-46M | 46,423,552 | 400,015,360 | 512 | 4.2849 | 72.5949 |
| Runa-116M | 116,040,960 | 1,998,848,000 | 512 | 3.4474 | 31.4190 |

Runa models are **base language models**, not instruction-tuned assistants. They predict continuations of text and do not use chat roles, system prompts, or a chat template.

## Release contents

The GitHub repository contains only the files required for inference and documentation. Training datasets, optimizer state, scheduler state, gradient-scaler state, random-number-generator state, and training scripts are not part of the open-weights release.

Download the desired weight file from the repository's **Releases** page:

| Model | Download | Precision |
| --- | --- | --- |
| Runa-46M | [Download weights](https://github.com/Mneeeeeew/Runa-open-weights/releases/download/v1.0.0/runa-46m-f32.safetensors) | FP32 |
| Runa-116M | [Download weights](https://github.com/Mneeeeeew/Runa-open-weights/releases/download/v1.0.0/runa-116m-f32.safetensors) | FP32 |

The weights are stored in SafeTensors format. The tied `lm_head.weight` tensor is represented by `embedding.weight` and is recreated by the loader, avoiding a redundant copy in each file.

## Quick start

Requirements:

- Python 3.10 or newer
- PyTorch 2.1 or newer
- `safetensors`
- `tokenizers`

Clone and install the inference package:

```bash
git clone https://github.com/Mneeeeeew/Runa-open-weights.git
cd Runa-open-weights
python -m pip install .
```

Download one Release asset and place it in the corresponding model directory:

```text
models/Runa-46M/runa-46m-f32.safetensors
```

or:

```text
models/Runa-116M/runa-116m-f32.safetensors
```

Generate a continuation:

```bash
runa-generate \
  --model-dir models/Runa-116M \
  --prompt "The scientist discovered that" \
  --max-new-tokens 80 \
  --temperature 0.8 \
  --top-k 40
```

Windows PowerShell example:

```powershell
runa-generate `
  --model-dir .\models\Runa-116M `
  --prompt "The scientist discovered that" `
  --max-new-tokens 80 `
  --temperature 0.8 `
  --top-k 40
```

For deterministic greedy decoding, set `--temperature 0`.

### Python API

```python
from runa import generate_text, load_bundle

bundle = load_bundle("models/Runa-116M")
text = generate_text(
    bundle,
    prompt="The scientist discovered that",
    max_new_tokens=80,
    temperature=0.8,
    top_k=40,
    seed=42,
)
print(text)
```

The loader selects CUDA when available and otherwise uses the CPU. Pass `device="cpu"`, `device="cuda"`, or `device="mps"` to override automatic selection.

## Model architecture

Both variants are decoder-only Transformers with:

- causal self-attention;
- rotary position embeddings (RoPE);
- RMSNorm;
- SwiGLU feed-forward layers;
- pre-normalized residual blocks;
- tied input and output embeddings;
- no attention or MLP biases;
- a maximum context length of 512 tokens.

| Configuration | Runa-46M | Runa-116M |
| --- | ---: | ---: |
| Layers | 8 | 12 |
| Hidden size | 512 | 768 |
| Attention heads | 8 | 12 |
| Head dimension | 64 | 64 |
| Feed-forward size | 1,408 | 2,048 |
| Vocabulary size | 40,478 | 40,478 |
| RoPE base | 10,000 | 10,000 |
| RMSNorm epsilon | 1e-6 | 1e-6 |

The implementation is custom and is not currently a native Hugging Face Transformers, GGUF, llama.cpp, or LM Studio architecture. Use the included PyTorch loader for faithful inference.

## Tokenizer

Both models use the same lowercasing, 40,478-token BPE tokenizer based on the OpenAI GPT tokenizer. The tokenizer applies BERT-style normalization and pre-tokenization followed by BPE with an `</w>` end-of-word suffix.

The tokenizer has an unknown token (`<unk>`) but no BOS, EOS, padding, or chat-control tokens. Generation therefore stops after `max_new_tokens`; it does not stop on an EOS token.

The tokenizer configuration is included separately in each model directory so that every model artifact is self-describing.

## Training data

Runa was trained on the `sample-10BT` configuration of [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), revision:

```text
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
```

FineWeb-Edu is an English web dataset released under the ODC-By 1.0 license and is also subject to the Common Crawl Terms of Use. No training text is redistributed in this repository.

Runa-46M was trained for 400,015,360 tokens. The released Runa-116M checkpoint was trained for 1,998,848,000 tokens (approximately 2 billion).

## Evaluation

The reported values were measured on `Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, test split, revision:

```text
b08601e04326c79dfdd32d625aee71d232d685c3
```

Evaluation settings:

- FP32 inference;
- context length 512;
- stride 256;
- documents joined with two newline characters;
- no special tokens added;
- every target counted once, excluding the first token in the stream;
- positions restarted for each evaluation window;
- 279,022 evaluated tokens.

The tokenizer and evaluation protocol differ from many modern language-model benchmarks. These values should only be compared with results obtained under the same protocol. Training/test overlap beyond exact validation-document exclusion was not audited.

## Intended use

Runa is intended for:

- educational study of small language models;
- local next-token prediction and text-completion experiments;
- inference and benchmarking research;
- analysis of scaling behavior between small model sizes.

Runa is not intended to be treated as a factual authority, safety classifier, autonomous agent, or production chat assistant.

## Limitations and risks

- The models are small and may produce incoherent, repetitive, or contradictory text.
- They are base completion models and have not been instruction tuned or preference aligned.
- They have not received dedicated safety training.
- Their web-derived training data may contain bias, stereotypes, offensive material, personal information, or copyrighted text.
- Outputs may be factually incorrect or fabricated.
- The 512-token context window is short by current standards.
- The tokenizer lowercases input, so original capitalization is not preserved during encoding.
- The models are primarily English-language models. Other languages are not supported or evaluated.
- Memorization and benchmark contamination have not been comprehensively audited.

Users are responsible for evaluating the models for their own application and for adding appropriate safeguards.

## Reproducibility and integrity

SHA-256 checksums for both weight files are provided in `release-assets/SHA256SUMS.txt`. Verify a downloaded file before use.

Linux:

```bash
sha256sum -c SHA256SUMS.txt
```

PowerShell:

```powershell
Get-FileHash .\runa-116m-f32.safetensors -Algorithm SHA256
```

Additional provenance, including source-checkpoint hashes, checkpoint steps, validation loss, tensor counts, and release-file hashes, is recorded in `release-assets/manifest.json`.

## Training resumption

This release contains model parameters only. It does not contain the state required to exactly resume the original training runs.

Because the parameters and architecture are openly available, third parties can technically fine-tune the models or start a new training process. An open-weights release cannot technically prevent that; it can only omit the original optimizer and training state.

## License

The Runa inference code and Runa model weights are released under the [Apache License 2.0](LICENSE). The tokenizer is based on the MIT-licensed OpenAI GPT tokenizer. Dataset and third-party attributions are listed in [NOTICE](NOTICE).
