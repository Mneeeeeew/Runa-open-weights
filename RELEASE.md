# Publishing checklist

This directory is a clean, inference-only repository prepared for a GitHub open-weights release.

## Before publishing

1. Replace `OWNER/REPOSITORY` in `README.md` with the final GitHub namespace.
2. Review the Apache-2.0 license choice and replace the generic copyright holder in `NOTICE` if desired.
3. Confirm that `release-assets/manifest.json` and `release-assets/SHA256SUMS.txt` match the files being uploaded.
4. Confirm that no source checkpoint, optimizer state, dataset file, secret, or local path is present.
5. Run the verification commands documented below.

## Repository files versus Release assets

Commit the contents of this directory except the SafeTensors files ignored by `.gitignore`.

Upload these files to a GitHub Release tagged `v1.0.0`:

- `release-assets/runa-46m-f32.safetensors`
- `release-assets/runa-116m-f32.safetensors`
- `release-assets/SHA256SUMS.txt`
- `release-assets/manifest.json`

Do not upload any `.pt`, `.pth`, `.ckpt`, optimizer, scheduler, scaler, dataset, or training-state files.

## Suggested release title

```text
Runa Open Weights v1.0.0
```

## Suggested release notes

```text
Initial open-weights release of Runa-46M and Runa-116M.

- SafeTensors FP32 inference weights
- Runa-46M trained for 400,015,360 tokens
- Runa-116M released checkpoint trained for 1,998,848,000 tokens
- Shared 40,478-token OpenAI GPT-style BPE tokenizer
- Minimal PyTorch inference package and command-line generator
- No optimizer, scheduler, gradient-scaler, RNG, dataset, or resumable training state

See README.md for architecture details, evaluation results, limitations, and usage instructions.
```

## Final local checks

From the repository root:

```powershell
python -m pip install .
runa-generate --model-dir .\models\Runa-46M --weights .\release-assets\runa-46m-f32.safetensors --prompt "The scientist discovered that" --temperature 0 --max-new-tokens 8
runa-generate --model-dir .\models\Runa-116M --weights .\release-assets\runa-116m-f32.safetensors --prompt "The scientist discovered that" --temperature 0 --max-new-tokens 8
```

Inspect files that must not be published:

```powershell
Get-ChildItem -Recurse -File -Include *.pt,*.pth,*.ckpt,*.bin
```

The command should return no training checkpoints or dataset binaries.
