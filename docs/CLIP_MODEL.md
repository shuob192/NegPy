# CLIP model assets

Search by Meaning uses local CPU inference. Downloads start only after the user enables the feature and confirms Download in Preferences.
Readiness checks and inference never fetch missing files.

## Pinned upstream files

Repository: [`Xenova/clip-vit-base-patch16`](https://huggingface.co/Xenova/clip-vit-base-patch16).
Revision: [`9a1940eefaddfa46fe89df96accca7d90bf3a57b`](https://huggingface.co/Xenova/clip-vit-base-patch16/commit/9a1940eefaddfa46fe89df96accca7d90bf3a57b).

The expected hashes and sizes live in `negpy/services/assets/semantic_model.py`.
Downloads use the full commit ID in every URL.
The runtime never accepts a checksum from a download response or a checksum sidecar.

| Upstream file | Size in bytes | SHA-256 |
| --- | ---: | --- |
| `onnx/vision_model_quantized.onnx` | 87461602 | `44eece4fe5fe4e0359a88268a327adf758633a1aade3917690b952bef1501f96` |
| `onnx/text_model_quantized.onnx` | 64504507 | `9106b51e6c663a56b99182ec617c2b3d53577b037e7e24a7717eb78048a0c97a` |
| `merges.txt` | 524619 | `9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a` |

The [revision metadata](https://huggingface.co/api/models/Xenova/clip-vit-base-patch16/revision/9a1940eefaddfa46fe89df96accca7d90bf3a57b?blobs=true) records each file's Git blob ID.
The ONNX hashes come from that revision's Git LFS pointers and match its LFS metadata.
The tokenizer hash is computed from the exact `merges.txt` bytes after checking its Git blob ID.

| Upstream file | Git blob ID |
| --- | --- |
| `onnx/vision_model_quantized.onnx` pointer | `9a951b4e7d21f027194ac3f5b9db4d86f9574b46` |
| `onnx/text_model_quantized.onnx` pointer | `ed0e7111a8b8ce86f551ef320c4107490b888b8d` |
| `merges.txt` | `76e821f1b6f0a9709293c3b6b51ed90980b3166b` |

The revision's configuration matches the preprocessing constants, context length, and embedding size in NegPy.
The model version remains `clip-vit-base-patch16-v1`; existing files remain usable only if they pass verification.

## Cache and download rules

- Readiness requires the expected size and SHA-256 for every asset.
- Each download uses a unique temporary file in the cache directory.
- Verify the closed temporary file, then replace the destination with `os.replace` on the same filesystem.
- Cancellation and failures remove the active temporary file. Completed verified assets remain available for a retry.
- A failed repair leaves the previous destination untouched. An invalid destination cannot pass readiness or load checks.
- Loaders verify cached bytes again and pass those same bytes to ONNX Runtime or the tokenizer.
- Passing verified bytes requires an extra in-memory model copy.
- The ONNX sessions use `CPUExecutionProvider`. Images and search text stay local.

## Update the revision and hashes

1. Review the upstream changes and select a full commit ID. Do not use `main`, a branch, or a movable tag.
2. Inspect that exact revision with the maintenance helper:

   ```bash
   uv run python scripts/inspect_clip_revision.py 9a1940eefaddfa46fe89df96accca7d90bf3a57b
   ```

   Replace the argument with the reviewed revision for a model update.
   The helper reads only official Hugging Face metadata, small Git LFS pointers, tokenizer merges, and configuration files.
   It does not download the ONNX weights.
   It checks the returned revision, Git blob IDs, LFS hashes, and sizes before printing candidate metadata.

3. Review the printed data against the upstream revision. Do not copy hashes from an arbitrary mirror or download response.
4. Update `MODEL_REVISION`, each asset's SHA-256 and size, and this document together.
5. Review `preprocessor_config.json` and `config.json`. Update the preprocessing constants if their values change.
   Check `IMAGE_SIZE`, `_IMAGE_MEAN`, `_IMAGE_STD`, `TEXT_CONTEXT_LENGTH`, and `EMBEDDING_DIM` against the printed configuration.
6. Bump `MODEL_VERSION` if weights, tokenizer data, or preprocessing change. This separates cached model files and stored embeddings.
7. Run the mocked tests and relevant integration checks:

   ```bash
   uv run pytest tests/test_semantic_model.py tests/test_clip_tokenizer.py tests/test_semantic_download_dialog.py tests/test_embeddings_batch.py tests/test_embedding_worker.py tests/test_preferences_dialog.py
   ```

Ordinary tests use tiny synthetic assets with test hashes and mocked network responses.
Keep any full model download or real inference check separate and explicitly requested.
