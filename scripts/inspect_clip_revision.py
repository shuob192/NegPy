"""Read fixed-revision CLIP hashes from Hugging Face without fetching ONNX weights."""

import argparse
import hashlib
import json
import re
import urllib.request
from typing import Any

REPO = "Xenova/clip-vit-base-patch16"
FILES = ("onnx/vision_model_quantized.onnx", "onnx/text_model_quantized.onnx", "merges.txt")


def _fetch(url: str, limit: int = 1_048_576) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "NegPy-Model-Manifest"})
    with urllib.request.urlopen(request, timeout=20) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f"Metadata exceeds the size limit: {url}")
    return data


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


def inspect_revision(revision: str) -> dict[str, Any]:
    """Cross-check Git blobs and LFS pointers at one explicit upstream revision."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Use a complete, lowercase Git commit ID.")
    metadata_url = f"https://huggingface.co/api/models/{REPO}/revision/{revision}?blobs=true"
    metadata = json.loads(_fetch(metadata_url))
    if metadata["sha"] != revision:
        raise ValueError("The API returned a different revision.")
    entries = {entry["rfilename"]: entry for entry in metadata["siblings"]}
    raw_base = f"https://huggingface.co/{REPO}/raw/{revision}"

    def git_file(path: str, limit: int = 1_048_576) -> bytes:
        data = _fetch(f"{raw_base}/{path}", limit)
        if _git_blob(data) != entries[path]["blobId"]:
            raise ValueError(f"Git blob mismatch: {path}")
        return data

    assets = []
    for path in FILES:
        entry = entries[path]
        if path.endswith(".onnx"):
            pointer = git_file(path, 4096).decode("ascii")
            oid = re.search(r"^oid sha256:([0-9a-f]{64})$", pointer, re.MULTILINE)
            length = re.search(r"^size ([0-9]+)$", pointer, re.MULTILINE)
            if oid is None or length is None:
                raise ValueError(f"Invalid LFS pointer: {path}")
            sha256, size = oid[1], int(length[1])
            if entry["lfs"]["sha256"] != sha256 or entry["lfs"]["size"] != size:
                raise ValueError(f"LFS metadata mismatch: {path}")
        else:
            data = git_file(path)
            sha256, size = hashlib.sha256(data).hexdigest(), len(data)
        if size != entry["size"]:
            raise ValueError(f"Asset size mismatch: {path}")
        assets.append({"path": path, "filename": path.rsplit("/", 1)[-1], "sha256": sha256, "size": size, "blob_id": entry["blobId"]})

    config = json.loads(git_file("config.json"))
    return {
        "repo": REPO,
        "revision": revision,
        "metadata_url": metadata_url,
        "assets": assets,
        "preprocessor_config": json.loads(git_file("preprocessor_config.json")),
        "text_context_length": config["text_config"]["max_position_embeddings"],
        "embedding_dim": config["projection_dim"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", help="A reviewed, complete upstream Git commit ID")
    print(json.dumps(inspect_revision(parser.parse_args().revision), indent=2))
