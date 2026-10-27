"""CLIP text/image embeddings for "search by meaning": one shared vector space where a
photo and a plain-language description of it land close together, so a query becomes a
nearest-neighbour lookup over cached image vectors instead of a keyword match.

The feature costs nothing until it is turned on: the ONNX sessions and tokenizer build
lazily, and the weights download on demand (download_clip_model) rather than shipping with
the app, which carries only the `onnxruntime` engine.

Model: Xenova/clip-vit-base-patch16 on Hugging Face, the quantized per-tower ONNX exports
plus the tokenizer's merges.txt. The preprocessing constants below come from that repo's
preprocessor_config.json, so a model swap needs new constants rather than new file names.
MODEL_VERSION names the cache directory for that reason: a retired model's download cannot
pass for the current one under the same generic filenames.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from PIL import Image

from negpy.kernel.system.config import APP_CONFIG
from negpy.kernel.system.i18n import tr
from negpy.kernel.system.logging import get_logger
from negpy.kernel.system.version import get_app_version

if TYPE_CHECKING:
    from negpy.services.assets.clip_tokenizer import ClipTokenizer

logger = get_logger(__name__)

# Bumped whenever the model or preprocessing changes, so a retired model's cached
# embeddings are never compared against a new model's (repository.py's
# image_embeddings.model_version). It also names the download cache directory below.
MODEL_VERSION = "clip-vit-base-patch16-v1"
# Named once here, for the two surfaces that warn about the download before it starts.
MODEL_DOWNLOAD_SIZE = "about 150 MB"

_REPO = "Xenova/clip-vit-base-patch16"
MODEL_REVISION = "9a1940eefaddfa46fe89df96accca7d90bf3a57b"
_BASE_URL = f"https://huggingface.co/{_REPO}/resolve/{MODEL_REVISION}"
VISION_FILE = "vision_model_quantized.onnx"
TEXT_FILE = "text_model_quantized.onnx"
MERGES_FILE = "merges.txt"


@dataclass(frozen=True)
class _ModelAsset:
    path: str
    name: str
    sha256: str
    size: int


# Pinned Git LFS OIDs for ONNX; merges.txt is hashed from its verified Git blob.
# docs/CLIP_MODEL.md records the upstream metadata and the revision update procedure.
_ASSETS = (
    _ModelAsset(f"onnx/{VISION_FILE}", VISION_FILE, "44eece4fe5fe4e0359a88268a327adf758633a1aade3917690b952bef1501f96", 87461602),
    _ModelAsset(f"onnx/{TEXT_FILE}", TEXT_FILE, "9106b51e6c663a56b99182ec617c2b3d53577b037e7e24a7717eb78048a0c97a", 64504507),
    _ModelAsset(MERGES_FILE, MERGES_FILE, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a", 524619),
)

_USER_AGENT = f"NegPy/{get_app_version()} (+https://github.com/marcinz606/NegPy)"
_CHUNK = 256 * 1024

# preprocessor_config.json: shortest-edge resize to this, then a centre crop of the
# same size, bicubic (PIL.Image.BICUBIC), rescaled to [0,1] then normalized.
IMAGE_SIZE = 224
_IMAGE_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_IMAGE_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# config.json: text_config.max_position_embeddings.
TEXT_CONTEXT_LENGTH = 77

EMBEDDING_DIM = 512


class ClipDownloadError(Exception):
    pass


def _model_dir() -> str:
    return os.path.join(APP_CONFIG.cache_dir, "clip_model", MODEL_VERSION)


def _model_path(filename: str) -> str:
    return os.path.join(_model_dir(), filename)


def clip_model_ready() -> bool:
    """Every cached asset matches the pinned size and SHA-256 without network access."""
    return all(_file_matches(_model_path(asset.name), asset) for asset in _ASSETS)


def _check_cancelled(is_cancelled: Optional[Callable[[], bool]]) -> None:
    if is_cancelled is not None and is_cancelled():
        raise ClipDownloadError(tr("Download cancelled."))


def _file_matches(path: str, asset: _ModelAsset, is_cancelled: Optional[Callable[[], bool]] = None) -> bool:
    _check_cancelled(is_cancelled)
    try:
        with open(path, "rb") as source:
            if os.fstat(source.fileno()).st_size != asset.size:
                return False
            digest = hashlib.sha256()
            size = 0
            while True:
                _check_cancelled(is_cancelled)
                chunk = source.read(_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > asset.size:
                    return False
                digest.update(chunk)
    except OSError:
        return False
    _check_cancelled(is_cancelled)
    return size == asset.size and digest.hexdigest() == asset.sha256


def _verification_error(name: str) -> ClipDownloadError:
    return ClipDownloadError(tr("Model verification failed for {name}. Download the model again from Preferences.", name=name))


def _verified_asset_bytes(name: str) -> bytes:
    """Return the verified bytes themselves so loaders cannot reopen a changed cache file."""
    asset = next(asset for asset in _ASSETS if asset.name == name)
    try:
        with open(_model_path(name), "rb") as source:
            data = source.read(asset.size + 1)
    except OSError as exc:
        raise _verification_error(name) from exc
    if len(data) != asset.size or hashlib.sha256(data).hexdigest() != asset.sha256:
        raise _verification_error(name)
    return data


def download_clip_model(
    on_progress: Optional[Callable[[int, int], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> None:
    """Fetch and verify pinned assets, then replace each cached file atomically.

    Completed valid files survive retries. Cancellation removes the active temporary file.
    Progress reports bytes across the complete asset set.
    """
    _check_cancelled(is_cancelled)
    try:
        os.makedirs(_model_dir(), exist_ok=True)
    except OSError as exc:
        raise ClipDownloadError(tr("Download failed: {exc}", exc=exc)) from exc
    total = sum(asset.size for asset in _ASSETS)
    done = 0
    for asset in _ASSETS:
        _check_cancelled(is_cancelled)
        dest = _model_path(asset.name)
        if _file_matches(dest, asset, is_cancelled):
            done += asset.size
            if on_progress is not None:
                on_progress(done, total)
            continue
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=_model_dir(), prefix=f".{asset.name}.", suffix=".part", delete=False) as out:
                tmp = out.name
                _check_cancelled(is_cancelled)
                request = urllib.request.Request(f"{_BASE_URL}/{asset.path}", headers={"User-Agent": _USER_AGENT})
                with urllib.request.urlopen(request, timeout=30) as response:
                    received = 0
                    while True:
                        _check_cancelled(is_cancelled)
                        chunk = response.read(_CHUNK)
                        _check_cancelled(is_cancelled)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > asset.size:
                            raise _verification_error(asset.name)
                        out.write(chunk)
                        done += len(chunk)
                        if on_progress is not None:
                            on_progress(done, total)
            if not _file_matches(tmp, asset, is_cancelled):
                raise _verification_error(asset.name)
            _check_cancelled(is_cancelled)
            os.replace(tmp, dest)
        except ClipDownloadError:
            raise
        except Exception as exc:
            raise ClipDownloadError(tr("Download failed: {exc}", exc=exc)) from exc
        finally:
            if tmp is not None:
                _discard(tmp)
    _check_cancelled(is_cancelled)


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-12 else vec


# CLIP cosine similarities run low even for a good match and cluster around a baseline
# that shifts with the query and the library, so a fixed cutoff cannot separate a match
# from noise. The cutoff is relative instead: how many standard deviations above this
# query's own mean score a candidate sits.
SIMILARITY_Z_SCORE = 3.0


def rank_by_similarity(query: np.ndarray, candidates: dict) -> list:
    """Keys of `candidates` (key -> L2-normalized vector) ranked by cosine similarity
    to `query`, most relevant first. A candidate scoring less than SIMILARITY_Z_SCORE
    standard deviations above this query's own mean score is excluded rather than kept
    at the bottom, so a search narrows instead of just reordering -- the one rule both
    the in-session and whole-library rankings share."""
    if not candidates:
        return []
    keys = list(candidates.keys())
    scores = np.stack([candidates[key] for key in keys]) @ query
    threshold = scores.mean() + SIMILARITY_Z_SCORE * scores.std()
    order = np.argsort(-scores)
    return [keys[i] for i in order if scores[i] >= threshold]


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """PIL image -> (1, 3, 224, 224) float32, CLIP's own resize/crop/normalize."""
    img = image.convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    img = img.resize((max(IMAGE_SIZE, round(w * scale)), max(IMAGE_SIZE, round(h * scale))), Image.BICUBIC)
    w, h = img.size
    left, top = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
    img = img.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - _IMAGE_MEAN) / _IMAGE_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def _run(session, **named_arrays) -> list:
    """Feeds `named_arrays` to `session` by matching each of its own declared input
    names -- a naming mismatch against the model's actual ONNX graph then fails with a
    clear KeyError instead of silently running the wrong tensor into the wrong slot."""
    feed = {}
    for inp in session.get_inputs():
        if inp.name not in named_arrays:
            raise KeyError(f"CLIP ONNX model expects input '{inp.name}', not provided (have: {sorted(named_arrays)})")
        feed[inp.name] = named_arrays[inp.name]
    return session.run(None, feed)


class ClipModel:
    """Lazy CLIP inference: the ONNX sessions and tokenizer are constructed on first
    use, not at import or construction time, so holding an idle instance costs
    nothing. One instance is meant to be shared/reused (sessions are expensive to
    build, cheap to run repeatedly)."""

    def __init__(self) -> None:
        self._vision_session = None
        self._text_session = None
        self._tokenizer = None

    def _ensure_vision(self):
        if self._vision_session is None:
            model_bytes = _verified_asset_bytes(VISION_FILE)
            import onnxruntime as ort

            self._vision_session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        return self._vision_session

    def _ensure_text(self):
        if self._text_session is None:
            model_bytes = _verified_asset_bytes(TEXT_FILE)
            import onnxruntime as ort

            self._text_session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
        return self._text_session

    def _ensure_tokenizer(self) -> "ClipTokenizer":
        if self._tokenizer is None:
            from negpy.services.assets.clip_tokenizer import ClipTokenizer

            self._tokenizer = ClipTokenizer.from_bytes(_verified_asset_bytes(MERGES_FILE))
        return self._tokenizer

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """L2-normalized (EMBEDDING_DIM,) vector -- similarity to embed_text's output
        is then a plain dot product."""
        pixel_values = _preprocess_image(image)
        outputs = _run(self._ensure_vision(), pixel_values=pixel_values)
        return _l2_normalize(np.asarray(outputs[0][0], dtype=np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        input_ids, attention_mask = self._ensure_tokenizer().encode(text, TEXT_CONTEXT_LENGTH)
        outputs = _run(self._ensure_text(), input_ids=input_ids, attention_mask=attention_mask)
        return _l2_normalize(np.asarray(outputs[0][0], dtype=np.float32))
