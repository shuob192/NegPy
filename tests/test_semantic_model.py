"""download_clip_model / clip_model_ready / preprocessing and embedding math. The ONNX
sessions themselves are mocked -- these tests never run real inference or touch the
network."""

import hashlib
import io
import os
import sys
from dataclasses import replace
from pathlib import Path
from threading import Event
import urllib.request
from collections.abc import Callable
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from negpy.services.assets import semantic_model as sm


@pytest.fixture(autouse=True)
def _sandbox_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sm.APP_CONFIG, "cache_dir", str(tmp_path))
    monkeypatch.setattr(urllib.request, "urlopen", MagicMock(side_effect=AssertionError("Unexpected model download")))


def _use_assets(monkeypatch: pytest.MonkeyPatch, payloads: dict[str, bytes]) -> None:
    monkeypatch.setattr(
        sm,
        "_ASSETS",
        tuple(
            replace(asset, sha256=hashlib.sha256(payloads[asset.name]).hexdigest(), size=len(payloads[asset.name])) for asset in sm._ASSETS
        ),
    )


@pytest.fixture
def fake_assets(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    payloads = {asset.name: f"data-for-{asset.name}".encode() for asset in sm._ASSETS}
    payloads[sm.MERGES_FILE] = b"#version: 0.2\nc a\nca t\n"
    _use_assets(monkeypatch, payloads)
    return payloads


@pytest.fixture
def mock_download(monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes]) -> MagicMock:
    def open_asset(request: urllib.request.Request, timeout: float = 30) -> _FakeResponse:
        for asset in sm._ASSETS:
            if request.full_url == f"{sm._BASE_URL}/{asset.path}":
                assert request.get_method() == "GET"
                return _FakeResponse(fake_assets[asset.name])
        raise AssertionError(f"Unexpected URL: {request.full_url}")

    network = MagicMock(side_effect=open_asset)
    monkeypatch.setattr(urllib.request, "urlopen", network)
    return network


def _write_cache(payloads: dict[str, bytes]) -> None:
    os.makedirs(sm._model_dir(), exist_ok=True)
    for name, data in payloads.items():
        Path(sm._model_path(name)).write_bytes(data)


def _temporary_files() -> list[Path]:
    return list(Path(sm._model_dir()).glob("*.part"))


class _FakeResponse:
    def __init__(self, data: bytes, headers: dict | None = None):
        self._buf = io.BytesIO(data)
        self.headers = headers or {"Content-Length": str(len(data))}
        self.status = 200

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_clip_model_ready_false_before_any_download():
    assert sm.clip_model_ready() is False


def test_download_writes_every_expected_file(fake_assets: dict[str, bytes], mock_download: MagicMock) -> None:
    sm.download_clip_model()

    assert sm.clip_model_ready() is True
    for asset in sm._ASSETS:
        assert Path(sm._model_path(asset.name)).read_bytes() == fake_assets[asset.name]
    assert mock_download.call_count == len(sm._ASSETS)
    assert not _temporary_files()


def test_download_reports_progress_across_all_files(
    monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes], mock_download: MagicMock
) -> None:
    fake_assets.update({name: b"x" * 100 for name in fake_assets})
    _use_assets(monkeypatch, fake_assets)
    seen = []
    sm.download_clip_model(on_progress=lambda done, total: seen.append((done, total)))

    assert seen
    final_done, final_total = seen[-1]
    assert final_done == final_total == 300


def test_download_cancelled_midway_leaves_no_ready_files(mock_download: MagicMock) -> None:
    cancelled = Event()
    with pytest.raises(sm.ClipDownloadError, match="cancelled"):
        sm.download_clip_model(on_progress=lambda done, total: cancelled.set(), is_cancelled=cancelled.is_set)

    assert sm.clip_model_ready() is False
    assert mock_download.call_count == 1
    assert not any(os.path.isfile(sm._model_path(asset.name)) for asset in sm._ASSETS)
    assert not _temporary_files()


def test_download_is_resumable_per_file(fake_assets: dict[str, bytes], mock_download: MagicMock) -> None:
    first = sm._ASSETS[0]
    _write_cache({first.name: fake_assets[first.name]})

    sm.download_clip_model()

    assert mock_download.call_count == len(sm._ASSETS) - 1
    assert all(first.path not in call.args[0].full_url for call in mock_download.call_args_list)
    assert sm.clip_model_ready()


def test_urls_pin_a_complete_upstream_commit() -> None:
    assert len(sm.MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in sm.MODEL_REVISION)
    assert sm._BASE_URL.endswith(f"/resolve/{sm.MODEL_REVISION}")
    assert "/main" not in sm._BASE_URL
    assert all(len(asset.sha256) == 64 and asset.size > 0 for asset in sm._ASSETS)


def test_an_idle_model_and_readiness_check_do_not_request_downloads() -> None:
    model = sm.ClipModel()
    assert model._vision_session is model._text_session is model._tokenizer is None
    assert not sm.clip_model_ready()
    urllib.request.urlopen.assert_not_called()


@pytest.mark.parametrize("name", [sm.VISION_FILE, sm.TEXT_FILE, sm.MERGES_FILE])
def test_readiness_rejects_same_size_cache_corruption(fake_assets: dict[str, bytes], name: str) -> None:
    _write_cache(fake_assets)
    assert sm.clip_model_ready()
    path = Path(sm._model_path(name))
    original_stat = path.stat()
    path.write_bytes(b"!" + fake_assets[name][1:])
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert not sm.clip_model_ready()
    urllib.request.urlopen.assert_not_called()


def test_corrupt_cache_is_replaced_only_after_a_valid_download(fake_assets: dict[str, bytes], mock_download: MagicMock) -> None:
    _write_cache(fake_assets)
    first = sm._ASSETS[0]
    Path(sm._model_path(first.name)).write_bytes(b"truncated")

    sm.download_clip_model()

    mock_download.assert_called_once()
    assert mock_download.call_args.args[0].full_url.endswith(first.path)
    assert sm.clip_model_ready()
    assert not _temporary_files()


@pytest.mark.parametrize("kind", ["same-size", "truncated", "oversized"])
@pytest.mark.parametrize("name", [sm.VISION_FILE, sm.TEXT_FILE, sm.MERGES_FILE])
def test_invalid_downloads_never_promote_an_asset(
    monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes], kind: str, name: str
) -> None:
    data = fake_assets[name]
    invalid = {"same-size": b"!" + data[1:], "truncated": data[:-1], "oversized": data + b"!"}[kind]
    headers = {"Content-Length": str(len(invalid)), "X-Checksum-Sha256": hashlib.sha256(invalid).hexdigest()}

    def open_asset(request: urllib.request.Request, timeout: float = 30) -> _FakeResponse:
        asset = next(asset for asset in sm._ASSETS if request.full_url.endswith(asset.path))
        return _FakeResponse(invalid, headers) if asset.name == name else _FakeResponse(fake_assets[asset.name])

    monkeypatch.setattr(urllib.request, "urlopen", MagicMock(side_effect=open_asset))
    promote = MagicMock(wraps=os.replace)
    monkeypatch.setattr(sm.os, "replace", promote)

    with pytest.raises(sm.ClipDownloadError, match="verification failed"):
        sm.download_clip_model()

    assert all(call.args[1] != sm._model_path(name) for call in promote.call_args_list)
    assert not Path(sm._model_path(name)).exists()
    assert not _temporary_files()


def test_interrupted_download_keeps_completed_assets_and_removes_the_partial_file(
    monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes]
) -> None:
    first, second = sm._ASSETS[:2]
    response = _FakeResponse(fake_assets[second.name])
    response.read = MagicMock(side_effect=[fake_assets[second.name][:4], ConnectionResetError("interrupted")])
    network = MagicMock(side_effect=[_FakeResponse(fake_assets[first.name]), response])
    monkeypatch.setattr(urllib.request, "urlopen", network)

    with pytest.raises(sm.ClipDownloadError, match="interrupted"):
        sm.download_clip_model()

    assert Path(sm._model_path(first.name)).read_bytes() == fake_assets[first.name]
    assert not Path(sm._model_path(second.name)).exists()
    assert not sm.clip_model_ready()
    assert not _temporary_files()


def test_connection_failure_cleans_up_the_temporary_file(monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes]) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", MagicMock(side_effect=OSError("offline")))
    with pytest.raises(sm.ClipDownloadError, match="offline"):
        sm.download_clip_model()
    assert not _temporary_files()
    assert not sm.clip_model_ready()


def test_atomic_promotion_failure_cleans_up_the_temporary_file(monkeypatch: pytest.MonkeyPatch, mock_download: MagicMock) -> None:
    monkeypatch.setattr(sm.os, "replace", MagicMock(side_effect=OSError("promotion failed")))
    with pytest.raises(sm.ClipDownloadError, match="promotion failed"):
        sm.download_clip_model()
    assert not _temporary_files()
    assert not sm.clip_model_ready()


def test_cancellation_before_start_makes_no_request_or_cache_directory() -> None:
    with pytest.raises(sm.ClipDownloadError, match="cancelled"):
        sm.download_clip_model(is_cancelled=lambda: True)
    urllib.request.urlopen.assert_not_called()
    assert not Path(sm._model_dir()).exists()


def test_cancellation_after_verification_prevents_promotion(monkeypatch: pytest.MonkeyPatch, mock_download: MagicMock) -> None:
    cancelled = Event()
    matches = sm._file_matches

    def cancel_after_verification(path: str, asset: sm._ModelAsset, is_cancelled: Callable[[], bool] | None = None) -> bool:
        result = matches(path, asset, is_cancelled)
        if path.endswith(".part"):
            cancelled.set()
        return result

    monkeypatch.setattr(sm, "_file_matches", cancel_after_verification)
    with pytest.raises(sm.ClipDownloadError, match="cancelled"):
        sm.download_clip_model(is_cancelled=cancelled.is_set)
    assert not Path(sm._model_path(sm.VISION_FILE)).exists()
    assert not _temporary_files()


def test_valid_caches_need_no_network(fake_assets: dict[str, bytes]) -> None:
    _write_cache(fake_assets)
    sm.download_clip_model()
    assert sm.clip_model_ready()
    urllib.request.urlopen.assert_not_called()


@pytest.mark.parametrize(
    "method,name", [("_ensure_vision", sm.VISION_FILE), ("_ensure_text", sm.TEXT_FILE), ("_ensure_tokenizer", sm.MERGES_FILE)]
)
def test_loaders_reject_cache_corruption_even_after_a_readiness_check(
    monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes], method: str, name: str
) -> None:
    _write_cache(fake_assets)
    assert sm.clip_model_ready()
    Path(sm._model_path(name)).write_bytes(b"!" + fake_assets[name][1:])
    runtime = MagicMock()
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    from negpy.services.assets.clip_tokenizer import ClipTokenizer

    tokenizer = MagicMock()
    monkeypatch.setattr(ClipTokenizer, "from_bytes", tokenizer)

    with pytest.raises(sm.ClipDownloadError, match="verification failed"):
        getattr(sm.ClipModel(), method)()

    runtime.InferenceSession.assert_not_called()
    tokenizer.assert_not_called()
    urllib.request.urlopen.assert_not_called()


@pytest.mark.parametrize("method,name", [("_ensure_vision", sm.VISION_FILE), ("_ensure_text", sm.TEXT_FILE)])
def test_runtime_loads_the_verified_bytes_locally(
    monkeypatch: pytest.MonkeyPatch, fake_assets: dict[str, bytes], method: str, name: str
) -> None:
    _write_cache(fake_assets)
    runtime = MagicMock()
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)

    getattr(sm.ClipModel(), method)()

    runtime.InferenceSession.assert_called_once_with(fake_assets[name], providers=["CPUExecutionProvider"])
    urllib.request.urlopen.assert_not_called()


def test_tokenizer_loads_verified_bytes_without_reopening_the_file(fake_assets: dict[str, bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    from negpy.services.assets.clip_tokenizer import ClipTokenizer

    _write_cache(fake_assets)
    construct = ClipTokenizer.from_bytes

    def remove_cache_then_construct(data: bytes) -> ClipTokenizer:
        Path(sm._model_path(sm.MERGES_FILE)).unlink()
        assert data == fake_assets[sm.MERGES_FILE]
        return construct(data)

    monkeypatch.setattr(ClipTokenizer, "from_bytes", remove_cache_then_construct)
    tokenizer = sm.ClipModel()._ensure_tokenizer()
    assert tokenizer.encode("cat", sm.TEXT_CONTEXT_LENGTH)[0].shape == (1, sm.TEXT_CONTEXT_LENGTH)


def test_l2_normalize_unit_length():
    vec = np.array([3.0, 4.0], dtype=np.float32)
    normed = sm._l2_normalize(vec)
    assert abs(float(np.linalg.norm(normed)) - 1.0) < 1e-6


def test_l2_normalize_leaves_a_zero_vector_alone():
    vec = np.zeros(4, dtype=np.float32)
    assert np.array_equal(sm._l2_normalize(vec), vec)


def test_preprocess_image_shape_and_range():
    img = Image.new("RGB", (400, 300), color=(128, 64, 32))
    arr = sm._preprocess_image(img)
    assert arr.shape == (1, 3, sm.IMAGE_SIZE, sm.IMAGE_SIZE)
    assert arr.dtype == np.float32
    # A uniform-color image should preprocess to a uniform (per-channel) array.
    assert np.allclose(arr[0, 0], arr[0, 0, 0, 0])


def test_preprocess_image_handles_a_narrow_source():
    """The shorter edge governs the resize, so a very wide or very tall source still
    lands on the model's fixed square input instead of failing the centre crop."""
    img = Image.new("RGB", (50, 500), color=(10, 20, 30))
    arr = sm._preprocess_image(img)
    assert arr.shape == (1, 3, sm.IMAGE_SIZE, sm.IMAGE_SIZE)


def test_run_raises_a_clear_error_on_an_input_name_mismatch():
    fake_input = MagicMock()
    fake_input.name = "totally_unexpected_name"
    session = MagicMock()
    session.get_inputs.return_value = [fake_input]

    with pytest.raises(KeyError):
        sm._run(session, pixel_values=np.zeros((1, 3, 2, 2), dtype=np.float32))


def test_embed_image_uses_the_visual_session_and_normalizes():
    model = sm.ClipModel()
    fake_input = MagicMock()
    fake_input.name = "pixel_values"
    session = MagicMock()
    session.get_inputs.return_value = [fake_input]
    session.run.return_value = [np.array([[3.0, 4.0] + [0.0] * (sm.EMBEDDING_DIM - 2)], dtype=np.float32)]
    model._vision_session = session

    out = model.embed_image(Image.new("RGB", (224, 224)))

    assert out.shape == (sm.EMBEDDING_DIM,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5


def test_embed_text_uses_the_tokenizer_and_text_session():
    model = sm.ClipModel()
    tokenizer = MagicMock()
    tokenizer.encode.return_value = (np.zeros((1, 77), dtype=np.int64), np.ones((1, 77), dtype=np.int64))
    model._tokenizer = tokenizer

    fake_ids_input, fake_mask_input = MagicMock(), MagicMock()
    fake_ids_input.name, fake_mask_input.name = "input_ids", "attention_mask"
    session = MagicMock()
    session.get_inputs.return_value = [fake_ids_input, fake_mask_input]
    session.run.return_value = [np.array([[1.0] + [0.0] * (sm.EMBEDDING_DIM - 1)], dtype=np.float32)]
    model._text_session = session

    out = model.embed_text("a cat")

    tokenizer.encode.assert_called_once_with("a cat", sm.TEXT_CONTEXT_LENGTH)
    assert out.shape == (sm.EMBEDDING_DIM,)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
