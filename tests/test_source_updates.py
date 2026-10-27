import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from negpy.desktop.session import DesktopSessionManager
from negpy.desktop.view.update_notice import UpdateNotice
from negpy.desktop.view.widgets import update_dialog
from negpy.desktop.view.widgets.preferences_dialog import PreferencesDialog
from negpy.infrastructure.storage.repository import StorageRepository
from negpy.kernel.system import updater, version
from conftest import FakeController


@pytest.fixture
def settings_repo(tmp_path: Path) -> StorageRepository:
    repo = StorageRepository(str(tmp_path / "edits.db"), str(tmp_path / "settings.db"))
    repo.initialize()
    repo.save_global_settings({"ui_scale": 1.2, "personal_setting": {"keep": True}})
    return repo


@pytest.fixture
def releases(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    payload = {
        "tag_name": "v9.9.9",
        "html_url": "https://github.com/marcinz606/NegPy/releases/tag/v9.9.9",
        "body": "Upstream release notes",
        "assets": [{"name": "NegPy-9.9.9-Win64-Setup.exe", "browser_download_url": "https://example.test/setup.exe"}],
    }
    response = MagicMock()
    response.status = 200
    response.read.return_value = json.dumps(payload).encode()
    response.__enter__.return_value = response
    network = MagicMock(return_value=response)
    monkeypatch.setattr(version.urllib.request, "urlopen", network)
    monkeypatch.setattr(updater, "get_app_version", lambda: "0.65.0")
    monkeypatch.setattr(updater.sys, "frozen", False, raising=False)
    monkeypatch.setattr(update_dialog, "find_update", updater.find_update)
    first = len(update_dialog._RUNNING)
    yield network
    for worker in update_dialog._RUNNING[first:]:
        assert worker.wait(2000)
    qapp.processEvents()
    del update_dialog._RUNNING[first:]


def _finish_checks(qapp: QApplication) -> None:
    for worker in update_dialog._RUNNING:
        assert worker.wait(2000)
    qapp.processEvents()


@pytest.mark.parametrize("saved", [None, False, "false"])
def test_default_startup_makes_no_release_request(
    qapp: QApplication, settings_repo: StorageRepository, releases: MagicMock, saved: bool | str | None
) -> None:
    if saved is not None:
        settings_repo.save_global_setting("automatic_update_checks", saved)
    notice = UpdateNotice(QWidget(), settings_repo)

    _finish_checks(qapp)

    releases.assert_not_called()
    assert notice.update_info is None
    assert settings_repo.get_global_setting("ui_scale") == 1.2
    assert settings_repo.get_global_setting("personal_setting") == {"keep": True}


def test_enabling_checks_persists_and_checks_upstream_on_the_next_startup(
    qapp: QApplication, settings_repo: StorageRepository, releases: MagicMock
) -> None:
    session = DesktopSessionManager(settings_repo)
    controller = FakeController(settings_repo)
    controller.session = session
    dialog = PreferencesDialog(controller, pinned_keys=set())
    assert not dialog.automatic_updates_box.isChecked()

    dialog.automatic_updates_box.setChecked(True)

    releases.assert_not_called()
    assert not dialog._restart_hint.isHidden()
    reopened = StorageRepository(settings_repo.edits_db_path, settings_repo.settings_db_path)
    restored = DesktopSessionManager(reopened)
    assert restored.state.automatic_update_checks is True
    notice = UpdateNotice(QWidget(), reopened)
    _finish_checks(qapp)

    releases.assert_called_once()
    assert releases.call_args.args[0].full_url == version.LATEST_RELEASE_API
    assert notice.update_info.version == "9.9.9"
    assert not notice.update_info.can_self_install
    assert not notice.update_info.download_url
    assert reopened.get_global_setting("ui_scale") == 1.2
    assert reopened.get_global_setting("personal_setting") == {"keep": True}

    restored.set_automatic_update_checks(False)
    releases.reset_mock()
    UpdateNotice(QWidget(), reopened)
    _finish_checks(qapp)
    releases.assert_not_called()
    assert reopened.get_global_setting("automatic_update_checks") is False


def test_manual_check_works_with_automatic_checks_disabled(
    qapp: QApplication, settings_repo: StorageRepository, releases: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened = []
    monkeypatch.setattr(
        "negpy.desktop.view.update_notice.UpdateDialog",
        lambda info, parent, repo=None: MagicMock(exec=lambda: opened.append(info)),
    )
    notice = UpdateNotice(QWidget(), settings_repo)

    notice.check_for_updates()
    _finish_checks(qapp)

    releases.assert_called_once()
    assert opened[0].version == "9.9.9"
    assert not opened[0].download_url
    assert settings_repo.get_global_setting("automatic_update_checks", False) is False


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_source_installs_refuse_binary_downloads_and_installation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str) -> None:
    source = tmp_path / "local.py"
    source.write_text("personal changes", encoding="utf-8")
    destination = tmp_path / "download"
    network, spawn = MagicMock(), MagicMock()
    monkeypatch.setattr(updater.sys, "frozen", False, raising=False)
    monkeypatch.setattr(updater.sys, "platform", platform)
    monkeypatch.setenv("APPIMAGE", str(source))
    monkeypatch.setattr(updater.urllib.request, "urlopen", network)
    monkeypatch.setattr(updater.subprocess, "Popen", spawn)
    info = updater.UpdateInfo("9.9.9", "", "https://example.test/release", "setup.exe", "https://example.test/setup.exe")

    assert updater.install_kind() == ""
    assert not info.can_self_install
    with pytest.raises(updater.UpdateError, match="manually with Git"):
        updater.download_asset(info, destination)
    with pytest.raises(updater.UpdateError, match="manually with Git"):
        updater.apply_update(source, info)

    network.assert_not_called()
    spawn.assert_not_called()
    assert not destination.exists()
    assert source.read_text(encoding="utf-8") == "personal changes"


def test_source_release_dialog_offers_upstream_notes_without_a_download(qapp: QApplication, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(updater.sys, "frozen", False, raising=False)
    worker, opened = MagicMock(), MagicMock()
    monkeypatch.setattr(update_dialog, "DownloadWorker", worker)
    monkeypatch.setattr(update_dialog.QDesktopServices, "openUrl", opened)
    info = updater.UpdateInfo("9.9.9", "", "https://example.test/release", "setup.exe", "https://example.test/setup.exe")
    dialog = update_dialog.UpdateDialog(info)

    assert dialog.windowTitle() == "Upstream NegPy 9.9.9 Is Available"
    assert "source fork" in dialog.subtitle.text()
    assert "Git" in dialog.subtitle.text()
    assert dialog.install_button.text() == "Open Upstream Release Notes"
    dialog.install_button.click()

    worker.assert_not_called()
    assert opened.call_args.args[0].toString() == info.page_url


def test_manual_source_check_never_claims_the_fork_is_up_to_date(
    qapp: QApplication, settings_repo: StorageRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages = MagicMock()
    monkeypatch.setattr("negpy.desktop.view.update_notice.start_update_check", lambda callback: callback(None))
    monkeypatch.setattr("negpy.desktop.view.update_notice.QMessageBox.information", messages)
    monkeypatch.setattr(updater.sys, "frozen", False, raising=False)
    notice = UpdateNotice(QWidget(), settings_repo)

    notice.check_for_updates()

    assert messages.call_args.args[1] == "Upstream Releases"
    assert "does not update your source fork" in messages.call_args.args[2]
