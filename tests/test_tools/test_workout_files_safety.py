from tp_mcp.tools import workout_files


def test_download_path_rejects_absolute_path_outside_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(workout_files, "FILE_DATA_DIR", tmp_path / "sandbox")

    target = workout_files._resolve_download_path("/tmp/escape.fit", "123", "456", "x.fit")

    assert target is None


def test_download_path_accepts_relative_path_inside_sandbox(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(workout_files, "FILE_DATA_DIR", sandbox)

    target = workout_files._resolve_download_path("subdir/file.fit", "123", "456", "ignored.fit")

    assert target == (sandbox / "subdir" / "file.fit").resolve()


def test_download_default_uses_safe_filename_inside_sandbox(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    monkeypatch.setattr(workout_files, "FILE_DATA_DIR", sandbox)

    target = workout_files._resolve_download_path(None, "123", "456", "../unsafe.fit")

    assert target == (sandbox / "unsafe.fit").resolve()
