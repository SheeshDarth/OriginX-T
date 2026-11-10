"""Tests for the ORIGIN-T-Bench release (B12, HANDBOOK §9.5): nothing unverified ships."""

import hashlib
import json
import tarfile

import pytest

from src.bench.release import card, licence_id, main, package, push, verify


def _bench(tmp_path, licence="CC BY-SA 3.0 (and GFDL), as the source"):
    bench = tmp_path / "bench"
    files = []
    for t, r in (("synthetic", "0.5"), ("recursive", "1")):
        path = bench / "toy" / t / f"r{r}" / "s0.jsonl"
        path.parent.mkdir(parents=True)
        path.write_bytes(f'{{"sample_id": "{t}-0"}}\n'.encode())
        files.append({"path": path.relative_to(bench).as_posix(), "type": t, "ratio": float(r), "seed": 0,
                      "n": 1, "n_contaminated": 1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema": "bench-manifest/1", "version": "0.1",
                "datasets": {"toy": {"licence": licence, "files": files}}}
    (bench / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bench / "README.md").write_text("# ORIGIN-T-Bench\n\nThe datasheet.\n", encoding="utf-8")
    return bench


def test_package_writes_a_verified_folder_archive_and_checksum(tmp_path):
    stage = package(_bench(tmp_path), tmp_path / "dist")
    assert stage.name == "origin-t-bench-v0.1"
    assert (stage / "toy/synthetic/r0.5/s0.jsonl").exists() and (stage / "manifest.json").exists()
    readme = (stage / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\nlicense: cc-by-sa-3.0\n") and "The datasheet." in readme
    archive = tmp_path / "dist" / "origin-t-bench-v0.1.tar.gz"
    with tarfile.open(archive) as tar:
        assert "origin-t-bench-v0.1/toy/recursive/r1/s0.jsonl" in tar.getnames()
    sums = (tmp_path / "dist" / "SHA256SUMS").read_text(encoding="utf-8")
    assert sums == f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"


def test_a_changed_file_is_refused(tmp_path):
    bench = _bench(tmp_path)
    (bench / "toy/synthetic/r0.5/s0.jsonl").write_bytes(b'{"sample_id": "edited"}\n')
    with pytest.raises(ValueError, match="does not match its manifest SHA-256"):
        verify(bench)


def test_a_missing_or_unlisted_file_is_refused(tmp_path):
    bench = _bench(tmp_path)
    (bench / "toy/notes.txt").write_text("scratch", encoding="utf-8")
    with pytest.raises(ValueError, match="not in the manifest"):
        verify(bench)
    (bench / "toy/notes.txt").unlink()
    (bench / "toy/recursive/r1/s0.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        verify(bench)


def test_an_uncleared_licence_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not cleared"):
        verify(_bench(tmp_path, licence="All rights reserved"))
    assert licence_id("CDLA-Sharing-1.0, as the source") == "cdla-sharing-1.0"
    mixed = {"version": "0.2", "datasets": {"a": {"licence": "CC BY-SA 3.0"}, "b": {"licence": "CC0"}}}
    assert card(mixed, "x").startswith("---\nlicense: other\n")


def test_push_uploads_the_stage_folder_private_by_default(tmp_path):
    stage = package(_bench(tmp_path), tmp_path / "dist")
    calls = []
    push(stage, "someone/origin-t-bench", upload=lambda **kw: calls.append(kw))
    assert calls == [{"folder_path": str(stage), "repo_id": "someone/origin-t-bench", "repo_type": "dataset",
                      "commit_message": "ORIGIN-T-Bench 0.1"}]


def test_cli_packages_without_pushing(tmp_path, capsys):
    assert main(["--bench", str(_bench(tmp_path)), "--out", str(tmp_path / "dist")]) == 0
    assert "verified and packaged 2 files" in capsys.readouterr().out
