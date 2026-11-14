"""ORIGIN-T-Bench release (B12, HANDBOOK §9.5): verify, package and (optionally) upload.

    python -m src.bench.release --bench bench --out dist
    python -m src.bench.release --bench bench --out dist --push <user>/origin-t-bench [--public]

Run it after ``python -m src.bench.build`` on the machine with the grid's pools. It refuses
to package anything it cannot vouch for:

- every file in ``manifest.json`` exists with its recorded SHA-256, and the bench folder holds
  no file the manifest does not list (so nothing unreviewed ships);
- every dataset's licence is one we know how to redistribute (``LICENCES``), and the card
  declares it.

It writes ``<out>/origin-t-bench-v<version>/`` (the files, ``manifest.json`` and a Hugging Face
dataset card built from ``bench/README.md``), a ``.tar.gz`` of that folder for the GitHub
release, and ``SHA256SUMS``. ``--push`` uploads the folder to a Hub dataset repo; it needs a
token from ``hf auth login`` and creates the repo private unless ``--public`` is given.
``huggingface_hub`` is imported only then.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Optional

from .build import SCHEMA

# The start of a manifest "licence" string -> the Hub's licence id.
LICENCES = {
    "CC BY-SA 3.0": "cc-by-sa-3.0",
    "CDLA-Sharing-1.0": "cdla-sharing-1.0",
    "CC0": "cc0-1.0",
}
CARD_HEADER = """---
license: {licence}
pretty_name: ORIGIN-T-Bench v{version}
language:
- en
task_categories:
- text-generation
tags:
- data-contamination
- model-collapse
- synthetic-data
---

"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def licence_id(text: str) -> str:
    """The Hub licence id for a manifest licence string; refuses one we have not cleared."""
    for prefix, hub in LICENCES.items():
        if text.startswith(prefix):
            return hub
    raise ValueError(f"licence {text!r} is not cleared for release; known: {list(LICENCES)}")


def verify(bench: Path) -> dict[str, Any]:
    """The manifest, after checking every hash, that nothing unlisted is present, and the licences."""
    manifest_path = bench / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no {manifest_path}: run python -m src.bench.build first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema {manifest.get('schema')!r}, expected {SCHEMA!r}")
    listed = set()
    for name, entry in manifest["datasets"].items():
        licence_id(entry["licence"])
        for f in entry["files"]:
            path = bench / f["path"]
            if not path.exists():
                raise FileNotFoundError(f"{name}: {f['path']} is in the manifest but missing")
            if _sha256(path) != f["sha256"]:
                raise ValueError(f"{name}: {f['path']} does not match its manifest SHA-256")
            listed.add(Path(f["path"]).as_posix())
    extra = sorted(p.relative_to(bench).as_posix() for p in bench.rglob("*")
                   if p.is_file() and p.name not in ("manifest.json", "README.md")
                   and p.relative_to(bench).as_posix() not in listed)
    if extra:
        raise ValueError(f"{len(extra)} file(s) not in the manifest, e.g. {extra[:3]}")
    return manifest


def card(manifest: Mapping[str, Any], datasheet: str) -> str:
    """The Hub dataset card: YAML metadata, then the datasheet (``bench/README.md``)."""
    ids = sorted({licence_id(e["licence"]) for e in manifest["datasets"].values()})
    licence = ids[0] if len(ids) == 1 else "other"
    return CARD_HEADER.format(licence=licence, version=manifest["version"]) + datasheet


def package(bench: Path, out: Path) -> Path:
    """Verify ``bench`` and write the release folder, its ``.tar.gz`` and ``SHA256SUMS``."""
    manifest = verify(bench)
    name = f"origin-t-bench-v{manifest['version']}"
    stage = out / name
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for entry in manifest["datasets"].values():
        for f in entry["files"]:
            dst = stage / f["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(bench / f["path"], dst)
    shutil.copyfile(bench / "manifest.json", stage / "manifest.json")
    datasheet = (bench / "README.md").read_text(encoding="utf-8").replace("\r\n", "\n")
    (stage / "README.md").write_text(card(manifest, datasheet), encoding="utf-8", newline="\n")
    archive = out / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                tar.add(p, arcname=p.relative_to(out).as_posix())
    (out / "SHA256SUMS").write_text(f"{_sha256(archive)}  {archive.name}\n", encoding="utf-8", newline="\n")
    return stage


def push(stage: Path, repo: str, *, public: bool = False,
         upload: Optional[Callable[..., Any]] = None) -> None:
    """Upload the release folder to the Hub dataset ``repo`` (private unless ``public``)."""
    if upload is None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo, repo_type="dataset", private=not public, exist_ok=True)
        upload = api.upload_folder
    upload(folder_path=str(stage), repo_id=repo, repo_type="dataset",
           commit_message=f"ORIGIN-T-Bench {stage.name.rsplit('-v', 1)[-1]}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--bench", default="bench")
    p.add_argument("--out", default="dist")
    p.add_argument("--push", metavar="REPO", help="Hub dataset repo, e.g. <user>/origin-t-bench")
    p.add_argument("--public", action="store_true", help="create the Hub repo public (default private)")
    args = p.parse_args(argv)
    stage = package(Path(args.bench), Path(args.out))
    n = sum(1 for f in stage.rglob("*.jsonl"))
    print(f"verified and packaged {n} files -> {stage} (+ .tar.gz, SHA256SUMS)")
    if args.push:
        push(stage, args.push, public=args.public)
        print(f"uploaded to https://huggingface.co/datasets/{args.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
