"""The merged paper draft (T3, #23) is built from the section files and kept current."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _build():
    spec = importlib.util.spec_from_file_location("paper_build", ROOT / "docs/paper/build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_draft_is_current_with_the_section_files():
    assert _build().main(["--check"]) == 0, "run: python docs/paper/build.py"


def test_draft_has_every_chapter_once_and_no_working_lists():
    text = _build().build()
    heads = [line for line in text.splitlines() if line.startswith("## ")]
    assert heads[0] == "## Abstract" and heads[-1] == "## References"
    assert [h.split(" ", 2)[2] for h in heads[1:3]] == ["Introduction", "Related work"]
    assert sum("Track" in h for h in heads) == 3
    assert "Open items" not in text and "Figures and tables for the merged draft" not in text
    assert "draft for T3" not in "\n".join(heads)
