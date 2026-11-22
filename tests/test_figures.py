import pytest

pytest.importorskip("matplotlib")  # not in the CI install
pytest.importorskip("yaml")

from src.evaluation import figures  # noqa: E402


def test_every_paper_figure_is_drawn_from_the_committed_reports(tmp_path):
    figures.main(["--out", str(tmp_path)])
    for name in figures.FIGURES:
        png = tmp_path / f"{name}.png"
        assert png.is_file() and png.stat().st_size > 20_000, name
