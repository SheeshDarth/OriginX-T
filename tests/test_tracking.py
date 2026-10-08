"""Tests for run logging."""

import pytest

from src.evaluation.tracking import Run, track_run


@pytest.mark.parametrize("cfg", [None, {}, {"backend": "none"}])
def test_no_backend_logs_nothing_but_still_works(cfg):
    with track_run(cfg, "run", {"a": 1}) as run:
        assert isinstance(run, Run)
        run.log_metrics({"x": 1.0}, step=0)
        run.log_artifact("whatever")  # discarded, not even opened


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="wandb"):
        with track_run({"backend": "wandb"}, "run", {}):
            pass


def test_mlflow_run_records_params_tags_and_stepped_metrics(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    cfg = {"backend": "mlflow", "uri": f"sqlite:///{tmp_path}/sub/mlflow.db", "experiment": "t"}

    with track_run(cfg, "r1", {"seed": 3, "lr": 0.1}, {"kind": "test"}) as run:
        run.log_metrics({"ppl": 50.0}, step=0)
        run.log_metrics({"ppl": 60.0}, step=1)

    runs = mlflow.search_runs(experiment_names=["t"])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["tags.mlflow.runName"] == "r1"
    assert row["params.seed"] == "3"
    assert row["tags.kind"] == "test"
    assert row["metrics.ppl"] == 60.0  # latest step
