"""Run logging (CONTRIBUTING.md Definition of Done: "run logged to MLflow/W&B").

One ``track_run`` per training run. The backend is chosen in the config, so
experiment code never imports MLflow and tests never need it:

    tracking:
      backend: mlflow                          # or "none" to log nothing
      uri: sqlite:///mlruns/mlflow.db          # local, offline, git-ignored
      experiment: origin-t-grid

Offline here means a local SQLite file, no server. MLflow's plain-directory
store is in maintenance mode and recent versions refuse to open it, so the
default is SQLite; view it with ``mlflow ui --backend-store-uri <uri>``.
``mlflow`` is imported only when the backend is used.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

DEFAULT_URI = "sqlite:///mlruns/mlflow.db"


class Run:
    """What a training run can log. This base class discards everything."""

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        pass

    def log_artifact(self, path: str | Path) -> None:
        pass


class _MlflowRun(Run):
    def __init__(self, mlflow: Any) -> None:
        self._mlflow = mlflow

    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        self._mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: str | Path) -> None:
        self._mlflow.log_artifact(str(path))


def _prepare_sqlite(uri: str) -> None:
    """SQLite will not create the directory it lives in."""
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        Path(uri[len(prefix):]).parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def track_run(
    cfg: Optional[Mapping[str, Any]],
    name: str,
    params: Mapping[str, Any],
    tags: Optional[Mapping[str, str]] = None,
) -> Iterator[Run]:
    """Open a tracked run for the length of the ``with`` block.

    ``cfg`` is the config's ``tracking`` section; ``None`` or ``backend: none``
    logs nothing. ``params`` are the settings that define the run (they show up
    as columns when runs are compared); ``tags`` label it for filtering.
    """
    backend = (cfg or {}).get("backend", "none")
    if backend == "none":
        yield Run()
        return
    if backend != "mlflow":
        raise ValueError(f"tracking.backend must be 'mlflow' or 'none', got {backend!r}")

    import mlflow  # heavy import, only when used

    uri = cfg.get("uri", DEFAULT_URI)
    _prepare_sqlite(uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.get("experiment", "origin-t"))
    with mlflow.start_run(run_name=name):
        mlflow.log_params(dict(params))
        if tags:
            mlflow.set_tags(dict(tags))
        yield _MlflowRun(mlflow)
