"""Log prompt-eval runs to a local MLflow store.

MLflow 3.x no longer supports the plain mlruns/ folder store, so runs go
into a local SQLite database (mlflow.db, gitignored). A small JSON summary
is also exported to data/eval/ so the results stay visible in the repo
(see ab_runner.py).

Browse the runs with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from __future__ import annotations

from src.config import PROJECT_ROOT

DEFAULT_EXPERIMENT = "prompt-regression"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"


def log_run(
    prompt_version: str,
    params: dict,
    metrics: dict,
    experiment: str = DEFAULT_EXPERIMENT,
    tracking_uri: str = DEFAULT_TRACKING_URI,
) -> str:
    """Log one prompt-eval run and return its MLflow run id."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"prompt_{prompt_version}") as run:
        mlflow.set_tag("prompt_version", prompt_version)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        return run.info.run_id
