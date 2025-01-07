from pathlib import Path

from kfp import compiler
from kfp import dsl

PIPELINE_NAME = "prepare-horse-racing-data"

GCP_PROJECT_ID = "yukob-horse-racing"  # TODO: 環境変数から取得する.


@dsl.component  # type: ignore
def scrape_and_save_race_results(year: int, month: int) -> str:
    return f"{year=}, {month=}"


@dsl.pipeline(  # type: ignore
    name=PIPELINE_NAME,
    pipeline_root=f"gs://horse-racing-pipelines/{PIPELINE_NAME}/history",
)
def pipeline(year: int, month: int) -> None:
    task1 = scrape_and_save_race_results(year=year, month=month)  # noqa: F841


if __name__ == "__main__":
    # Compile the pipeline

    pipeline_dir = Path(__file__).parent
    package_path = pipeline_dir / "pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=str(package_path),
    )
