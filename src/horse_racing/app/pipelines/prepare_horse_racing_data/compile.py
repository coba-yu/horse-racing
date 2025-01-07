from pathlib import Path

from kfp import compiler, components, dsl

PIPELINE_NAME = "prepare-horse-racing-data"

GCP_PROJECT_ID = "yukob-horse-racing"  # TODO: 環境変数から取得する.


@dsl.pipeline(  # type: ignore
    name=PIPELINE_NAME,
    pipeline_root=f"gs://horse-racing-pipelines/{PIPELINE_NAME}/history",
)
def pipeline(year: int, month: int) -> None:
    pipelines_dir = Path(__file__).parent
    components_dir = pipelines_dir.parent / "components"

    scrape_netkeiba_yaml = components_dir / "scrape_netkeiba_race_results" / "component.yaml"
    scrape_netkeiba_op = components.load_component_from_file(scrape_netkeiba_yaml)  # noqa: F841


if __name__ == "__main__":
    # Compile the pipeline

    pipeline_dir = Path(__file__).parent
    package_path = pipeline_dir / "pipeline.yaml"

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=str(package_path),
    )
