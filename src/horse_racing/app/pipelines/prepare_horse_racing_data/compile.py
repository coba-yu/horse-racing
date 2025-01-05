from pathlib import Path

from kfp import compiler
from kfp import dsl

PIPELINE_NAME = "prepare-horse-racing-data"

GCP_PROJECT_ID = "yukob-horse-racing"  # TODO: 環境変数から取得する.


@dsl.component  # type: ignore
def add(a: int, b: int) -> int:
    return a + b


@dsl.pipeline(  # type: ignore
    name=PIPELINE_NAME,
    pipeline_root=f"gs://horse-racing-pipelines/{PIPELINE_NAME}/history",
)
def pipeline() -> int:
    task1 = add(a=1, b=5)
    return int(task1.output)


# Compile the pipeline

pipeline_dir = Path(__file__).parent
package_path = pipeline_dir / "pipeline.yaml"

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=str(package_path),
)
