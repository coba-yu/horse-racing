import base64
import json
import os
from typing import Any

import functions_framework
from cloudevents.http import CloudEvent
from google.cloud import aiplatform

REGION = os.environ["REGION"]


@functions_framework.cloud_event  # type: ignore
def subscribe(cloud_event: CloudEvent) -> None:
    # decode the event payload string
    payload_message = base64.b64decode(cloud_event.data["message"]["data"]).decode()
    # parse payload string into JSON object
    payload_json = json.loads(payload_message)
    # trigger pipeline run with payload
    trigger_pipeline_run(payload_json)


def trigger_pipeline_run(payload_json: dict[str, Any]) -> None:
    pipeline_display_name = payload_json["pipeline_display_name"]
    pipeline_spec_uri = payload_json["pipeline_spec_uri"]
    pipeline_root = payload_json["pipeline_root"]
    parameter_values = payload_json.get("parameter_values", {})

    year = 2024
    month = 12
    parameter_values["year"] = year
    parameter_values["month"] = month

    # Create a PipelineJob using the compiled pipeline from pipeline_spec_uri
    aiplatform.init(location=REGION)
    job = aiplatform.PipelineJob(
        display_name=pipeline_display_name,
        template_path=pipeline_spec_uri,
        pipeline_root=pipeline_root,
        enable_caching=False,
        parameter_values=parameter_values,
    )

    # Submit the PipelineJob
    job.submit()
