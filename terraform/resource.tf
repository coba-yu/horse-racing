#==========================#
# IAM with Service Account #
#==========================#

data "google_service_account" "yukob_horse_racing" {
  account_id = "yukob-horse-racing"
}

resource "google_project_iam_member" "gcp_iam_member" {
  for_each = toset(var.gcp_iam_roles)
  project  = var.google_project
  role     = each.value
  member   = "serviceAccount:${data.google_service_account.yukob_horse_racing.email}"
}

resource "google_service_account" "yukob_horse_racing_job" {
  account_id                   = "yukob-horse-racing-job"
  create_ignore_already_exists = true
}

resource "google_project_iam_member" "yukob_horse_racing_job" {
  for_each = toset(var.gcp_iam_roles__yukob_horse_racing_job)
  project  = var.google_project
  role     = each.value
  member   = "serviceAccount:${google_service_account.yukob_horse_racing_job.email}"
}

#==================#
# data preparation #
#==================#

resource "google_pubsub_topic" "horse_racing_data" {
  name       = "horse-racing-data"
  depends_on = [google_project_iam_member.gcp_iam_member]
}

resource "google_cloud_scheduler_job" "horse_racing_data" {
  name     = "horse-racing-data"
  schedule = "0 22 * * *"

  pubsub_target {
    topic_name = google_pubsub_topic.horse_racing_data.id
    data = base64encode(
      jsonencode({
        message = "trigger scheduled run"
      })
    )
  }
  depends_on = [
    google_project_iam_member.gcp_iam_member,
    google_pubsub_topic.horse_racing_data,
  ]
}

resource "google_cloud_run_v2_job" "scrape_netkeiba" {
  location = var.region
  name     = "scrape-netkeiba-job"
  template {
    template {
      containers {
        image   = "${var.region}-docker.pkg.dev/${var.google_project}/${var.gcp_artifact_repository_name}/${var.gcp_horse_racing_image_name}:latest"
        command = ["python", "src/horse_racing/app/runs/jobs/scrape_netkeiba.py"]
      }
      service_account = "${google_service_account.yukob_horse_racing_job.account_id}@${var.google_project}.iam.gserviceaccount.com"
      timeout         = "21600s"
    }
  }
  depends_on = [
    google_service_account.yukob_horse_racing_job,
  ]
}

# data "archive_file" "gcf_src_scheduled_pipeline" {
#   type        = "zip"
#   source_dir  = "${path.module}/../src/horse_racing/app/functions/scheduled_pipeline"
#   output_path = "${path.module}/../src/horse_racing/app/functions/scheduled_pipeline.zip"
# }
#
# resource "google_storage_bucket_object" "scheduled_pipeline_function_src" {
#   name       = "functions/scheduled_pipeline/${data.archive_file.gcf_src_scheduled_pipeline.output_md5}.zip"
#   bucket     = data.google_storage_bucket.horse_racing_pipelines.name
#   source     = data.archive_file.gcf_src_scheduled_pipeline.output_path
#   depends_on = [google_project_iam_member.gcp_iam_member]
# }
#
# resource "google_cloudfunctions2_function" "scheduled_pipeline_function" {
#   name     = "scheduled-pipeline-horse-racing-data"
#   location = var.region
#
#   build_config {
#     runtime     = "python312"
#     entry_point = "subscribe"
#     source {
#       storage_source {
#         bucket = data.google_storage_bucket.horse_racing_pipelines.name
#         object = google_storage_bucket_object.scheduled_pipeline_function_src.name
#       }
#     }
#   }
#
#   service_config {
#     timeout_seconds  = 540
#     available_memory = "1Gi"
#     available_cpu    = "1"
#     environment_variables = {
#       REGION = var.region
#     }
#     service_account_email = var.service_account_email
#   }
#
#   event_trigger {
#     event_type   = "google.cloud.pubsub.topic.v1.messagePublished"
#     pubsub_topic = google_pubsub_topic.horse_racing_data.id
#     retry_policy = "RETRY_POLICY_DO_NOT_RETRY"
#   }
#
#   depends_on = [
#     google_project_iam_member.gcp_iam_member,
#     google_storage_bucket_object.scheduled_pipeline_function_src,
#     google_pubsub_topic.horse_racing_data,
#   ]
# }
