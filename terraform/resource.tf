#==========================#
# IAM with Service Account #
#==========================#

data "google_service_account" "yukob_horse_racing" {
  account_id = "yukob-horse-racing"
}

resource "google_project_iam_member" "pubsub_editor" {
  project = var.google_project
  role    = "roles/pubsub.editor"
  member  = "serviceAccount:${data.google_service_account.yukob_horse_racing.email}"
}

#==================#
# data preparation #
#==================#

resource "google_pubsub_topic" "horse_racing_data" {
  name       = "horse-racing-data"
  depends_on = [google_project_iam_member.pubsub_editor]
}

resource "google_cloud_scheduler_job" "horse_racing_data" {
  name     = "horse-racing-data"
  schedule = "*/20 * * * *"

  pubsub_target {
    topic_name = google_pubsub_topic.horse_racing_data.id
    data       = base64encode("horse racing data")
  }
  depends_on = [google_pubsub_topic.horse_racing_data]
}
