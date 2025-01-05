# Enable Pub/Sub API
resource "google_project_service" "pubsub" {
  service            = "pubsub.googleapis.com"
  disable_on_destroy = false
}

resource "google_pubsub_topic" "horse_racing_data" {
  name       = "horse-racing-data"
  depends_on = [google_project_service.pubsub]
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
