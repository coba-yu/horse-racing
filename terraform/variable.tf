variable "google_project" {
  default = "yukob-horse-racing"
}

variable "gcp_iam_roles" {
  default = [
    "roles/cloudscheduler.admin",
    "roles/cloudfunctions.developer",
    "roles/pubsub.editor",
  ]
}

variable "region" {
  default = "asia-northeast1"
}
