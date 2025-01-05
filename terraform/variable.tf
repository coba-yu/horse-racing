variable "google_project" {
  default = "yukob-horse-racing"
}

variable "gcp_iam_roles" {
  default = [
    "roles/cloudscheduler.admin",
    "roles/cloudfunctions.developer",
    "roles/pubsub.editor",
    "roles/resourcemanager.projectIamAdmin",
  ]
}

variable "region" {
  default = "asia-northeast1"
}
