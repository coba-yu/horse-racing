variable "google_project" {
  default = "yukob-horse-racing"
}

variable "gcp_iam_roles" {
  default = [
    "roles/aiplatform.user",
    "roles/cloudscheduler.admin",
    "roles/cloudfunctions.developer",
    "roles/iam.serviceAccountUser", # IAMから手動で追加する必要がある.
    "roles/pubsub.editor",
    "roles/resourcemanager.projectIamAdmin",
    "roles/storage.admin",
  ]
}

variable "region" {
  default = "asia-northeast1"
}

variable "service_account_email" {
  default = "yukob-horse-racing@yukob-horse-racing.iam.gserviceaccount.com"
}
