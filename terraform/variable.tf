variable "google_project" {
  default = "yukob-horse-racing"
}

variable "gcp_iam_roles" {
  default = [
    "roles/aiplatform.user",
    "roles/cloudscheduler.admin",
    "roles/cloudfunctions.developer",
    "roles/iam.serviceAccountAdmin", # IAMから手動で追加する必要がある.
    "roles/pubsub.editor",
    "roles/resourcemanager.projectIamAdmin",
    "roles/storage.admin",
  ]
}

variable "gcp_iam_roles__yukob_horse_racing_job" {
  default = [
    # https://cloud.google.com/storage/docs/access-control/iam-roles
    "roles/storage.objectUser",
  ]
}

variable "region" {
  default = "asia-northeast1"
}

variable "service_account_email" {
  default = "yukob-horse-racing@yukob-horse-racing.iam.gserviceaccount.com"
}
