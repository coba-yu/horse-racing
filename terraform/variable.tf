variable "google_project" {
  default = "yukob-horse-racing"
}

variable "gcp_iam_roles" {
  default = [
    "roles/aiplatform.user",
    "roles/cloudscheduler.admin",
    "roles/cloudfunctions.developer",
    "roles/iam.serviceAccountAdmin", # IAMから手動で追加する必要がある.
    "roles/iam.serviceAccountUser",
    "roles/pubsub.editor",
    "roles/resourcemanager.projectIamAdmin",
    "roles/storage.admin",
    # https://cloud.google.com/workflows/docs/access-control
    "roles/workflows.editor",
  ]
}

variable "gcp_iam_roles__yukob_horse_racing_job" {
  default = [
    # https://cloud.google.com/storage/docs/access-control/iam-roles
    "roles/storage.admin",
    "roles/iam.serviceAccountUser",
  ]
}

variable "gcp_iam_roles__yukob_horse_racing_workflow" {
  default = [
    # https://cloud.google.com/run/docs/reference/iam/roles
    "roles/run.developer",
    "roles/iam.serviceAccountUser",
  ]
}

variable "region" {
  default = "asia-northeast1"
}

variable "service_account_email" {
  default = "yukob-horse-racing@yukob-horse-racing.iam.gserviceaccount.com"
}

variable "gcp_artifact_repository_name" {
  default = "horse-racing"
}

variable "gcp_horse_racing_image_name" {
  default = "horse-racing"
}
