terraform {
  backend "gcs" {
  }
  required_providers {
    google = {
      source = "hashicorp/google"
    }
  }
}

provider "google" {
  # project: GOOGLE_PROJECT を読み込んでいる
  # credentials: GOOGLE_CREDENTIALS を読み込んでいる
  region = "asia-northeast1"
}
