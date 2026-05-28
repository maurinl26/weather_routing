# Box d'entraînement GPU Scaleway (L40S), ÉPHÉMÈRE et à la demande.
# Auth : SCW_ACCESS_KEY, SCW_SECRET_KEY, SCW_DEFAULT_PROJECT_ID.
#
# Flux : terraform apply → SSH → scripts/setup_training.sh → wxr-train →
# scripts/push_checkpoint.sh → terraform destroy. NE PAS laisser tourner (coût GPU).
#
# NOTE : scaffold non appliqué. Vérifier slug d'image GPU OS + type avant apply.

terraform {
  required_version = ">= 1.6"
  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.40"
    }
  }
}

provider "scaleway" {
  zone       = var.zone
  region     = var.region
  project_id = var.project_id
}

resource "scaleway_instance_ip" "train" {}

resource "scaleway_instance_security_group" "train" {
  name                    = "wxrouting-train"
  inbound_default_policy  = "drop"
  outbound_default_policy = "accept"
  inbound_rule {
    action   = "accept"
    port     = 22
    ip_range = var.admin_ip_range
  }
}

resource "scaleway_instance_server" "train" {
  name              = "wxrouting-train"
  type              = var.instance_type # ex. L40S-1-48G
  image             = var.image         # image GPU OS (drivers NVIDIA)
  ip_id             = scaleway_instance_ip.train.id
  security_group_id = scaleway_instance_security_group.train.id
  tags              = ["wxrouting", "training", "gpu", "ephemeral"]

  root_volume {
    size_in_gb = var.root_volume_gb # modèle + cache ERA5 + checkpoints
  }

  user_data = {
    cloud-init = file("${path.module}/cloud-init.yaml")
  }
}
