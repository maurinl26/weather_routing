variable "project_id" {
  type        = string
  description = "Scaleway project ID (ou SCW_DEFAULT_PROJECT_ID)."
}

variable "region" {
  type    = string
  default = "fr-par"
}

variable "zone" {
  type    = string
  default = "fr-par-2" # L40S dispo en PAR-2
}

variable "instance_type" {
  type        = string
  default     = "L40S-1-48G" # 1× L40S, 48 GB VRAM — à VÉRIFIER selon dispo
  description = "Type GPU d'entraînement (L40S conseillé ; L4-1-24G si backbone gelé)."
}

variable "image" {
  type        = string
  default     = "ubuntu_jammy" # NOTE: privilégier une image GPU OS (drivers NVIDIA)
  description = "Slug d'image — utiliser l'image GPU OS Scaleway."
}

variable "root_volume_gb" {
  type    = number
  default = 150
}

variable "admin_ip_range" {
  type        = string
  default     = "0.0.0.0/0" # ⚠️ restreindre à votre IP (x.x.x.x/32)
  description = "Plage IP autorisée pour SSH."
}
