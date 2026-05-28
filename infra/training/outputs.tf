output "public_ip" {
  value = scaleway_instance_ip.train.address
}

output "ssh" {
  description = "Connexion SSH à la box d'entraînement."
  value       = "ssh root@${scaleway_instance_ip.train.address}"
}
