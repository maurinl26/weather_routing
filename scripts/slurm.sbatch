#!/bin/bash
#SBATCH --job-name=wxr-finetune
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # = devices par nœud
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --partition=gpu_p4
#SBATCH --qos=qos_gpu-t3
# #SBATCH --account=xxxx@gpu        # à remplir

set -euo pipefail
mkdir -p logs

module purge
module load cuda/12.4 || true
module load python/3.11 || true

source .venv/bin/activate

# Variables Lightning DDP
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=INFO

srun uv run wxr-train \
    experiment=${EXP:-finetune_bog} \
    cluster=slurm \
    cluster.devices=${SLURM_NTASKS_PER_NODE} \
    cluster.num_nodes=${SLURM_JOB_NUM_NODES}
