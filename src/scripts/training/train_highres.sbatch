#!/bin/bash
#SBATCH --job-name=vnet_7t
#SBATCH --output=logs/vnet_7t_%j.out
#SBATCH --error=logs/vnet_7t_%j.err
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

set -euo pipefail

# Create log directory if it does not exist
mkdir -p logs

# Go to project root
cd "${SLURM_SUBMIT_DIR}"

echo "Job started on $(hostname)"
echo "Working directory: $(pwd)"
echo "CUDA devices: ${CUDA_VISIBLE_DEVICES:-not set}"


module purge
module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pip install --user h5py setproctitle matplotlib

python -m src.training.train_7t \
  --data path/to/datafile.hdf5 \
  --pretrained path/to/lowres/vnet_model_best.pth.tar \
  --save path/to/save \
  --nEpochs 500 \
  --batchSz 1 \
  --ngpu 1