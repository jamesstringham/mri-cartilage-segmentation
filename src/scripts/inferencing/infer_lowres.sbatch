#!/bin/bash
#SBATCH --job-name=lowres_infer_dice
#SBATCH --partition=ondemand-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/lowres_infer_dice_%j.out
#SBATCH --error=logs/lowres_infer_dice_%j.err

set -e

cd ${SLURM_SUBMIT_DIR}
mkdir -p logs predictions

module load pytorch_2.4.1_tensorflow_2.17_cuda_12.4

echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
nvidia-smi

# Run inference exactly through current standalone inference script
python -m src.inference.infer_knee \
  --ckpt "path/to/vnet_model_best.pth.tar" \
  --im "path/to/image.im" \
  --seg "path/to/segmentation.seg" \
  --out "path/to/prediction.npy" \
  --device cuda \
  --profile-memory