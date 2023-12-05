#!/bin/sh

#SBATCH --job-name=train_event_model
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=ksu-gen-gpu.q
#SBATCH --gres=gpu:2

data_path="ball-data/fold${SLURM_ARRAY_TASK_ID}/ball-data.yaml"
model_name="ball-tracker-fold${SLURM_ARRAY_TASK_ID}"

module load Python/3.11.2-GCCcore-12.2.0-bare
cd datasets
conda run -n cis530 --no-capture-output yolo detect train data=$data_path model=yolov8m.pt epochs=100 imgsz=1080 batch=-1 device=0,1 name=$model_name