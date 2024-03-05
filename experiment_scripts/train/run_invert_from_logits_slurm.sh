#!/bin/bash
#SBATCH --gres=gpu:a6000:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue

export BASEDIR="/home/mattfinlays/vec2text/vec2text"
export NCCL_P2P_LEVEL=NVL

echo "hostname:"
hostname
echo "nvidia-smi"
nvidia-smi
echo "running df -h TMPDIR ($TMPDIR).."
df -h $TMPDIR
echo "start:: running command with BASEDIR=$BASEDIR"
cd $BASEDIR
pwd
torchrun run.py \
  --per_device_train_batch_size 256 \
  --per_device_eval_batch_size 256 \
  --max_seq_length 16 \
  --num_train_epochs 40 \
  --max_eval_samples 500 \
  --eval_steps 25000 \
  --warmup_steps 100000 \
  --learning_rate 0.0002 \
  --ddp_find_unused_parameters 1 \
  --dataset_name one_million_instructions \
  --model_name_or_path t5-small \
  --use_wandb=1 \
  --experiment inversion_from_logits \
  --bf16=1 \
  --lr_scheduler_type constant_with_warmup \
  --use_frozen_embeddings_as_input 1 \
  --mock_embedder 0 \
  --use_wandb 1 \
  --use_less_data 1000000 \
  --embedder_model_name gpt2 \
  --exp_group_name 2023-09-24-ablation


echo "finished:: running command with BASEDIR=$BASEDIR"
