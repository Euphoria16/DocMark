#!/bin/bash
unset OMPI_COMM_WORLD_LOCAL_RANK
task_index=$OMPI_COMM_WORLD_RANK

MASTER=$(head -n 1 /job/hostfile | awk '{print $1}')
echo $MASTER
hostgpus=$(nvidia-smi -L|grep GPU|wc -l)

exp=InternVL2-2B-pretrain
model_path=OpenGVLab/InternVL2-2B
max_num=12


exp_name="$exp"
mkdir -p output/"$exp_name"

MAX_NUM=$max_num \
NNODES=$(wc -l < /job/hostfile) \
NODE_RANK=$task_index \
MASTER_ADDR=$MASTER \
NPROC_PER_NODE=$hostgpus \
swift sft \
  --model "$model_path" \
  --dataset data/docmark_pile.jsonl \
  --train_type full \
  --freeze_vit True \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --gradient_accumulation_steps 1 \
  --eval_steps 500 \
  --save_steps 2000 \
  --save_total_limit 5 \
  --logging_steps 5 \
  --max_length 8192 \
  --output_dir output/"$exp_name" \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4 \
  --dataset_num_proc 4 \
  --save_only_model False \
  --lr_scheduler_type cosine \
  --split_dataset_ratio 0 \
  --ddp_timeout 86400 \
  --deepspeed zero3_offload \
  --add_version False \
  2>&1 | tee -a output/"$exp_name"/output.log
