#!/bin/bash -l
#SBATCH -J train_64K
#SBATCH -N 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:8
#SBATCH --mem=400G
#SBATCH -c 32

# !!!! Load your own environment here !!!! #
# !!!! Load your own environment here !!!! #

# Fine-tune from this model 
model=${MODEL:-princeton-nlp/Llama-3-8B-ProLong-512k-Base}
# Point to the base dir of the ProLong 64K data
dataset=${DATASET:-"datasets"}

# Directories in the dataset root folder where @ is followed by the mixing proportion 
domains=(
    prolong-ultrachat-64K@1.0
)
domains_name=ultrachat


bsz=${BSZ:-64} # * 64k (seq len) = 4M
seq=${SEQ:-1} # per-device batch size
lr=${LR:-2e-5}
steps=${STEPS:-250}
save_steps=${SAVE:-250}
warmup=${WARMUP:-0.05}
suffix=${SUFFIX:-""} # for model saving name


run_name="sft_$(basename $model)_$(basename $dataset)_${domains_name}_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}${suffix}"
out_dir="checkpoints/$run_name"

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(jq -n "[$CUDA_VISIBLE_DEVICES] | length")
fi
num_gpus=${NUM_GPUS:-$num_gpus}

num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

# Gradient accumulation
accu=$(($bsz / $seq / $num_gpus / $num_nodes))


# [0] Disable
# [1] FULL_SHARD (shards optimizer states, gradients and parameters),
# [2] SHARD_GRAD_OP (shards optimizer states and gradients),
# [3] NO_SHARD (DDP),
# [4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node while each node has full copy),
# [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has full copy). For more information, please refer the official PyTorch docs.
fsdp=${FSDP:-"1"}
gc=${GC:-"1"}

export LOGIT_BLOCK_SIZE=2048  # Compute Llama logits in blocks of 2048 tokens

mkdir -p $out_dir
nvidia-smi

if [ $num_nodes -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR:-$master_addr}

    # Launch via srun
    header="srun torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$master_addr:56321 \
    --nnodes=$num_nodes \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
 
    # Launch without srun
    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"
fi
echo "slurm_nodelist=${SLURM_NODELIST} num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

export OMP_NUM_THREADS=$num_gpus
export WANDB_PROJECT="prolong"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline" # We turn off wandb online sync by default
export TOKENIZERS_PARALLELISM=true


base_arguments=(
    --report_to wandb
    --do_train

    --model_name $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --config_overrides_json "$overrides"
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq

    --bf16
    --learning_rate $lr
    --min_lr_ratio 0.1
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio $warmup
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps $steps
    --save_steps $save_steps
    --dataloader_num_workers 1

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --per_device_max_tokens 65536

    --cuda_empty_cache

    --apply_instruct_masks # mask out the tokens from instructions (instead of responses) when calculating losses
    --token_scaled_loss # average losses over valid training tokens instead of devices
)



if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp 
    base_arguments+=( --fsdp "auto_wrap" )
    # [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

if [ $gc -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
fi

base_arguments+=( --tokenized_mds_train )
for domain in "${domains[@]}"; do
    base_arguments+=( $dataset/$domain )
done

base_arguments+=( $@ )

echo command: "${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out
