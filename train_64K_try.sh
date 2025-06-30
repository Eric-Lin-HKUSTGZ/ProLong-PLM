#!/bin/bash -l
#SBATCH -J train_64K
#SBATCH -N 1
#SBATCH --output=slurm/%x-%j.out
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH -c 32

# !!!! Load your own environment here !!!! #
# !!!! Load your own environment here !!!! #

# load cuda
module load cuda/12.4
# activate conda envs
source /data/user/qxiao183/xqf/miniconda3/envs/quanllm/bin/activate

# solute OOM Problem - Conservative memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8
export CUDA_LAUNCH_BLOCKING=1
export FSDP_SHARING_STRATEGY="FULL_SHARE"
export TORCH_FSDP_ENABLE_HYBRID_SHARE=False
# Additional memory optimization
export CUDA_MODULE_LOADING=LAZY
export TORCH_NCCL_AVOID_RECORD_STREAMS=1

#Fine-tune from this model 
model=${MODEL:-/data/user/qxiao183/lwq_llm/ProLong-main/Meta-Llama-3-8B-Instruct}
# Point to the base dir of the ProLong 64K data
dataset=${DATASET:-"datasets/long-context-65536"}

# Directories in the dataset root folder where @ is followed by the mixing proportion 
domains=(
    thestackv1_concat_by_repo-65536@0.3
    book-65536@0.3
    fineweb-edu@0.1
    fineweb-2023-50@0.1
    stackexchange@0.04
    dolmawiki@0.04
    tuluv2@0.03
    arxiv@0.03
    openwebmath@0.03
    textbooks@0.03
)
domains_name=ProLong64KMix

bsz=${BSZ:-64} # * 64k (seq len) = 4M - Keep original for experiment reproduction
seq=${SEQ:-1} # per-device batch size
lr=${LR:-1e-5}
steps=${STEPS:-5000}
save_steps=${SAVE:-125}  # Increase save interval to reduce memory pressure
warmup=${WARMUP:-0.1}
suffix=${SUFFIX:-""} # for model saving name


run_name="lcft_$(basename $model)_$(basename $dataset)_${domains_name}_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}${suffix}"
out_dir="checkpoints/$run_name"

# 第一步：解决GPU数量问题 ----------------------------
# 尝试从SLURM环境变量获取GPU数量
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    num_gpus=$SLURM_GPUS_ON_NODE
    echo "Using SLURM_GPUS_ON_NODE: $num_gpus"
elif [ -n "$SLURM_JOB_GPUS" ]; then
    num_gpus=$(echo $SLURM_JOB_GPUS | awk -F, '{print NF}')
    echo "Using SLURM_JOB_GPUS: $num_gpus"
else
    # SLURM变量不可用，使用备选方案
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # 使用纯Bash方法解析CUDA_VISIBLE_DEVICES
        IFS=',' read -ra gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
        num_gpus=${#gpu_ids[@]}
        echo "Using CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES -> $num_gpus GPUs"
    else
        # 最后尝试通过nvidia-smi获取
        num_gpus=$(nvidia-smi -L | wc -l)
        echo "Using nvidia-smi: $num_gpus GPUs"
    fi
fi

# 确保num_gpus有效
if [ -z "$num_gpus" ] || [ "$num_gpus" -eq 0 ]; then
    echo "ERROR: Failed to determine GPU count! Aborting."
    exit 1
fi

num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

# Gradient accumulation - Force higher accumulation for memory efficiency
accu=$(($bsz / $seq / $num_gpus / $num_nodes))
# 确保梯度积累步数至少为2，强制减少每次的内存使用
if [ $accu -lt 2 ]; then
    accu=2
    echo "WARNING: Forced gradient accumulation to $accu for memory efficiency"
fi

# [0] Disable
# [1] FULL_SHARD (shards optimizer states, gradients and parameters),
# [2] SHARD_GRAD_OP (shards optimizer states and gradients),
# [3] NO_SHARD (DDP),
# [4] HYBRID_SHARD (shards optimizer states, gradients and parameters within each node while each node has full copy),
# [5] HYBRID_SHARD_ZERO2 (shards optimizer states and gradients within each node while each node has full copy). For more information, please refer the official PyTorch docs.
fsdp=${FSDP:-"1"}  # Use FULL_SHARD for maximum memory savings
gc=${GC:-"1"}  # Force enable gradient checkpointing

# 第二步：解决OMP_NUM_THREADS问题 ----------------------
# 计算合理的OMP_NUM_THREADS值
cpus_per_task=${SLURM_CPUS_PER_TASK:-8}  # 默认8个CPU线程
threads_per_gpu=$(( cpus_per_task / num_gpus ))
if [ $threads_per_gpu -lt 1 ]; then
    threads_per_gpu=1
fi
export OMP_NUM_THREADS=$threads_per_gpu
echo "Setting OMP_NUM_THREADS: $OMP_NUM_THREADS (cpus_per_task=$cpus_per_task, num_gpus=$num_gpus)"

# 第三步：设置其他环境变量 ----------------------------
export LOGIT_BLOCK_SIZE=1024  # Reduced from 2048 to save memory
export WANDB_PROJECT="prolong"
export WANDB_DIR=$out_dir
export WANDB_MODE="offline" # We turn off wandb online sync by default
export TOKENIZERS_PARALLELISM=false  # Avoid memory overhead from parallel tokenization

mkdir -p $out_dir
nvidia-smi > $out_dir/nvidia-smi.log

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
    echo "Multi-node setup with $num_nodes nodes ($num_gpus GPUs per node)"
else
    # 获取可用的随机端口
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    [ -z "$master_port" ] && master_port=51058  # 如果失败使用默认端口
    
    # 确保--nproc-per-node被正确设置
    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.train_language_model"
    echo "Single-node setup with $num_gpus GPUs"
fi
echo "slurm_nodelist=${SLURM_NODELIST} num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

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
    --dataloader_num_workers 0  # Reduce to 0 to save memory

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --per_device_max_tokens 65536  # Keep original for experiment reproduction
    # Emergency fallback: uncomment the line below if still getting OOM
    # --per_device_max_tokens 32768  # Reduced for memory constraints

    # --torch_compile
    --cuda_empty_cache
    --config_overrides "rope_theta=8000000"
    
    # Memory optimization arguments without changing core experiment parameters
    --dataloader_pin_memory false
    --save_safetensors true
    --eval_accumulation_steps 1  # Minimize eval memory usage
)

# 打印重要参数验证
echo "================= CONFIGURATION ================="
echo "GPUs per node: $num_gpus"
echo "Nodes: $num_nodes"
echo "Gradient accumulation steps: $accu"
echo "Total batch size: $((seq * num_gpus * num_nodes * accu)) (seq=$seq, GPUs=$num_gpus, nodes=$num_nodes, accu=$accu)"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "================================================"

if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp 
    base_arguments+=( --fsdp "auto_wrap" )
    # [1] FULL_STATE_DICT, [2] LOCAL_STATE_DICT, [3] SHARDED_STATE_DICT
    export FSDP_STATE_DICT_TYPE="SHARDED_STATE_DICT"  # Use sharded for memory efficiency
    export FSDP_CPU_RAM_EFFICIENT_LOADING="true"  # Efficient loading
    # Conservative CPU offloading for memory savings
    export FSDP_OFFLOAD_PARAMS="true"  # Enable parameter offloading
    export FSDP_BACKWARD_PREFETCH="BACKWARD_PRE"  # Prefetch for efficiency
    echo "Using FSDP with strategy $fsdp, parameter offloading, and conservative memory settings"
fi

if [ $gc -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
    echo "Using gradient checkpointing"
fi

base_arguments+=( --tokenized_mds_train )
for domain in "${domains[@]}"; do
    base_arguments+=( $dataset/$domain )
    echo "Dataset domain: $domain"
done

base_arguments+=( $@ )

echo "Full command: ${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out

echo "Job completed with exit status: $?"
