#!/usr/bin/env bash
# Multi-turn GRPO with tool-call code execution — 8 GPUs, Qwen3-1.7B.
#
# The model receives a coding problem, calls the `code_interpreter` tool to
# test its solution (LocalSandboxTool → subprocess), iterates up to MAX_TURNS
# times, then earns a final reward = fraction of all test cases passed.
#
# All multi-turn infrastructure is upstream verl (tool_agent_loop, AgentLoopManager,
# ToolParser/Hermes, MultiTurnConfig).  New files on this branch:
#   verl/tools/local_sandbox_tool.py          BaseTool subclass, subprocess exec
#   tool_config/local_sandbox_tool_config.yaml tool schema + config
#   examples/data_preprocess/verifiable_code_multiturn.py  dataset
#   scripts/local_sandbox_reward.py            final-episode reward scorer
#
# Data setup (run once):
#   python examples/data_preprocess/verifiable_code_multiturn.py \
#       --local_save_dir ~/z00918512/data/code_multiturn
#
# Usage:
#   bash scripts/run_multiturn_toolcall_8gpu_code.sh            # full run
#   TOTAL_TRAINING_STEPS=2 bash scripts/run_multiturn_toolcall_8gpu_code.sh

set -xeuo pipefail

HOME_DIR="/home/ubuntu/z00918512"
VERL_DIR="${HOME_DIR}/verl"

MODEL_PATH="${HOME_DIR}/models/Qwen3-1.7B"
EAGLE3_MODEL_PATH="${HOME_DIR}/models/Qwen3-1.7B_eagle3"
NUM_SPECULATIVE_TOKENS=3

TRAIN_BATCH_SIZE=64        # halved vs single-turn: each episode is ~4-8x longer
PPO_MINI_BATCH_SIZE=16
PPO_MICRO_BATCH_SIZE_PER_GPU=1
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=4096   # per-turn response cap (not total episode length)
MAX_TOOL_RESPONSE_LENGTH=512
MAX_TURNS=8
AGENT_NUM_WORKERS=8
GROUP_SIZE=5

MAX_TOKEN_LEN_PER_GPU=$((MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH))

NGPUS_PER_NODE=8
NNODES=1

TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-}

PROJECT_NAME="verl_eagle3_multiturn"
EXPERIMENT_NAME="qwen3_1.7b_multiturn_toolcall_eagle3_k3_$(date +%Y%m%d_%H%M)"
LOGS_DIR="${HOME_DIR}/logs"
LOG_FILE="${LOGS_DIR}/${EXPERIMENT_NAME}.log"
mkdir -p "${LOGS_DIR}"

SPEC_CFG="{method:eagle3,model:${EAGLE3_MODEL_PATH},num_speculative_tokens:${NUM_SPECULATIVE_TOKENS}}"

ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False

    "data.train_files=['${HOME_DIR}/data/code_multiturn/train.parquet']"
    "data.val_files=['${HOME_DIR}/data/code_multiturn/test.parquet']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
    "data.tool_config_path=${VERL_DIR}/tool_config/local_sandbox_tool_config.yaml"

    # Custom reward: extracts last code block from conversation, runs all test cases
    "custom_reward_function.path=${VERL_DIR}/scripts/local_sandbox_reward.py"
    custom_reward_function.name=compute_score

    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True

    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.clip_ratio=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.28
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.use_torch_compile=False

    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55
    actor_rollout_ref.rollout.n=${GROUP_SIZE}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.load_format=auto
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.max_model_len=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.disable_log_stats=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}

    # Multi-turn tool call configuration (upstream verl)
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS}
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=${MAX_TOOL_RESPONSE_LENGTH}
    "actor_rollout_ref.rollout.multi_turn.tool_config_path=${VERL_DIR}/tool_config/local_sandbox_tool_config.yaml"
    actor_rollout_ref.rollout.multi_turn.format=hermes

    # EAGLE3 speculative decoding (same as single-turn experiments)
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config=${SPEC_CFG}"

    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True

    trainer.balance_batch=True
    "trainer.logger=['console']"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=999
    trainer.test_freq=999
    trainer.total_epochs=1
    reward_model.reward_manager=naive
)

if [[ -n "${TOTAL_TRAINING_STEPS}" ]]; then
    ARGS+=(trainer.total_training_steps=${TOTAL_TRAINING_STEPS})
fi

echo "[run] ${EXPERIMENT_NAME}  log → ${LOG_FILE}"

PYTHON=${PYTHON:-/home/ubuntu/miniconda3/envs/online-drafter/bin/python}
CU13_LIBS=/home/ubuntu/miniconda3/envs/online-drafter/lib/python3.11/site-packages/nvidia/cu13/lib

cd /tmp

VLLM_LOGGING_LEVEL=INFO \
LD_LIBRARY_PATH="${CU13_LIBS}:${LD_LIBRARY_PATH:-}" \
VLLM_USE_DEEP_GEMM=0 \
"${PYTHON}" -m verl.trainer.main_ppo \
    "${ARGS[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
