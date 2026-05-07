#!/usr/bin/env bash
# GRPO | Qwen3-1.7B + EAGLE3 online drafter | FSDP training | NVIDIA GPUs
#
# This script demonstrates end-to-end online EAGLE3 drafter training alongside
# GRPO policy optimisation on Qwen3-1.7B.
#
# What it does:
#   1. Runs rollout with EAGLE3 speculative decoding active in vLLM.
#   2. Captures EAGLE3 aux hidden states during the actor's compute_log_prob pass.
#   3. Trains the EAGLE3 draft model online every N RL steps via CE distillation.
#   4. Hot-swaps the updated draft weights back into vLLM via AsyncLLM.update_draft_weights.
#
# Prerequisites:
#   huggingface-cli download Qwen/Qwen3-1.7B --local-dir $HOME/models/Qwen3-1.7B
#   huggingface-cli download AngelSlim/Qwen3-1.7B_eagle3 --local-dir $HOME/models/Qwen3-1.7B_eagle3
#
# Quick smoke-test (single GPU, tiny batch):
#   NGPUS_PER_NODE=1 ROLLOUT_TP=1 TRAIN_BATCH_SIZE=32 MAX_RESPONSE_LENGTH=256 \
#     bash run_qwen3_1_7b_eagle3_online_drafter.sh

set -xeuo pipefail

HOME="/home/ubuntu/z00918512"
EAGLE3_ENABLE=${EAGLE3_ENABLE:-True}
ONLINE_DRAFTER_ENABLE=${ONLINE_DRAFTER_ENABLE:-True}
########################### user-adjustable ###################################
MODEL_PATH=${MODEL_PATH:-"$HOME/models/Qwen3-1.7B"}
EAGLE3_MODEL_PATH=${EAGLE3_MODEL_PATH:-"$HOME/models/Qwen3-1.7B_eagle3"}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

train_batch_size=${TRAIN_BATCH_SIZE:-512}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-128}
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.001}
entropy_coeff=${ENTROPY_COEFF:-0}

rollout_tp=${ROLLOUT_TP:-2}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.55}
rollout_n=${ROLLOUT_N:-5}

# Online drafter knobs
drafter_lr=${DRAFTER_LR:-1e-4}
drafter_update_interval=${DRAFTER_UPDATE_INTERVAL:-1}
drafter_num_steps=${DRAFTER_NUM_STEPS:-5}
drafter_replay_buf=${DRAFTER_REPLAY_BUF:-32768}
drafter_reward_weight=${DRAFTER_REWARD_WEIGHT:-0.0}

# EAGLE3 spec-decode: number of draft tokens proposed per step
eagle3_num_spec_tokens=${EAGLE3_NUM_SPEC_TOKENS:-3}

total_epochs=${TOTAL_EPOCHS:-15}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_qwen3_eagle3}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_1.7b_grpo_eagle3_online_drafter_$(date +%Y%m%d_%H%M)}
########################### end user-adjustable ################################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    # "data.train_files=['$HOME/data/gsm8k/train.parquet', '$HOME/data/math/train.parquet']"
    "data.train_files=['$HOME/data/math/train.parquet']"
    "data.val_files=['$HOME/data/math/test.parquet']"

    # "data.val_files=['$HOME/data/gsm8k/test.parquet', '$HOME/data/math/test.parquet']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # EAGLE3 spec-decode config for vLLM rollout
    actor_rollout_ref.rollout.eagle3.enable=${EAGLE3_ENABLE}
    "actor_rollout_ref.rollout.eagle3.model=${EAGLE3_MODEL_PATH}"
    actor_rollout_ref.rollout.eagle3.num_speculative_tokens=${eagle3_num_spec_tokens}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

# Online drafter: trains the EAGLE3 draft model in a background Ray worker.
# The trainer captures aux hidden states from the actor's compute_log_prob pass
# and runs CE distillation, then hot-swaps updated weights into vLLM.
ONLINE_DRAFTER=(
    online_drafter.enable=${ONLINE_DRAFTER_ENABLE}
    "online_drafter.draft_model_path=${EAGLE3_MODEL_PATH}"
    online_drafter.lr=${drafter_lr}
    online_drafter.update_interval_rl_steps=${drafter_update_interval}
    online_drafter.num_steps_per_update=${drafter_num_steps}
    online_drafter.replay_buffer_max_tokens=${drafter_replay_buf}
    online_drafter.reward_weight=${drafter_reward_weight}
    online_drafter.device=cuda:0
)

TRAINER=(
    trainer.balance_batch=True
    "trainer.logger=['console']"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${ONLINE_DRAFTER[@]}" \
    "${TRAINER[@]}" \
    "$@"
