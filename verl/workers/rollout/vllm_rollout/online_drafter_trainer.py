"""VeRL worker for online EAGLE3 drafter training alongside RL rollout.

``OnlineDrafterWorker`` is designed to run as an additional Ray worker in a
VeRL training loop.  It holds a trainable replica of the EAGLE draft model,
collects hidden-state data produced by the actor (target) model forward
passes that VeRL already runs for policy-gradient / KL loss computation, and
periodically pushes updated drafter weights back to the vLLM rollout engine.

Integration sketch::

    # ---- in your VeRL trainer (e.g. PPOTrainer / GRPOTrainer) ----

    drafter_worker = OnlineDrafterWorker.remote(
        draft_model_path="AngelSlim/Qwen3-1.7B_eagle3",
        target_hidden_size=2048,
        num_target_layers=28,
        vllm_engine=rollout_wg.async_llm_engine,   # AsyncLLM handle
        config=OnlineDrafterConfig(),
    )

    for step in range(total_steps):
        # 1. Generate rollouts.
        rollout_data = rollout_wg.generate_sequences(batch)

        # 2. Actor forward pass (already done for policy gradient).
        #    Pass output_hidden_states=True and capture aux layers.
        actor_out = actor_wg.forward_with_hidden_states(rollout_data)

        # 3. Feed data to drafter worker (non-blocking).
        drafter_worker.add_rollout_data.remote(
            aux_hidden_states=actor_out.aux_hidden_states,
            target_logits=actor_out.logits,
            input_ids=rollout_data.input_ids,
            rewards=rollout_data.rewards,
        )

        # 4. Trigger drafter update on schedule (non-blocking).
        ray.get(drafter_worker.maybe_update.remote(step))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class OnlineDrafterConfig:
    """Configuration for the VeRL online drafter worker."""

    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0

    # Run a drafter update every N RL steps.
    update_interval_rl_steps: int = 1

    # Gradient steps per update.
    num_steps_per_update: int = 5

    # Replay buffer capacity in tokens.
    replay_buffer_max_tokens: int = 32_768

    # Reward-weighted CE coefficient (0 = plain CE, >0 = ReSpec-style RWKD).
    reward_weight: float = 0.0

    # Whether to pause vLLM generation before pushing new weights.
    pause_engine_during_update: bool = False

    # dtype for training.
    torch_dtype: torch.dtype = torch.bfloat16


class OnlineDrafterWorker:
    """Manages online training of the EAGLE draft model within a VeRL run.

    Intended to be wrapped in a Ray remote actor by the caller::

        RemoteWorker = ray.remote(OnlineDrafterWorker)
        worker = RemoteWorker.remote(...)

    Can also be used directly (non-Ray) in single-process experiments.
    """

    def __init__(
        self,
        draft_model_path: str,
        target_hidden_size: int,
        num_target_layers: int,
        vllm_engine: Any,  # AsyncLLM, list of Ray actor handles, or single actor handle
        config: OnlineDrafterConfig | None = None,
        device: str = "cuda:0",
    ) -> None:
        from vllm.v1.spec_decode.online_drafter_trainer import (
            DrafterTrainingConfig,
            EagleOnlineDrafterTrainer,
        )

        self.config = config or OnlineDrafterConfig()
        self.engine = vllm_engine

        train_cfg = DrafterTrainingConfig(
            lr=self.config.lr,
            betas=self.config.betas,
            weight_decay=self.config.weight_decay,
            num_steps_per_update=self.config.num_steps_per_update,
            update_interval_rl_steps=self.config.update_interval_rl_steps,
            replay_buffer_max_tokens=self.config.replay_buffer_max_tokens,
            reward_weight=self.config.reward_weight,
            dtype=self.config.torch_dtype,
        )

        self.trainer = EagleOnlineDrafterTrainer.from_pretrained(
            draft_model_name_or_path=draft_model_path,
            target_hidden_size=target_hidden_size,
            num_target_layers=num_target_layers,
            device=device,
            config=train_cfg,
            torch_dtype=self.config.torch_dtype,
        )
        logger.info(
            "OnlineDrafterWorker initialised: model=%s  device=%s",
            draft_model_path,
            device,
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_rollout_data(
        self,
        aux_hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        rewards: torch.Tensor | None = None,
    ) -> None:
        """Buffer training data for one or more rollout sequences.

        Parameters
        ----------
        aux_hidden_states:
            Shape ``[T, 3 * target_hidden_size]``.  Concatenated EAGLE3
            hidden states from three target-model layers for the generated
            tokens.  Obtain by running the actor model with
            ``output_hidden_states=True`` and extracting layers
            [1, n//2-1, n-4].
        input_ids:
            Shape ``[T]``.  Token ids of the sequence.
        rewards:
            Optional scalar reward per token.  If ``None``, all tokens
            receive weight 1.0 (plain CE).
        """
        if rewards is None:
            reward_scalar = 1.0
        else:
            # Use the mean reward for the trajectory as the scalar weight.
            reward_scalar = float(rewards.float().mean().item())

        self.trainer.add_rollout_data(
            aux_hidden_states=aux_hidden_states,
            input_ids=input_ids,
            reward=reward_scalar,
        )

    # ------------------------------------------------------------------
    # Update trigger
    # ------------------------------------------------------------------

    def maybe_update(self, rl_step: int) -> Optional[float]:
        """Run a drafter update if the schedule says it's time.

        Parameters
        ----------
        rl_step:
            Current RL training step index.

        Returns
        -------
        float or None
            Mean training loss if an update was performed, else ``None``.
        """
        self.trainer.increment_rl_step()
        if not self.trainer.should_update():
            return None

        loss = self.trainer.train_step()
        new_sd = self.trainer.get_trainable_state_dict()
        self._push_weights(new_sd)
        logger.info("drafter updated at RL step %d  loss=%.4f", rl_step, loss)
        return loss

    def force_update(self) -> float:
        """Unconditionally run a training pass and push weights."""
        loss = self.trainer.train_step()
        new_sd = self.trainer.get_trainable_state_dict()
        self._push_weights(new_sd)
        return loss

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _push_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Push updated drafter parameters to the vLLM inference engine.

        ``self.engine`` may be:
          - a list of Ray actor handles (vLLMHttpServer per replica) — fan out;
          - a single Ray actor handle — call directly via ``.remote()``;
          - an in-process AsyncLLM-like object — call ``await update_draft_weights``.
        """
        # Case 1: list of Ray actor handles (the production path).
        if isinstance(self.engine, (list, tuple)):
            import ray

            futures = [h.update_draft_weights.remote(state_dict) for h in self.engine]
            ray.get(futures)
            return

        # Case 2: single Ray actor handle.
        if hasattr(self.engine, "update_draft_weights") and hasattr(
            self.engine.update_draft_weights, "remote"
        ):
            import ray

            ray.get(self.engine.update_draft_weights.remote(state_dict))
            return

        # Case 3: in-process AsyncLLM (used by unit tests).
        import asyncio

        async def _do_push() -> None:
            if self.config.pause_engine_during_update:
                await self.engine.pause_generation()
            try:
                await self.engine.update_draft_weights(state_dict)
            finally:
                if self.config.pause_engine_during_update:
                    await self.engine.resume_generation()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(_do_push(), loop)
                fut.result(timeout=60)
            else:
                loop.run_until_complete(_do_push())
        except RuntimeError:
            asyncio.run(_do_push())


# ---------------------------------------------------------------------------
# Helper: extract EAGLE3 aux hidden states from a HuggingFace forward pass
# ---------------------------------------------------------------------------

def extract_eagle3_aux_hidden_states(
    hidden_states_all_layers: list[torch.Tensor],
    num_target_layers: int,
) -> torch.Tensor:
    """Concatenate the three EAGLE3 source layers from a full hidden-state list.

    Parameters
    ----------
    hidden_states_all_layers:
        List of per-layer hidden states returned by a HuggingFace model when
        called with ``output_hidden_states=True``.  Length is
        ``num_hidden_layers + 1`` (embedding layer first).
    num_target_layers:
        Total number of transformer layers in the target model.

    Returns
    -------
    torch.Tensor
        Shape ``[T, 3 * hidden_size]``.
    """
    # EAGLE3 uses layers: 1, num_layers//2 - 1, num_layers - 4
    # (indices into hidden_states_all_layers which includes the embedding at 0)
    idx_early = 1
    idx_mid = num_target_layers // 2       # offset by 1 for embedding
    idx_late = num_target_layers - 4 + 1   # offset by 1 for embedding

    h_early = hidden_states_all_layers[idx_early]  # [B, T, d]
    h_mid = hidden_states_all_layers[idx_mid]
    h_late = hidden_states_all_layers[idx_late]

    # Flatten batch dimension if present.
    def _flatten(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:
            B, T, d = t.shape
            return t.view(B * T, d)
        return t

    return torch.cat([_flatten(h_early), _flatten(h_mid), _flatten(h_late)], dim=-1)
