from __future__ import annotations

from argparse import Namespace

import torch

from examples.single_turn_math_gold.gold_utils import (
    apply_response_prefix_loss_mask,
    call_llm_judge,
    compute_math_reward,
)
from slime.utils.types import Sample


async def reward_func_math(args: Namespace, sample: Sample, **kwargs) -> float:
    return compute_math_reward(sample)


async def reward_func_judge(args: Namespace, sample: Sample, **kwargs) -> float:
    return float(await call_llm_judge(args, sample))


async def reward_func(args: Namespace, sample: Sample, **kwargs) -> float:
    return await (
        reward_func_judge(args, sample, **kwargs)
        if getattr(args, "rm_type", None) == "llm_judge"
        else reward_func_math(args, sample, **kwargs)
    )


def _normalize_outcome_rewards(args: Namespace, raw_rewards: list[float]) -> list[float]:
    rewards = torch.tensor(raw_rewards, dtype=torch.float32)

    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])

        rewards = rewards - rewards.mean(dim=-1, keepdim=True)

        if args.advantage_estimator in ["grpo", "gspo"] and getattr(args, "grpo_std_normalization", False):
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)

    return rewards.flatten().tolist()


def post_process_rewards(args: Namespace, samples: list[Sample], **kwargs):
    raw_rewards = [float(sample.reward) for sample in samples]
    for sample in samples:
        apply_response_prefix_loss_mask(args, sample)
    # outcome归一化
    processed_rewards = _normalize_outcome_rewards(args, raw_rewards)
    zero_rewards = [0.0] * len(samples)
    return raw_rewards, zero_rewards
