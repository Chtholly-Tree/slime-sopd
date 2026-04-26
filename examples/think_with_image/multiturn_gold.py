from __future__ import annotations

import re
import torch

from examples.think_with_image.rewards.llm_judge import async_rm as judge_async_rm
from slime.rollout.multimodal_gold import _apply_response_prefix_loss_mask
from slime.rollout.rm_hub.multimodal import compute_math_reward
from slime.utils.types import Sample


async def reward_func_math(args, sample: Sample, **kwargs) -> float:
    return compute_math_reward(sample)


async def reward_func_judge(args, sample: Sample, **kwargs) -> float:
    return float(await judge_async_rm(args, sample))


async def reward_func(args, sample: Sample, **kwargs) -> float:
    return await (
        reward_func_judge(args, sample, **kwargs)
        if getattr(args, "rm_type", None) == "llm_judge"
        else reward_func_math(args, sample, **kwargs)
    )


def _normalize_group_rewards(args, rewards: list[float]) -> list[float]:
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    if (
        args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        if rewards_tensor.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards_tensor = rewards_tensor.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])
        rewards_tensor = rewards_tensor - rewards_tensor.mean(dim=-1, keepdim=True)
        if args.advantage_estimator in ["grpo", "gspo"] and getattr(args, "grpo_std_normalization", False):
            rewards_tensor = rewards_tensor / (rewards_tensor.std(dim=-1, keepdim=True) + 1e-6)
    return rewards_tensor.flatten().tolist()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    outcome_reward_coef = float(getattr(args, "outcome_reward_coef", 1.0))
    format_reward_coef = float(getattr(args, "format_reward_coef", 1.0))
    outcome_rewards: list[float] = []
    format_rewards: list[float] = []
    weighted_total_rewards: list[float] = []
    raw_outcome_rewards = [float(sample.reward or 0.0) for sample in samples]
    for sample in samples:
        outcome_reward = 1.0 if float(sample.reward or 0.0) > 0.0 else 0.0
        format_reward = 1.0 if re.search(r"<answer>.*?</answer>", sample.response or "", flags=re.IGNORECASE | re.DOTALL) else 0.0
        total_reward = outcome_reward_coef * outcome_reward + format_reward_coef * format_reward

        outcome_rewards.append(outcome_reward)
        format_rewards.append(format_reward)
        weighted_total_rewards.append(total_reward)
        sample.metadata["raw_reward"] = float(sample.reward or 0.0)
        sample.metadata["weighted_reward"] = total_reward

        if sample.teacher_log_probs is None:
            sample.teacher_log_probs = torch.tensor(sample.rollout_log_probs or [], dtype=torch.float32)
        _apply_response_prefix_loss_mask(args, sample)

    normalized_rewards = _normalize_group_rewards(args, weighted_total_rewards)
    return raw_outcome_rewards, normalized_rewards
