from __future__ import annotations

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


def post_process_rewards(args, samples: list[Sample], **kwargs):
    raw = [float(sample.reward) for sample in samples]
    for sample in samples:
        if sample.teacher_log_probs is None:
            sample.teacher_log_probs = torch.tensor(sample.rollout_log_probs or [], dtype=torch.float32)
        _apply_response_prefix_loss_mask(args, sample)
    return raw, [0.0] * len(samples)

