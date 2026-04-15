from __future__ import annotations

from argparse import Namespace

import aiohttp

from slime.utils.types import Sample


def _extract_boxed_answer(response: str) -> str | None:
    import re

    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1)
    return None


def compute_math_reward(sample: Sample) -> float:
    from slime.rollout.rm_hub.math_utils import grade_answer_verl

    return 1.0 if grade_answer_verl(sample.response, str(sample.label)) else 0.0


async def call_llm_judge(args: Namespace, sample: Sample) -> float:
    try:
        math_reward = compute_math_reward(sample)
        if math_reward > 0.5:
            return 1.0
    except Exception:
        pass

    question = sample.metadata.get("prompt") or (sample.prompt if isinstance(sample.prompt, str) else str(sample.prompt))
    ground_truth = str(sample.label)
    prediction = sample.response

    boxed_answer = _extract_boxed_answer(prediction)
    prediction_for_judge = f"\\boxed{{{boxed_answer}}}" if boxed_answer else prediction

    judge_prompt = (
        "Please evaluate whether the model's answer is correct by comparing it "
        "with the standard answer.\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Predicted Answer: {prediction_for_judge}\n\n"
        "**Instructions:**\n"
        "- Compare the model's answer with the standard answer\n"
        "- Focus on factual accuracy and key points\n"
        "- Allow for different wording if the core meaning is the same\n"
        "- Consider the answer correct if it captures the main points\n"
        '- Output only one word: "correct" or "incorrect".\n\n'
        "**Output format:**\ncorrect/incorrect"
    )

    payload = {
        "text": judge_prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 30,
            "skip_special_tokens": False,
        },
    }

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.judge_url, json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()

    content = result.get("text", "").strip().lower()
    if "incorrect" in content:
        return 0.0
    if "correct" in content:
        return 1.0
    return 0.0


async def reward_func_math(args: Namespace, sample: Sample, **kwargs) -> float:
    return compute_math_reward(sample)


async def reward_func_judge(args: Namespace, sample: Sample, **kwargs) -> float:
    return float(await call_llm_judge(args, sample))


async def reward_func(args: Namespace, sample: Sample, **kwargs) -> float:
    mode = getattr(args, "gold_raw_reward_mode", None) or "judge"
    if mode == "math":
        return await reward_func_math(args, sample, **kwargs)
    if mode == "judge":
        return await reward_func_judge(args, sample, **kwargs)
    raise ValueError(f"Unknown gold_raw_reward_mode: {mode}")
