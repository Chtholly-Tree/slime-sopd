import os

import aiohttp

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def call_multimodal_teacher(
    args,
    sample: Sample,
    *,
    return_text_in_logprobs: bool = False,
    use_teacher_context_env: bool = False,
) -> dict:
    """Call SGLang teacher server for multimodal token logprobs."""
    if use_teacher_context_env and os.environ.get("USE_TEACHER_CONTEXT", "0") == "1":
        if "teacher_prompt_ids" in sample.metadata:
            response_ids = sample.tokens[-sample.response_length:]
            input_ids = list(sample.metadata["teacher_prompt_ids"]) + list(response_ids)
            teacher_images = sample.metadata.get(
                "teacher_images",
                (sample.multimodal_inputs or {}).get("images", []),
            )
        else:
            input_ids = sample.tokens
            teacher_images = (sample.multimodal_inputs or {}).get("images", [])
    else:
        input_ids = sample.tokens
        teacher_images = (sample.multimodal_inputs or {}).get("images", [])

    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    if return_text_in_logprobs:
        payload["return_text_in_logprobs"] = True
    if teacher_images:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in teacher_images]

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def call_llm_judge(args, sample: Sample) -> float:
    """Call vLLM judge server to score correctness."""
    judge_url = os.environ.get("JUDGE_URL")
    assert judge_url, "JUDGE_URL environment variable is not set"
    judge_model = os.environ.get("JUDGE_MODEL", "default")

    question = sample.metadata.get("question", str(sample.label))
    ground_truth = str(sample.label)
    prediction = sample.response

    judge_prompt = (
        "Please evaluate whether the model's answer is correct by comparing it "
        "with the standard answer.\n"
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Predicted Answer: {prediction}\n\n"
        "**Instructions:**\n"
        "- Compare the model's answer with the standard answer\n"
        "- Focus on factual accuracy and key points\n"
        "- Allow for different wording if the core meaning is the same\n"
        "- Consider the answer correct if it captures the main points\n"
        "- Output only one word: \"correct\" or \"incorrect\".\n\n"
        "**Output format:**\ncorrect/incorrect"
    )

    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": judge_prompt}],
        "temperature": 0,
        "max_tokens": 16,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{judge_url.rstrip('/')}/v1/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            result = await resp.json()

    content = result["choices"][0]["message"]["content"].strip().lower()
    if "incorrect" in content:
        return 0.0
    if "correct" in content:
        return 1.0
    return 0.0


def compute_math_reward(sample: Sample) -> float:
    """Rule-based math reward for a sample."""
    from slime.rollout.rm_hub.math_utils import grade_answer_verl

    return 1.0 if grade_answer_verl(sample.response, str(sample.label)) else 0.0
