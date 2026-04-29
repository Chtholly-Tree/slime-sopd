from __future__ import annotations

import asyncio
from typing import Any

from examples.single_turn_math_gold.gold_utils import (
    apply_response_prefix_loss_mask,
    build_teacher_prompt_ids,
    build_teacher_result,
    get_gold_http_session,
    prepare_teacher_log_probs,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def _run_student_generate(url: str, payload: dict, headers: dict | None = None) -> dict:
    session = get_gold_http_session("student")
    async with session.post(url, json=payload, headers=headers) as resp:
        resp.raise_for_status()
        return await resp.json()


async def generate(args: Any, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)

    sample.tokens = list(prompt_ids)
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []

    student_payload = {
        "input_ids": prompt_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        student_payload["image_data"] = [
            encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]
        ]

    prepare_teacher_prompt_task = None
    if not evaluation:
        prepare_teacher_prompt_task = asyncio.create_task(asyncio.to_thread(build_teacher_prompt_ids, args, sample))

    student_output = await _run_student_generate(url, student_payload)
    meta = student_output.get("meta_info", {})

    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        new_tokens, new_logprobs = [], []

    sample.tokens.extend(new_tokens)
    sample.response += student_output.get("text", "")
    sample.response_length += len(new_tokens)
    sample.rollout_log_probs.extend(new_logprobs)
    sample.loss_mask.extend([1] * len(new_tokens))
    sample.update_from_meta_info(args, meta)

    if evaluation:
        return sample

    teacher_prompt_ids = await prepare_teacher_prompt_task if prepare_teacher_prompt_task is not None else []
    teacher_result = await build_teacher_result(args, sample, teacher_prompt_ids)
    sample.teacher_log_probs = await asyncio.to_thread(prepare_teacher_log_probs, args, sample, teacher_result)
    apply_response_prefix_loss_mask(args, sample)
    return sample
