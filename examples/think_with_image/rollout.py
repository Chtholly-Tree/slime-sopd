"""
Custom multi-turn rollout for VLM tool-calling (think-with-image).

Usage: --custom-generate-function-path examples.think_with_image.rollout.generate

Flow per sample:
    1. Model generates (assistant turn, loss_mask=1)
    2. Env processes: extract <tool_call> or <answer>
       - If <answer>: done, record final answer
       - If <tool_call>: execute tool, encode observation (loss_mask=0)
    3. Repeat until max_turns or model provides <answer>
    4. Finalize: decode response, merge multimodal inputs, set status
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

__all__ = ["generate"]

logger = logging.getLogger(__name__)

# Dummy messages for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    target = env_path or "examples.think_with_image.env"
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"env_{module_path.stem}", module_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    build_fn = getattr(env_module, "build_env", None)
    if not callable(build_fn):
        raise ValueError("Environment module must expose build_env(sample, args)")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        return build_fn(sample, args)


def _encode_observation(tokenizer, processor, message: dict, metadata: dict | None, args: Any):
    """Encode observation message for generation. Returns (prompt_ids, image_data)."""
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}

    trim_length = 0
    if getattr(args, "apply_chat_template", False):
        dummy = tokenizer.apply_chat_template(DUMMY_MESSAGES, tools=tools, tokenize=False, add_generation_prompt=False, **apply_kwargs)
        formatted = tokenizer.apply_chat_template(DUMMY_MESSAGES + [message], tools=tools, tokenize=False, add_generation_prompt=True, **apply_kwargs)
        trim_length = len(tokenizer.encode(dummy, add_special_tokens=False))
    else:
        formatted = [message]

    multimodal_inputs = None
    if processor:
        from qwen_vl_utils import process_vision_info
        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        output = processor(text=formatted, **multimodal_inputs)
        prompt_ids = output["input_ids"][0]
    else:
        prompt_ids = tokenizer.encode(formatted, add_special_tokens=False)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]

    return prompt_ids, image_data


def _merge_multimodal_inputs(chunks: list[dict | None]) -> dict | None:
    """Merge per-turn multimodal inputs by concatenating tensors."""
    if not chunks:
        return None
    values: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for k, v in chunk.items():
            if v is not None and isinstance(v, torch.Tensor):
                values.setdefault(k, []).append(v)
    return {k: torch.cat(v, dim=0) for k, v in values.items()} if values else None


async def _run_inference(url: str, tokens: list[int], sampling_params: dict, image_data):
    """Call SGLang for one generation turn."""
    payload = {"input_ids": tokens, "sampling_params": sampling_params, "return_logprob": True}
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]
    meta = output.get("meta_info", {})

    if "output_token_logprobs" in meta:
        new_tokens = [item[1] for item in meta["output_token_logprobs"]]
        new_logprobs = [item[0] for item in meta["output_token_logprobs"]]
    else:
        new_tokens, new_logprobs = [], []

    finish_type = meta.get("finish_reason", {}).get("type", "")
    return response_text, new_tokens, new_logprobs, finish_type


def _append_tokens(sample: Sample, response_tokens: list[int], tokens: list[int], logprobs: list[float], loss_mask: int):
    """Append tokens to sample with specified loss_mask."""
    sample.tokens.extend(tokens)
    response_tokens.extend(tokens)
    sample.loss_mask.extend([loss_mask] * len(tokens))
    sample.rollout_log_probs.extend(logprobs)


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """
    Main entry point for multi-turn VLM tool-calling rollout.
    """
    assert not args.partial_rollout, "Partial rollout not supported"

    # Inject tools schema
    from examples.think_with_image.tools.registry import get_tools
    sample.metadata = sample.metadata or {}
    sample.metadata.setdefault("tools", get_tools())

    # Init env and state
    env_module = _load_env_module(getattr(args, "rollout_interaction_env_path", None))
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path")

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    env = _build_env(env_module, sample, args)

    # Prepare initial inputs
    if sample.multimodal_inputs and state.processor:
        proc_output = state.processor(text=sample.prompt, **sample.multimodal_inputs)
        prompt_ids = proc_output["input_ids"][0]
        mm_train_inputs = {k: v for k, v in proc_output.items() if k not in ("input_ids", "attention_mask")} or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
        mm_train_inputs = None

    sample.tokens = list(prompt_ids)
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []

    current_images = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        current_images = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]

    response_tokens: list[int] = []
    mm_inputs_buffer: list = [mm_train_inputs] if mm_train_inputs else []
    budget = None
    if args.rollout_max_context_len:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens"):
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)

    sampling_params = deepcopy(sampling_params)
    sample.status = None

    try:
        env.reset()

        for turn in range(max_turns):
            # Check budget
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            # Generate
            cur_params = {**sampling_params, "max_new_tokens": budget} if budget else sampling_params
            response_text, new_tokens, new_logprobs, finish_type = await _run_inference(
                url, sample.tokens, cur_params, current_images
            )
            breakpoint()
            # Record assistant tokens (loss=1)
            _append_tokens(sample, response_tokens, new_tokens, new_logprobs, loss_mask=1)
            if budget is not None:
                budget -= len(new_tokens)

            # Check finish type
            if finish_type in ("length", "abort"):
                sample.status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.ABORTED
                break

            # Env step
            observation, done, _ = env.step(response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            # Encode observation (loss=0)
            next_msg = env.format_observation(observation)
            obs_ids, obs_images = _encode_observation(state.tokenizer, state.processor, next_msg, sample.metadata, args)
            if state.tokenizer.bos_token_id and obs_ids and obs_ids[0] == state.tokenizer.bos_token_id:
                obs_ids = obs_ids[1:]

            _append_tokens(sample, response_tokens, obs_ids, [0.0] * len(obs_ids), loss_mask=0)
            if budget is not None:
                budget -= len(obs_ids)

            # Accumulate images
            if obs_images:
                current_images.extend(obs_images)
            if observation.get("multi_modal_data"):
                sample.multimodal_inputs = sample.multimodal_inputs or {}
                for k, v in observation["multi_modal_data"].items():
                    if v:
                        sample.multimodal_inputs.setdefault(k, []).extend(v)
            mm_inputs_buffer.append(None)

            # Check max turns
            if turn + 1 >= max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        # Finalize
        if sample.status is None:
            sample.status = Sample.Status.COMPLETED
        sample.multimodal_train_inputs = _merge_multimodal_inputs(mm_inputs_buffer)
        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        return sample

    finally:
        try:
            env.close()
        except Exception:
            pass
