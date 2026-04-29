from __future__ import annotations

import logging
import math
import os
from typing import Any

import aiohttp
import torch
from transformers import AutoTokenizer

from slime.rollout.rm_hub.math_utils import grade_answer_verl
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
)
from slime.utils.teacher_pool import get_teacher_pool
from slime.utils.types import Sample

_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}
_PROCESSOR_CACHE: dict[str, object | None] = {}
_HTTP_SESSION_CACHE: dict[str, aiohttp.ClientSession] = {}
_HTTP_SESSION_TIMEOUTS = {
    "student": aiohttp.ClientTimeout(total=12000),
    "teacher": aiohttp.ClientTimeout(total=6000),
}
logger = logging.getLogger(__name__)


def get_gold_http_session(kind: str) -> aiohttp.ClientSession:
    session = _HTTP_SESSION_CACHE.get(kind)
    if session is None or session.closed:
        timeout = _HTTP_SESSION_TIMEOUTS[kind]
        session = aiohttp.ClientSession(timeout=timeout)
        _HTTP_SESSION_CACHE[kind] = session
    return session


def get_student_tokenizer(hf_checkpoint: str):
    tokenizer = _TOKENIZER_CACHE.get(hf_checkpoint)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(hf_checkpoint, trust_remote_code=True)
        _TOKENIZER_CACHE[hf_checkpoint] = tokenizer
    return tokenizer


def get_teacher_hf_checkpoint(args) -> str:
    teacher_hf_checkpoint = getattr(args, "gold_teacher_hf_checkpoint", None) or os.environ.get("GOLD_TEACHER_HF_CHECKPOINT")
    if not teacher_hf_checkpoint:
        raise ValueError("single_turn_math_gold requires --gold-teacher-hf-checkpoint")
    return teacher_hf_checkpoint


def get_teacher_tokenizer(args):
    teacher_hf_checkpoint = get_teacher_hf_checkpoint(args)
    tokenizer = _TOKENIZER_CACHE.get(teacher_hf_checkpoint)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(teacher_hf_checkpoint, trust_remote_code=True)
        _TOKENIZER_CACHE[teacher_hf_checkpoint] = tokenizer
    return tokenizer


def get_teacher_processor(args):
    teacher_hf_checkpoint = get_teacher_hf_checkpoint(args)
    if teacher_hf_checkpoint not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[teacher_hf_checkpoint] = load_processor(teacher_hf_checkpoint, trust_remote_code=True)
    return _PROCESSOR_CACHE[teacher_hf_checkpoint]


def build_teacher_prompt_text(args, sample: Sample) -> str:
    raw_prompt = sample.metadata.get("raw_prompt")
    if raw_prompt is None:
        assert isinstance(sample.prompt, str), "single_turn_math_gold expects sample.prompt to be a string"
        return sample.prompt

    tools = sample.metadata.get("tools")
    teacher_tokenizer = get_teacher_tokenizer(args)
    return teacher_tokenizer.apply_chat_template(
        raw_prompt,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def trim_prompt_eos(tokenizer, prompt_ids: list[int]) -> list[int]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and prompt_ids and prompt_ids[-1] == eos_token_id:
        return prompt_ids[:-1]
    return prompt_ids


def build_teacher_prompt_ids(args: Any, sample: Sample) -> list[int]:
    prompt_text = build_teacher_prompt_text(args, sample)
    teacher_processor = get_teacher_processor(args)
    if teacher_processor is not None and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = teacher_processor(text=prompt_text, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0]
        if isinstance(prompt_ids, torch.Tensor):
            prompt_ids = prompt_ids.tolist()
        else:
            prompt_ids = list(prompt_ids)
        teacher_tokenizer = getattr(teacher_processor, "tokenizer", None) or get_teacher_tokenizer(args)
        return trim_prompt_eos(teacher_tokenizer, prompt_ids)

    teacher_tokenizer = get_teacher_tokenizer(args)
    prompt_ids = teacher_tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
    return trim_prompt_eos(teacher_tokenizer, prompt_ids)


async def teacher_request(
    args: Any,
    sample: Sample,
    payload: dict[str, Any],
    *,
    request_name: str = "single_turn_math_gold_teacher",
) -> tuple[dict, str | None, str | None]:
    if getattr(args, "teacher_model_name", None) and getattr(args, "teacher_pool_config", None):
        result, endpoint = await get_teacher_pool(args).request_json(payload, request_name=request_name)
        sample.metadata["gold_teacher_endpoint"] = endpoint.name
        sample.metadata["gold_teacher_url"] = endpoint.url
        sample.metadata["gold_teacher_model_name"] = args.teacher_model_name
        return result, endpoint.name, endpoint.url

    session = get_gold_http_session("teacher")
    async with session.post(args.rm_url, json=payload) as resp:
        resp.raise_for_status()
        return await resp.json(), None, args.rm_url


async def build_teacher_result(args: Any, sample: Sample, teacher_prompt_ids: list[int]) -> dict:
    teacher_tokenizer = get_teacher_tokenizer(args)
    teacher_response_ids = teacher_tokenizer(sample.response, add_special_tokens=False)["input_ids"]
    teacher_input_ids = list(teacher_prompt_ids) + list(teacher_response_ids)
    payload = {
        "input_ids": teacher_input_ids,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": True,
    }

    teacher_images = (sample.multimodal_inputs or {}).get("images", [])
    if teacher_images:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in teacher_images]

    result, _, _ = await teacher_request(args, sample, payload)
    result.setdefault("meta_info", {})["gold_teacher_prompt_len"] = len(teacher_prompt_ids)
    result["meta_info"]["gold_teacher_input_ids"] = teacher_input_ids
    return result


def _decode_teacher_triplet_piece(item: list) -> str:
    return str(item[2]) if len(item) > 2 and item[2] is not None else str(item[1])


def _find_first_mismatch_index(local_pieces: list[str], remote_pieces: list[str]) -> int:
    limit = min(len(local_pieces), len(remote_pieces))
    for idx in range(limit):
        if local_pieces[idx] != remote_pieces[idx]:
            return idx
    if len(local_pieces) != len(remote_pieces):
        return limit
    return -1


def _format_teacher_token_window(local_token_ids: list[int], local_pieces: list[str], triplets: list[list], remote_pieces: list[str], start: int, end: int) -> str:
    lines = []
    for idx in range(start, end):
        local_token_id = local_token_ids[idx] if idx < len(local_token_ids) else None
        local_piece = local_pieces[idx] if idx < len(local_pieces) else None
        remote_logprob = float(triplets[idx][0]) if idx < len(triplets) and len(triplets[idx]) > 0 else None
        remote_token_id = triplets[idx][1] if idx < len(triplets) and len(triplets[idx]) > 1 else None
        remote_piece = remote_pieces[idx] if idx < len(remote_pieces) else None
        lines.append(
            f"idx={idx} local_token_id={local_token_id} local_piece={local_piece!r} remote_token_id={remote_token_id} remote_piece={remote_piece!r} remote_logprob={remote_logprob}"
        )
    return "\n".join(lines)


def _log_teacher_tokenization_mismatch(
    args: Any,
    sample: Sample,
    teacher_result: dict,
    prompt_len: int,
    response_len: int,
    triplets: list[list],
) -> None:
    teacher_tokenizer = get_teacher_tokenizer(args)
    teacher_input_ids = list(teacher_result["meta_info"].get("gold_teacher_input_ids", []))
    local_pieces = [_decode_teacher_triplet_piece([None, token_id, teacher_tokenizer.decode(token_id)]) for token_id in teacher_input_ids]
    remote_pieces = [_decode_teacher_triplet_piece(item) for item in triplets]
    mismatch_idx = _find_first_mismatch_index(local_pieces, remote_pieces)

    window_start = max(0, mismatch_idx - 5) if mismatch_idx >= 0 else max(0, min(len(local_pieces), len(remote_pieces)) - 5)
    window_end = min(max(len(local_pieces), len(remote_pieces)), (mismatch_idx + 6) if mismatch_idx >= 0 else window_start + 10)
    window_text = _format_teacher_token_window(
        teacher_input_ids,
        local_pieces,
        triplets,
        remote_pieces,
        window_start,
        window_end,
    )

    logger.error(
        "single_turn_math_gold teacher tokenization mismatch sample=%s expected_total=%s actual_total=%s prompt_len=%s response_len=%s mismatch_index=%s\n%s",
        sample.index,
        prompt_len + response_len,
        len(triplets),
        prompt_len,
        response_len,
        mismatch_idx,
        window_text,
    )


def extract_teacher_response_triplets(args: Any, sample: Sample, teacher_result: dict, prompt_len: int, response_len: int) -> list[list]:
    triplets = teacher_result["meta_info"]["input_token_logprobs"]
    if response_len <= 0:
        return []
    if len(triplets) != prompt_len + response_len:
        _log_teacher_tokenization_mismatch(args, sample, teacher_result, prompt_len, response_len, triplets)
        raise ValueError(
            "Teacher token logprob length does not match local teacher tokenization. "
            f"expected_total={prompt_len + response_len}, actual_total={len(triplets)}"
        )
    return triplets[prompt_len : prompt_len + response_len]


def to_canonical_pieces_from_ids(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode(token_id) for token_id in token_ids]


def to_canonical_pieces_from_triplets(teacher_triplets: list[list]) -> list[str]:
    pieces = []
    prev = ""
    cur = ""
    for item in teacher_triplets:
        piece = str(item[2]) if len(item) > 2 and item[2] is not None else str(item[1])
        cur += piece
        pieces.append(cur[len(prev) :])
        prev = cur
    return pieces


def build_alignment_groups(student_pieces: list[str], teacher_pieces: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    i = j = 0
    student_buf = teacher_buf = ""
    student_group: list[int] = []
    teacher_group: list[int] = []
    student_groups: list[list[int]] = []
    teacher_groups: list[list[int]] = []

    def flush_group():
        nonlocal student_buf, teacher_buf, student_group, teacher_group
        if student_group and teacher_group:
            student_groups.append(student_group.copy())
            teacher_groups.append(teacher_group.copy())
        student_buf = teacher_buf = ""
        student_group = []
        teacher_group = []

    while i < len(student_pieces) or j < len(teacher_pieces):
        if student_buf == teacher_buf and student_buf != "":
            flush_group()
            continue

        if student_buf == "" and i < len(student_pieces):
            student_buf += student_pieces[i]
            student_group.append(i)
            i += 1
            continue

        if teacher_buf == "" and j < len(teacher_pieces):
            teacher_buf += teacher_pieces[j]
            teacher_group.append(j)
            j += 1
            continue

        if len(student_buf) <= len(teacher_buf):
            if i < len(student_pieces):
                student_buf += student_pieces[i]
                student_group.append(i)
                i += 1
            elif j < len(teacher_pieces):
                teacher_buf += teacher_pieces[j]
                teacher_group.append(j)
                j += 1
        else:
            if j < len(teacher_pieces):
                teacher_buf += teacher_pieces[j]
                teacher_group.append(j)
                j += 1
            elif i < len(student_pieces):
                student_buf += student_pieces[i]
                student_group.append(i)
                i += 1

    if student_buf == teacher_buf and student_group and teacher_group:
        flush_group()
    elif student_group or teacher_group:
        student_groups.append(student_group.copy() if student_group else [])
        teacher_groups.append(teacher_group.copy() if teacher_group else [])

    return student_groups, teacher_groups


def prepare_teacher_log_probs(args: Any, sample: Sample, teacher_result: dict) -> torch.Tensor:
    student_log_probs = torch.tensor(sample.rollout_log_probs or [], dtype=torch.float32)
    if sample.response_length <= 0:
        sample.metadata["teacher_token_triplets"] = []
        sample.metadata["gold_groups"] = []
        sample.metadata["gold_debug"] = {"student_tokens": [], "teacher_tokens": []}
        return student_log_probs

    teacher_prompt_len = int(teacher_result["meta_info"]["gold_teacher_prompt_len"])
    teacher_response_len = len(teacher_result["meta_info"]["gold_teacher_input_ids"]) - teacher_prompt_len
    teacher_triplets = extract_teacher_response_triplets(args, sample, teacher_result, teacher_prompt_len, teacher_response_len)
    teacher_log_probs = torch.tensor([float(item[0]) for item in teacher_triplets], dtype=torch.float32)

    student_tokenizer = get_student_tokenizer(args.hf_checkpoint)
    student_token_ids = list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []
    student_pieces = to_canonical_pieces_from_ids(student_tokenizer, student_token_ids)
    teacher_pieces = to_canonical_pieces_from_triplets(teacher_triplets)

    student_len = min(len(student_pieces), len(student_log_probs))
    teacher_len = min(len(teacher_pieces), len(teacher_log_probs))
    student_pieces = student_pieces[:student_len]
    student_log_probs = student_log_probs[:student_len]
    teacher_pieces = teacher_pieces[:teacher_len]
    teacher_log_probs = teacher_log_probs[:teacher_len]
    teacher_triplets = teacher_triplets[:teacher_len]

    student_groups, teacher_groups = build_alignment_groups(student_pieces, teacher_pieces)

    synthetic_teacher_log_probs = student_log_probs.clone()
    gold_groups = []
    for student_group, teacher_group in zip(student_groups, teacher_groups, strict=False):
        if not student_group or not teacher_group:
            continue

        student_group_logprob = sum(float(student_log_probs[idx]) for idx in student_group)
        teacher_group_logprob = sum(float(teacher_log_probs[idx]) for idx in teacher_group)
        group_logprob_gap = student_group_logprob - teacher_group_logprob

        gold_groups.append(
            {
                "student_group": student_group,
                "teacher_group": teacher_group,
                "group_logprob_gap": group_logprob_gap,
            }
        )
        for idx in student_group:
            synthetic_teacher_log_probs[idx] = student_log_probs[idx] - group_logprob_gap

    sample.metadata["teacher_token_triplets"] = teacher_triplets
    sample.metadata["gold_groups"] = gold_groups
    sample.metadata["gold_debug"] = {
        "student_tokens": student_token_ids,
        "student_pieces": student_pieces,
        "teacher_pieces": teacher_pieces,
    }
    return synthetic_teacher_log_probs


def apply_response_prefix_loss_mask(args, sample: Sample) -> None:
    ratio = getattr(args, "gold_train_response_prefix_ratio", None)
    if ratio is None:
        return

    ratio = float(ratio)
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"gold_train_response_prefix_ratio must be in [0, 1], got {ratio}")

    response_length = int(sample.response_length)
    if response_length <= 0:
        sample.loss_mask = []
        return

    keep_tokens = math.ceil(response_length * ratio)
    keep_tokens = min(max(keep_tokens, 0), response_length)
    sample.loss_mask = [1] * keep_tokens + [0] * (response_length - keep_tokens)


def compute_math_reward(sample: Sample) -> float:
    return 1.0 if grade_answer_verl(sample.response, str(sample.label)) else 0.0


def extract_boxed_answer(response: str) -> str | None:
    import re

    match = re.search(r"\\boxed\{([^}]+)\}", response)
    if match:
        return match.group(1)
    return None


async def call_llm_judge(args: Any, sample: Sample) -> float:
    try:
        math_reward = compute_math_reward(sample)
        if math_reward > 0.5:
            return 1.0
    except Exception:
        pass

    question = sample.metadata.get("prompt", sample.prompt)
    ground_truth = str(sample.label)
    prediction = sample.response

    boxed_answer = extract_boxed_answer(prediction)
    if boxed_answer:
        prediction_for_judge = f"\\boxed{{{boxed_answer}}}"
    else:
        prediction_for_judge = prediction

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
        "- Output only one word: \"correct\" or \"incorrect\".\n\n"
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
