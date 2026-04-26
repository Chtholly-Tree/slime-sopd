from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import aiohttp
import torch
from transformers import AutoTokenizer

from slime.utils.processing_utils import encode_image_for_rollout_engine, load_processor, process_vision_info
from slime.utils.teacher_pool import get_teacher_pool
from slime.utils.types import Sample

_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}
_PROCESSOR_CACHE: dict[str, object | None] = {}
_DUMMY = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


@dataclass(slots=True)
class TeacherRolloutState:
    input_ids: list[int]
    image_data: list[Any] = field(default_factory=list)
    segments: list[dict[str, Any]] = field(default_factory=list)


def _teacher_ckpt(args) -> str:
    path = getattr(args, "gold_teacher_hf_checkpoint", None)
    if not path:
        raise ValueError("think_with_image multiturn distillation requires --gold-teacher-hf-checkpoint")
    return path


def _teacher_tokenizer(args):
    path = _teacher_ckpt(args)
    tok = _TOKENIZER_CACHE.get(path)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _TOKENIZER_CACHE[path] = tok
    return tok


def _teacher_processor(args):
    path = _teacher_ckpt(args)
    if path not in _PROCESSOR_CACHE:
        _PROCESSOR_CACHE[path] = load_processor(path, trust_remote_code=True)
    return _PROCESSOR_CACHE[path]


def _student_tokenizer(args):
    path = args.hf_checkpoint
    tok = _TOKENIZER_CACHE.get(path)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        _TOKENIZER_CACHE[path] = tok
    return tok


def _tolist(x):
    if isinstance(x, torch.Tensor):
        return x.tolist()
    return list(x)


def _encode_increment(tokenizer, processor, message: dict, metadata: dict | None, args: Any) -> tuple[list[int], list[Any]]:
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    trim = 0
    if getattr(args, "apply_chat_template", False):
        dummy = tokenizer.apply_chat_template(_DUMMY, tools=tools, tokenize=False, add_generation_prompt=False, **apply_kwargs)
        formatted = tokenizer.apply_chat_template(_DUMMY + [message], tools=tools, tokenize=False, add_generation_prompt=True, **apply_kwargs)
        trim = len(tokenizer.encode(dummy, add_special_tokens=False))
    else:
        formatted = [message]

    image_data: list[Any] = []
    if processor:
        vision_inputs = process_vision_info([message], processor)
        images = vision_inputs.get("images") or []
        videos = vision_inputs.get("videos") or []
        if images or videos:
            prompt_ids = _tolist(processor(text=formatted, images=images or None, videos=videos or None)["input_ids"][0])
            image_data = [encode_image_for_rollout_engine(img) for img in images]
        else:
            prompt_ids = _tolist(processor(text=formatted)["input_ids"][0])
    else:
        prompt_ids = _tolist(tokenizer.encode(formatted, add_special_tokens=False))
    if trim:
        prompt_ids = prompt_ids[trim:]
    return prompt_ids, image_data


def _build_teacher_prompt_text(args: Any, sample: Sample) -> str:
    raw_prompt = sample.metadata.get("raw_prompt") if sample.metadata else None
    if raw_prompt is None:
        assert isinstance(sample.prompt, str), "think_with_image expects sample.prompt to be a string when raw_prompt is missing"
        return sample.prompt

    if isinstance(raw_prompt, str):
        return raw_prompt

    tools = sample.metadata.get("tools") if sample.metadata else None
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", None) or {}
    tok = _teacher_tokenizer(args)
    return tok.apply_chat_template(
        raw_prompt,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **apply_kwargs,
    )


def build_teacher_rollout_state(args: Any, sample: Sample) -> TeacherRolloutState:
    tok = _teacher_tokenizer(args)
    proc = _teacher_processor(args)
    teacher_prompt = _build_teacher_prompt_text(args, sample)
    if proc and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        prompt_ids = _tolist(proc(text=teacher_prompt, **sample.multimodal_inputs)["input_ids"][0])
    else:
        prompt_ids = _tolist(tok.encode(teacher_prompt, add_special_tokens=False))
    image_data = [encode_image_for_rollout_engine(img) for img in (sample.multimodal_inputs or {}).get("images", [])]
    return TeacherRolloutState(prompt_ids, image_data, [{"kind": "prompt", "turn": -1, "start": 0, "end": len(prompt_ids)}])


def append_teacher_response_text(args: Any, state: TeacherRolloutState, response_text: str, turn: int) -> None:
    ids = _tolist(_teacher_tokenizer(args)(response_text, add_special_tokens=False)["input_ids"])
    start = len(state.input_ids)
    state.input_ids.extend(ids)
    state.segments.append({"kind": "assistant", "turn": turn, "start": start, "end": len(state.input_ids)})


def append_teacher_observation_message(args: Any, state: TeacherRolloutState, message: dict, metadata: dict | None, turn: int) -> None:
    tok = _teacher_tokenizer(args)
    proc = _teacher_processor(args)
    obs_ids, obs_images = _encode_increment(tok, proc, message, metadata, args)
    bos = getattr(tok, "bos_token_id", None)
    if bos is not None and obs_ids and obs_ids[0] == bos:
        obs_ids = obs_ids[1:]
    start = len(state.input_ids)
    state.input_ids.extend(obs_ids)
    state.segments.append({"kind": "observation", "turn": turn, "start": start, "end": len(state.input_ids)})
    state.image_data.extend(obs_images)


async def _teacher_request(args: Any, payload: dict[str, Any]) -> tuple[dict, str | None, str | None]:
    if getattr(args, "teacher_model_name", None) and getattr(args, "teacher_pool_config", None):
        result, endpoint = await get_teacher_pool(args).request_json(payload, request_name="think_with_image_teacher")
        return result, endpoint.name, endpoint.url

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json(), None, args.rm_url


def _pieces_from_ids(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode(x) for x in ids]


def _pieces_from_triplets(triplets: list[list]) -> list[str]:
    pieces, prev, cur = [], "", ""
    for item in triplets:
        piece = str(item[2]) if len(item) > 2 and item[2] is not None else str(item[1])
        cur += piece
        pieces.append(cur[len(prev):])
        prev = cur
    return pieces


def _build_alignment_groups(student_pieces: list[str], teacher_pieces: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    i = j = 0
    sb = tb = ""
    sg: list[int] = []
    tg: list[int] = []
    sgs: list[list[int]] = []
    tgs: list[list[int]] = []

    def flush():
        nonlocal sb, tb, sg, tg
        if sg and tg:
            sgs.append(sg.copy())
            tgs.append(tg.copy())
        sb = tb = ""
        sg = []
        tg = []

    while i < len(student_pieces) or j < len(teacher_pieces):
        if sb == tb and sb:
            flush()
            continue
        if not sb and i < len(student_pieces):
            sb += student_pieces[i]
            sg.append(i)
            i += 1
            continue
        if not tb and j < len(teacher_pieces):
            tb += teacher_pieces[j]
            tg.append(j)
            j += 1
            continue
        if len(sb) <= len(tb):
            if i < len(student_pieces):
                sb += student_pieces[i]
                sg.append(i)
                i += 1
            elif j < len(teacher_pieces):
                tb += teacher_pieces[j]
                tg.append(j)
                j += 1
        else:
            if j < len(teacher_pieces):
                tb += teacher_pieces[j]
                tg.append(j)
                j += 1
            elif i < len(student_pieces):
                sb += student_pieces[i]
                sg.append(i)
                i += 1
    if sb == tb and sg and tg:
        flush()
    elif sg or tg:
        sgs.append(sg.copy() if sg else [])
        tgs.append(tg.copy() if tg else [])
    return sgs, tgs


async def prepare_teacher_multiturn_gold(args: Any, sample: Sample, state: TeacherRolloutState | None) -> Sample:
    full_student = torch.tensor(sample.rollout_log_probs or [], dtype=torch.float32)
    if state is None:
        sample.teacher_log_probs = full_student
        return sample
    payload = {
        "input_ids": state.input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": True,
    }
    if state.image_data:
        payload["image_data"] = state.image_data
    result, endpoint_name, endpoint_url = await _teacher_request(args, payload)
    triplets = result.get("meta_info", {}).get("input_token_logprobs", [])
    if len(triplets) != len(state.input_ids):
        raise ValueError(f"Teacher triplets length mismatch: expected {len(state.input_ids)}, got {len(triplets)}")

    assistant_segments = [x for x in state.segments if x["kind"] == "assistant"]
    teacher_triplets = []
    for seg in assistant_segments:
        teacher_triplets.extend(triplets[seg["start"] : seg["end"]])

    positions = list(sample.metadata.get("student_assistant_response_positions", []))
    response_ids = list(sample.tokens[-sample.response_length:]) if sample.response_length > 0 else []
    student_ids = [response_ids[p] for p in positions if p < len(response_ids)]
    student_lps = torch.tensor([float(full_student[p]) for p in positions if p < len(full_student)], dtype=torch.float32)
    teacher_lps = torch.tensor([float(x[0]) for x in teacher_triplets], dtype=torch.float32)

    student_pieces = _pieces_from_ids(_student_tokenizer(args), student_ids)
    teacher_pieces = _pieces_from_triplets(teacher_triplets)
    sl = min(len(student_pieces), len(student_lps))
    tl = min(len(teacher_pieces), len(teacher_lps))
    student_pieces, student_lps = student_pieces[:sl], student_lps[:sl]
    teacher_pieces, teacher_lps, teacher_triplets = teacher_pieces[:tl], teacher_lps[:tl], teacher_triplets[:tl]
    sgs, tgs = _build_alignment_groups(student_pieces, teacher_pieces)

    rebuilt = student_lps.clone()
    gold_groups = []
    for sg, tg in zip(sgs, tgs, strict=False):
        if not sg or not tg:
            continue
        gap = sum(float(student_lps[i]) for i in sg) - sum(float(teacher_lps[i]) for i in tg)
        gold_groups.append({"student_group": sg, "teacher_group": tg, "group_logprob_gap": gap})
        for i in sg:
            rebuilt[i] = student_lps[i] - gap

    teacher_full = full_student.clone()
    for pos, val in zip(positions[: len(rebuilt)], rebuilt, strict=False):
        teacher_full[pos] = val
    sample.teacher_log_probs = teacher_full
    sample.metadata["teacher_token_triplets"] = teacher_triplets
    sample.metadata["teacher_rollout_segments"] = list(state.segments)
    sample.metadata["teacher_rollout_input_ids"] = list(state.input_ids)
    sample.metadata["gold_groups"] = gold_groups
    sample.metadata["gold_teacher_endpoint"] = endpoint_name
    sample.metadata["gold_teacher_url"] = endpoint_url
    return sample
