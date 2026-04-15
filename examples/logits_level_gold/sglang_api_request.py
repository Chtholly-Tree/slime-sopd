from __future__ import annotations

from slime.rollout.multimodal_gold import _build_teacher_prompt_ids, _get_teacher_tokenizer
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.teacher_pool import get_teacher_pool
from slime.utils.types import Sample

_GOLD_METADATA_KEY = "gold_logits"
_TOKEN_ID_MAPPING_CACHE: dict[tuple[int, int], tuple[dict[int, int], set[int], set[int]]] = {}


def get_teacher_topk_logprobs(args) -> int:
    topk = getattr(args, "gold_topk_logprobs", None)
    if topk is None:
        topk = getattr(args, "gold_teacher_topk_logprobs", None)
    return max(int(topk or 0), 0)


def build_topk_teacher_request_payload(args, sample: Sample, teacher_input_ids: list[int]) -> dict:
    topk = get_teacher_topk_logprobs(args)
    payload = {
        "input_ids": teacher_input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": 0, "skip_special_tokens": False},
        "return_logprob": True,
        "logprob_start_len": 0,
        "return_text_in_logprobs": True,
    }
    if topk > 0:
        payload["top_logprobs_num"] = topk
    teacher_images = (sample.multimodal_inputs or {}).get("images", [])
    if teacher_images:
        payload["image_data"] = [encode_image_for_rollout_engine(img) for img in teacher_images]
    return payload


def _normalize_single_topk_entry(entry):
    if entry is None:
        return []
    if isinstance(entry, dict):
        if "token_id" in entry:
            return [[float(entry.get("logprob", float("-inf"))), int(entry["token_id"]), entry.get("text")]]
        return []
    out = []
    if isinstance(entry, (list, tuple)):
        for item in entry:
            if isinstance(item, dict) and "token_id" in item:
                out.append([float(item.get("logprob", float("-inf"))), int(item["token_id"]), item.get("text")])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append([float(item[0]), int(item[1]), item[2] if len(item) > 2 else None])
    return out


def extract_teacher_topk_triplets(result: dict):
    meta_info = result.get("meta_info", {})
    value = meta_info.get("input_top_logprobs")
    if value:
        return [_normalize_single_topk_entry(entry) for entry in value]
    return []


async def _request_teacher(args, sample: Sample, payload: dict, request_name: str = "gold_teacher_topk") -> dict:
    teacher_model_name = getattr(args, "teacher_model_name", None)
    teacher_pool_config = getattr(args, "teacher_pool_config", None)
    if teacher_model_name and teacher_pool_config:
        result, endpoint = await get_teacher_pool(args).request_json(payload, request_name=request_name)
        sample.metadata["gold_teacher_endpoint"] = endpoint.name
        sample.metadata["gold_teacher_url"] = endpoint.url
        sample.metadata["gold_teacher_model_name"] = teacher_model_name
        return result

    import aiohttp

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def build_teacher_result_with_topk(args, sample: Sample, teacher_prompt_ids: list[int]) -> dict:
    teacher_tokenizer = _get_teacher_tokenizer(args)
    teacher_response_ids = teacher_tokenizer(sample.response, add_special_tokens=False)["input_ids"]
    teacher_input_ids = list(teacher_prompt_ids) + list(teacher_response_ids)
    payload = build_topk_teacher_request_payload(args, sample, teacher_input_ids)
    result = await _request_teacher(args, sample, payload)
    topk_triplets = extract_teacher_topk_triplets(result)
    result.setdefault("meta_info", {})["gold_teacher_prompt_len"] = len(teacher_prompt_ids)
    result["meta_info"]["gold_teacher_input_ids"] = teacher_input_ids
    result["meta_info"]["gold_teacher_requested_topk"] = get_teacher_topk_logprobs(args)
    if topk_triplets:
        result["meta_info"]["input_top_logprobs"] = topk_triplets
    return result


async def call_teacher_with_topk(args, sample: Sample) -> dict:
    teacher_prompt_ids = _build_teacher_prompt_ids(args, sample)
    return await build_teacher_result_with_topk(args, sample, teacher_prompt_ids)


def _sum_group_log_probs(token_log_probs: list[float], groups: list[list[int]]) -> list[float]:
    return [sum(float(token_log_probs[idx]) for idx in group) for group in groups if group]


def _build_group_texts(pieces: list[str], groups: list[list[int]]) -> list[str]:
    return ["".join(pieces[idx] for idx in group) for group in groups if group]


def _build_group_topk_metadata(
    teacher_token_topk_ids: list[list[int]],
    teacher_token_topk_log_probs: list[list[float]],
    teacher_groups: list[list[int]],
) -> tuple[list[list[int]], list[list[float]], list[list[bool]]]:
    group_topk_ids, group_topk_log_probs, group_match_mask = [], [], []
    for teacher_group in teacher_groups:
        if not teacher_group:
            continue
        first_idx = teacher_group[0]
        ids_row = list(teacher_token_topk_ids[first_idx])
        log_probs_row = list(teacher_token_topk_log_probs[first_idx])
        for teacher_idx in teacher_group[1:]:
            if not teacher_token_topk_log_probs[teacher_idx]:
                continue
            continuation_log_prob = float(teacher_token_topk_log_probs[teacher_idx][0])
            log_probs_row = [lp + continuation_log_prob for lp in log_probs_row]
        group_topk_ids.append(ids_row)
        group_topk_log_probs.append(log_probs_row)
        group_match_mask.append([False] * len(ids_row))
    return group_topk_ids, group_topk_log_probs, group_match_mask


def _build_token_id_mapping(student_tokenizer, teacher_tokenizer) -> tuple[dict[int, int], set[int], set[int]]:
    cache_key = (id(student_tokenizer), id(teacher_tokenizer))
    cached = _TOKEN_ID_MAPPING_CACHE.get(cache_key)
    if cached is not None:
        return cached

    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_token_to_id = dict(student_vocab.items())

    mapping: dict[int, int] = {}
    teacher_matched_ids: set[int] = set()
    student_matched_ids: set[int] = set()
    for token_str, teacher_id in teacher_vocab.items():
        student_id = student_token_to_id.get(token_str)
        if student_id is None:
            continue
        mapping[teacher_id] = student_id
        teacher_matched_ids.add(teacher_id)
        student_matched_ids.add(student_id)
    result = (mapping, teacher_matched_ids, student_matched_ids)
    _TOKEN_ID_MAPPING_CACHE[cache_key] = result
    return result
