from __future__ import annotations

from argparse import Namespace

import torch

from slime.rollout.multimodal_gold import (
    _apply_response_prefix_loss_mask,
    _build_alignment_groups,
    _decode_student_response_ids,
    _extract_teacher_response_triplets,
    _get_student_tokenizer,
    _get_teacher_tokenizer,
    _to_canonical_pieces_from_ids,
    _to_canonical_pieces_from_triplets,
)
from slime.utils.types import Sample

from examples.logits_level_gold.sglang_api_request import (
    _GOLD_METADATA_KEY,
    _build_group_texts,
    _build_group_topk_metadata,
    _build_token_id_mapping,
    _sum_group_log_probs,
)


def _normalize_outcome_rewards(args: Namespace, raw_scores: list[float]) -> list[float]:
    rewards = torch.tensor(raw_scores, dtype=torch.float32)

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


def build_gold_train_metadata(args, sample: Sample, teacher_result: dict) -> dict:
    prompt_len = int(teacher_result["meta_info"]["gold_teacher_prompt_len"])
    response_len = len(teacher_result["meta_info"]["gold_teacher_input_ids"]) - prompt_len
    teacher_triplets = _extract_teacher_response_triplets(teacher_result, prompt_len, response_len)
    teacher_log_probs = [float(item[0]) for item in teacher_triplets]
    student_rollout_log_probs = [float(x) for x in (sample.rollout_log_probs or [])]
    student_token_ids = _decode_student_response_ids(sample)
    student_tokenizer = _get_student_tokenizer(args.hf_checkpoint)
    teacher_tokenizer = _get_teacher_tokenizer(args)
    teacher_token_ids = list(teacher_result["meta_info"]["gold_teacher_input_ids"][prompt_len:])
    student_pieces = _to_canonical_pieces_from_ids(student_tokenizer, student_token_ids)
    teacher_pieces = _to_canonical_pieces_from_triplets(teacher_tokenizer, teacher_token_ids)
    student_len = min(len(student_token_ids), len(student_pieces), len(student_rollout_log_probs))
    teacher_len = min(len(teacher_token_ids), len(teacher_pieces), len(teacher_log_probs), len(teacher_triplets))
    student_token_ids = student_token_ids[:student_len]
    student_pieces = student_pieces[:student_len]
    student_rollout_log_probs = student_rollout_log_probs[:student_len]
    teacher_token_ids = teacher_token_ids[:teacher_len]
    teacher_pieces = teacher_pieces[:teacher_len]
    teacher_log_probs = teacher_log_probs[:teacher_len]
    teacher_triplets = teacher_triplets[:teacher_len]
    student_groups, teacher_groups = _build_alignment_groups(student_pieces, teacher_pieces)
    paired_student_groups, paired_teacher_groups = [], []
    for student_group, teacher_group in zip(student_groups, teacher_groups, strict=False):
        if student_group and teacher_group:
            paired_student_groups.append(student_group)
            paired_teacher_groups.append(teacher_group)
    student_group_log_probs = _sum_group_log_probs(student_rollout_log_probs, paired_student_groups)
    teacher_group_log_probs = _sum_group_log_probs(teacher_log_probs, paired_teacher_groups)
    student_group_texts = _build_group_texts(student_pieces, paired_student_groups)
    teacher_group_texts = _build_group_texts(teacher_pieces, paired_teacher_groups)
    teacher_token_topk_ids, teacher_token_topk_log_probs = [], []
    topk_triplets = (teacher_result.get("meta_info", {}) or {}).get("input_top_logprobs") or []
    if topk_triplets:
        topk_triplets = topk_triplets[prompt_len : prompt_len + teacher_len]
        for token_triplet, fallback_triplet in zip(topk_triplets, teacher_triplets, strict=False):
            ids_row = [int(item[1]) for item in token_triplet]
            log_probs_row = [float(item[0]) for item in token_triplet]
            actual_token_id = int(fallback_triplet[1])
            actual_log_prob = float(fallback_triplet[0])
            if actual_token_id not in ids_row:
                ids_row = [actual_token_id] + ids_row
                log_probs_row = [actual_log_prob] + log_probs_row
            teacher_token_topk_ids.append(ids_row)
            teacher_token_topk_log_probs.append(log_probs_row)
    teacher_group_topk_ids, teacher_group_topk_log_probs, teacher_group_match_mask = [], [], []
    if teacher_token_topk_ids and teacher_token_topk_log_probs:
        teacher_group_topk_ids, teacher_group_topk_log_probs, teacher_group_match_mask = _build_group_topk_metadata(
            teacher_token_topk_ids, teacher_token_topk_log_probs, paired_teacher_groups
        )
    vocab_mapping, teacher_matched_ids, student_matched_ids = _build_token_id_mapping(student_tokenizer, teacher_tokenizer)
    teacher_token_match_mask = (
        [[int(tok_id) in teacher_matched_ids for tok_id in row] for row in teacher_token_topk_ids]
        if teacher_token_topk_ids
        else []
    )
    gold_group_log_probs = [
        {
            "student_log_prob": s_lp,
            "teacher_log_prob": t_lp,
            "gap": s_lp - t_lp,
            "student_group": s_g,
            "teacher_group": t_g,
            "student_text": s_t,
            "teacher_text": t_t,
        }
        for s_lp, t_lp, s_g, t_g, s_t, t_t in zip(
            student_group_log_probs,
            teacher_group_log_probs,
            paired_student_groups,
            paired_teacher_groups,
            student_group_texts,
            teacher_group_texts,
            strict=False,
        )
    ]
    return {
        "algorithm": "logits_level_gold",
        "student_hf_checkpoint": args.hf_checkpoint,
        "student_token_ids": student_token_ids,
        "teacher_token_ids": teacher_token_ids,
        "student_groups": paired_student_groups,
        "teacher_groups": paired_teacher_groups,
        "teacher_log_probs": teacher_log_probs,
        "teacher_token_triplets": teacher_triplets,
        "student_pieces": student_pieces,
        "teacher_pieces": teacher_pieces,
        "student_group_log_probs": student_group_log_probs,
        "teacher_group_log_probs": teacher_group_log_probs,
        "student_group_texts": student_group_texts,
        "teacher_group_texts": teacher_group_texts,
        "gold_group_log_probs": gold_group_log_probs,
        "teacher_token_topk_ids": teacher_token_topk_ids,
        "teacher_token_topk_log_probs": teacher_token_topk_log_probs,
        "teacher_token_match_mask": teacher_token_match_mask,
        "teacher_group_topk_ids": teacher_group_topk_ids,
        "teacher_group_topk_log_probs": teacher_group_topk_log_probs,
        "teacher_group_match_mask": teacher_group_match_mask,
        "teacher_to_student_vocab_mapping": vocab_mapping,
        "teacher_matched_ids": sorted(teacher_matched_ids),
        "student_matched_ids": sorted(student_matched_ids),
    }


def post_process_logits_level_gold(args: Namespace, samples: list[Sample], **kwargs):
    raw_scores = []
    for sample in samples:
        gold_state = sample.metadata.get(_GOLD_METADATA_KEY)
        if gold_state is None:
            raise ValueError("Missing async GOLD state on sample.metadata['gold_logits']")
        if sample.reward is None:
            raise ValueError("Missing reward on sample for logits-level GOLD post processing")
        raw_reward = float(sample.reward)
        teacher_result = gold_state["teacher_result"]
        sample.train_metadata = build_gold_train_metadata(args, sample, teacher_result)
        sample.metadata["gold_raw_reward"] = raw_reward
        sample.metadata["teacher_token_triplets"] = sample.train_metadata["teacher_token_triplets"]
        sample.metadata["gold_group_log_probs"] = sample.train_metadata["gold_group_log_probs"]
        _apply_response_prefix_loss_mask(args, sample)
        raw_scores.append(raw_reward)
    processed_rewards = _normalize_outcome_rewards(args, raw_scores)
    return raw_scores, processed_rewards
