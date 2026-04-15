from __future__ import annotations

import torch
import torch.nn.functional as F


def _sum_group_log_probs_tensor(token_log_probs: torch.Tensor, groups: list[list[int]]) -> torch.Tensor:
    if not groups:
        return token_log_probs.new_zeros((0,))
    return torch.stack([token_log_probs[group].sum() for group in groups if group])


def _get_response_log_probs(args, batch, logits: torch.Tensor) -> list[torch.Tensor]:
    from slime.backends.megatron_utils.loss import get_responses

    student_temperature = float(getattr(args, "gold_student_temperature", 1.0))
    if student_temperature <= 0:
        raise ValueError(f"gold_student_temperature must be positive, got {student_temperature}")

    log_probs = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    ):
        if logits_chunk.numel() == 0:
            log_probs.append(torch.empty((0,), device=logits.device, dtype=logits.dtype))
            continue
        chunk_log_probs = F.log_softmax(logits_chunk / student_temperature, dim=-1)
        chunk_log_probs = chunk_log_probs.gather(dim=-1, index=tokens_chunk.unsqueeze(-1)).squeeze(-1)
        log_probs.append(chunk_log_probs)
    return log_probs


def _get_full_response_log_probs(args, batch, logits: torch.Tensor) -> list[torch.Tensor]:
    from slime.backends.megatron_utils.loss import get_responses

    student_temperature = float(getattr(args, "gold_student_temperature", 1.0))
    if student_temperature <= 0:
        raise ValueError(f"gold_student_temperature must be positive, got {student_temperature}")

    response_log_probs = []
    for logits_chunk, _tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    ):
        if logits_chunk.numel() == 0:
            response_log_probs.append(torch.empty((0, logits.size(-1)), device=logits.device, dtype=logits.dtype))
            continue
        response_log_probs.append(F.log_softmax(logits_chunk / student_temperature, dim=-1))
    return response_log_probs


def _append_tail_bin(log_probs: torch.Tensor) -> torch.Tensor:
    if log_probs.numel() == 0:
        return log_probs.new_empty((0, 1))
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail], dim=-1)


def _pad_last_dim(x: torch.Tensor, size: int, value: float) -> torch.Tensor:
    if x.size(-1) >= size:
        return x
    return F.pad(x, (0, size - x.size(-1)), value=value)


def _compute_jsd_for_log_probs(student_log_probs: torch.Tensor, teacher_log_probs: torch.Tensor) -> torch.Tensor:
    if student_log_probs.numel() == 0 or teacher_log_probs.numel() == 0:
        return student_log_probs.new_tensor(0.0)
    student_probs = student_log_probs.exp()
    teacher_probs = teacher_log_probs.exp()
    mixture = 0.5 * (student_probs + teacher_probs)
    mixture = mixture / mixture.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    mixture_log_probs = mixture.clamp_min(1e-12).log()
    jsd = 0.5 * F.kl_div(student_log_probs, mixture_log_probs, reduction="none", log_target=True).sum(dim=-1)
    jsd = jsd + 0.5 * F.kl_div(teacher_log_probs, mixture_log_probs, reduction="none", log_target=True).sum(dim=-1)
    return jsd.mean()


def _compute_hybrid_group_loss(student_group_log_probs: torch.Tensor, teacher_group_log_probs: torch.Tensor, group_match_mask: torch.Tensor) -> torch.Tensor:
    if student_group_log_probs.numel() == 0 or teacher_group_log_probs.numel() == 0:
        return student_group_log_probs.new_tensor(0.0)
    matched_mask = group_match_mask.bool()
    if not matched_mask.any():
        return student_group_log_probs.new_tensor(0.0)
    student_matched = student_group_log_probs.masked_select(matched_mask).view(student_group_log_probs.size(0), -1)
    teacher_matched = teacher_group_log_probs.masked_select(matched_mask).view(teacher_group_log_probs.size(0), -1)
    return _compute_jsd_for_log_probs(student_matched, teacher_matched)


def _build_group_sparse_distributions(args, sample_full_log_probs: torch.Tensor, sample_metadata: dict):
    student_groups = sample_metadata.get("student_groups") or []
    teacher_group_topk_ids = sample_metadata.get("teacher_group_topk_ids") or []
    teacher_group_topk_log_probs = sample_metadata.get("teacher_group_topk_log_probs") or []
    teacher_group_match_mask = sample_metadata.get("teacher_group_match_mask") or []
    student_token_ids = sample_metadata.get("student_token_ids") or []
    mapping = sample_metadata.get("teacher_to_student_vocab_mapping") or {}
    if not student_groups or not teacher_group_topk_ids or not teacher_group_topk_log_probs:
        return None, None, None
    student_rows, teacher_rows, match_rows = [], [], []
    for student_group, teacher_ids_row, teacher_log_probs_row, match_mask_row in zip(student_groups, teacher_group_topk_ids, teacher_group_topk_log_probs, teacher_group_match_mask, strict=False):
        if not student_group:
            continue
        teacher_log_probs_tensor = torch.tensor(teacher_log_probs_row, device=sample_full_log_probs.device, dtype=sample_full_log_probs.dtype)
        match_mask_tensor = torch.tensor(match_mask_row, device=sample_full_log_probs.device, dtype=torch.bool)
        mapped_student_ids = torch.tensor([mapping.get(int(teacher_id), -1) for teacher_id in teacher_ids_row], device=sample_full_log_probs.device, dtype=torch.long)
        valid_mask = mapped_student_ids >= 0
        student_row = sample_full_log_probs.new_full((len(teacher_ids_row),), float("-inf"))
        first_student_idx = int(student_group[0])
        if valid_mask.any():
            student_row[valid_mask] = sample_full_log_probs[first_student_idx, mapped_student_ids[valid_mask]]
        for student_idx in student_group[1:]:
            actual_student_token_id = int(student_token_ids[student_idx])
            student_row = student_row + sample_full_log_probs[student_idx, actual_student_token_id]
        student_rows.append(_append_tail_bin(student_row.unsqueeze(0)).squeeze(0))
        teacher_rows.append(_append_tail_bin(teacher_log_probs_tensor.unsqueeze(0)).squeeze(0))
        match_rows.append(torch.cat([match_mask_tensor, torch.ones(1, device=match_mask_tensor.device, dtype=torch.bool)]))
    if not student_rows:
        return None, None, None
    max_dim = max(row.numel() for row in student_rows + teacher_rows)
    student_rows = [_pad_last_dim(row.unsqueeze(0), max_dim, float("-inf")).squeeze(0) for row in student_rows]
    teacher_rows = [_pad_last_dim(row.unsqueeze(0), max_dim, float("-inf")).squeeze(0) for row in teacher_rows]
    match_rows = [_pad_last_dim(row.unsqueeze(0).float(), max_dim, 0.0).squeeze(0).bool() for row in match_rows]
    return torch.stack(student_rows, dim=0), torch.stack(teacher_rows, dim=0), torch.stack(match_rows, dim=0)


def _compute_ce_loss(ce_weight: float, response_log_probs: list[torch.Tensor], sum_of_sample_mean):
    if ce_weight == 0 or not response_log_probs:
        return None
    flat_log_probs = torch.cat(response_log_probs, dim=0)
    if flat_log_probs.numel() == 0:
        return None
    return -ce_weight * sum_of_sample_mean(flat_log_probs)


def _compute_distillation_loss(distill_weight: float, args, metadata: list[dict], response_log_probs: list[torch.Tensor]):
    if distill_weight == 0 or not metadata or not response_log_probs:
        return None
    per_sample_losses = []
    use_topk_hybrid = bool(getattr(args, "gold_use_topk_hybrid_loss", True))
    for sample_metadata, sample_log_probs in zip(metadata, response_log_probs, strict=False):
        if sample_metadata is None:
            continue
        if use_topk_hybrid:
            student_group_sparse, teacher_group_sparse, group_match_mask = _build_group_sparse_distributions(args, sample_log_probs, sample_metadata)
            if student_group_sparse is not None and teacher_group_sparse is not None and group_match_mask is not None:
                per_sample_losses.append(_compute_hybrid_group_loss(student_group_sparse, teacher_group_sparse, group_match_mask))
                continue
        student_groups = sample_metadata.get("student_groups") or []
        teacher_group_log_probs = sample_metadata.get("teacher_group_log_probs") or []
        if not student_groups or not teacher_group_log_probs:
            continue
        response_token_log_probs = sample_log_probs.gather(dim=-1, index=torch.tensor(sample_metadata["student_token_ids"], device=sample_log_probs.device, dtype=torch.long).unsqueeze(-1)).squeeze(-1)
        current_student_group_log_probs = _sum_group_log_probs_tensor(response_token_log_probs, student_groups)
        teacher_group_log_probs_tensor = torch.tensor(teacher_group_log_probs, device=sample_log_probs.device, dtype=sample_log_probs.dtype)
        group_count = min(current_student_group_log_probs.numel(), teacher_group_log_probs_tensor.numel())
        if group_count == 0:
            continue
        per_sample_losses.append(torch.abs(current_student_group_log_probs[:group_count] - teacher_group_log_probs_tensor[:group_count]).mean())
    if not per_sample_losses:
        return None
    return distill_weight * torch.stack(per_sample_losses).mean()


def logits_level_gold_loss(args, batch, logits, sum_of_sample_mean):
    # response_log_probs = _get_response_log_probs(args, batch, logits)
    full_response_log_probs = _get_full_response_log_probs(args, batch, logits)
    # ce_weight = float(getattr(args, "gold_cross_entropy_weight", 0.0))
    distill_weight = float(getattr(args, "gold_distillation_weight", 1.0))
    # ce_loss = _compute_ce_loss(ce_weight, response_log_probs, sum_of_sample_mean)
    distillation_loss = _compute_distillation_loss(distill_weight, args, batch.get("metadata") or [], full_response_log_probs)
    loss = None
    # if ce_loss is not None:
    #     loss = ce_loss
    if distillation_loss is not None:
        loss = distillation_loss if loss is None else loss + distillation_loss
    if loss is None:
        loss = logits.sum() * 0.0
    reported_loss = {"loss": loss.clone().detach(), "gold_distillation_loss": distillation_loss.clone().detach() if distillation_loss is not None else logits.new_tensor(0.0), "gold_ce_loss": ce_loss.clone().detach() if ce_loss is not None else logits.new_tensor(0.0)}
    return loss, reported_loss
