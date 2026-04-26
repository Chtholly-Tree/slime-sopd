"""
LLM Judge Reward Module.

Calls an LLM API (OpenAI-compatible) to judge whether the model's answer
matches the ground truth. Designed for high-throughput concurrent requests
with exponential-backoff retry logic.

Usage:
Via custom rm path:

    --custom-rm-path examples.think_with_image.rewards.llm_judge.async_rm

Or via rm-type (after registering in rm_hub):

    --rm-type llm_judge

Required CLI arguments (set in training scripts):

    --llm-judge-api-key      API key for the LLM judge service
    --llm-judge-base-url     Base URL of the API (e.g. https://maas.devops.xiaohongshu.com/v1)
    --llm-judge-model        Model name to query (e.g. qwen3.5-35b-a3b)

Optional CLI arguments:

    --llm-judge-max-tokens   max_tokens for generation (default 32)
    --llm-judge-temperature   temperature (default 0.0)
    --llm-judge-max-retries  max retries per sample (default 3)
    --llm-judge-batch-size   max concurrent API calls per batch (default 64)
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

import aiohttp

from slime.utils.types import Sample

from .prompts import build_judge_messages, parse_judge_response

logger = logging.getLogger(__name__)

_shared_session: aiohttp.ClientSession | None = None


def _get_session() -> aiohttp.ClientSession:
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        connector = aiohttp.TCPConnector(limit=256, enable_cleanup_closed=True)
        _shared_session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=60))
    return _shared_session


def _get_config(args) -> dict[str, Any]:
    return {
        "api_key": getattr(args, "llm_judge_api_key", None),
        "base_url": getattr(args, "llm_judge_base_url", None),
        "model": getattr(args, "llm_judge_model", None),
        "max_tokens": getattr(args, "llm_judge_max_tokens", 32),
        "temperature": getattr(args, "llm_judge_temperature", 0.0),
        "max_retries": getattr(args, "llm_judge_max_retries", 3),
    }


def _get_raw_question(sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    question = (
        metadata.get("problem")
        or metadata.get("prompt")
        or metadata.get("raw_prompt")
        or metadata.get("question")
        or sample.prompt
    )
    return question if isinstance(question, str) else str(question)


def _extract_final_answer(response: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No answer."


async def _call_judge(
    session: aiohttp.ClientSession,
    config: dict[str, Any],
    messages: list[dict],
) -> tuple[str, int]:
    """Call judge API with exponential-backoff retry. Returns (response, attempts)."""
    max_retries = config["max_retries"]
    headers = { "Content-Type": "application/json"}
    payload = {
        "model": config["model"],
        "messages": messages,
        "max_tokens": config["max_tokens"],
        "temperature": config["temperature"],
    }
    for attempt in range(max_retries + 1):
        try:
            async with session.post(
                f"{config['base_url'].rstrip('/')}/v1/chat/completions", headers=headers, json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if choices := data.get("choices"):
                    return choices[0].get("message", {}).get("content", ""), attempt + 1
                return "", attempt + 1
        except Exception as e:
            if attempt >= max_retries:
                logger.warning("Judge API failed after %d attempts: %s", max_retries + 1, e)
                return "", max_retries + 1
            await asyncio.sleep(min(2**attempt, 30))

    return "", max_retries + 1


async def _evaluate_single(config: dict[str, Any], sample: Sample) -> float:
    question = _get_raw_question(sample)
    standard_answer = str(sample.label) if sample.label is not None else ""
    model_answer = _extract_final_answer(sample.response or "")

    messages = build_judge_messages(question, standard_answer, model_answer)
    response, _ = await _call_judge(_get_session(), config, messages)
    if not response:
        logger.warning("Sample %d: empty judge response, reward=0", sample.index)
        return 0.0

    return 1.0 if parse_judge_response(response) else 0.0


async def async_rm(args: Any, samples: Sample | list[Sample]) -> float | list[float]:
    """
    LLM judge reward that supports both single-sample and batch invocation.

    Args:
        args: must contain llm-judge-* fields
        samples: one Sample or a list of Samples with .prompt, .label, .response

    Returns:
        float reward for single-sample calls, or list of float rewards for batch calls
    """
    config = _get_config(args)

    missing = [k for k in ("api_key", "base_url", "model") if not config[k]]
    if missing:
        raise ValueError(
            f"Missing required config: {missing}. Set --llm-judge-api-key, --llm-judge-base-url, --llm-judge-model."
        )

    single_sample = isinstance(samples, Sample)
    sample_list = [samples] if single_sample else samples

    if not sample_list:
        return 0.0 if single_sample else []

    total = len(sample_list)
    start = time.monotonic()

    semaphore = asyncio.Semaphore(getattr(args, "llm_judge_batch_size", 64))

    async def bounded_evaluate(sample: Sample) -> float:
        async with semaphore:
            return await _evaluate_single(config, sample)

    results = await asyncio.gather(*[bounded_evaluate(sample) for sample in sample_list], return_exceptions=True)

    rewards = [0.0 if isinstance(result, Exception) else result for result in results]
    elapsed = time.monotonic() - start
    correct = sum(1 for reward in rewards if reward > 0)

    logger.info(
        "LLM judge: %d/%d correct (%.1f%%) in %.1fs (%.1f samples/sec).",
        correct,
        total,
        100 * correct / total if total else 0,
        elapsed,
        total / elapsed if elapsed else 0,
    )

    return rewards[0] if single_sample else rewards