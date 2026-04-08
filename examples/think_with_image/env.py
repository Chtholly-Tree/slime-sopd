"""
VLM multi-turn tool-calling interaction environment.

Supports:
  - image_zoom_in_tool: crop & resize a region of an image to see finer details

The model emits tool calls via <tool_call>{...}</tool_call> tags.
The env parses them, executes the corresponding tool, and returns the
result as the next observation (user message) so the model can continue
reasoning or provide a final answer.
"""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from typing import Any

try:
    import orjson
except Exception:
    orjson = None

from examples.think_with_image.base_env import BaseInteractionEnv
from examples.think_with_image.tools.image_zoom_in_tool import ImageZoomInTool

logger = logging.getLogger(__name__)

# Matches the JSON payload emitted between <tool_call> ... </tool_call> tags.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# Matches the answer emitted between <answer> ... </answer> tags.
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


class VLMToolEnv(BaseInteractionEnv):
    """
    Interaction environment for VLM tool-calling with image zoom.

    Each episode consists of at most ``max_turns`` assistant generations.
    If the model emits a supported tool call, the env executes it and returns the
    result as a user message.  When no tool call is detected or the turn limit
    is reached, the episode ends and a reward can be computed externally.
    """

    def __init__(
        self,
        *,
        max_turns: int,
        work_dir: str = "/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4/examples/think_with_image/vlm_rollout",
        supported_tool_names: list[str] | None = None,
        initial_image_paths: list[str] | None = None,
    ):
        self.max_turns = max_turns
        self.turn = 0
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        self.supported_tool_names = supported_tool_names or ["image_zoom_in_tool"]

        # Preserve original input images across env resets.
        self._initial_image_paths = list(initial_image_paths or [])

        # Lazy-initialised tool instance (created on first use to avoid import overhead)
        self._img_tool = None

        # Accumulated conversation history (list of message dicts with role + content)
        self._messages_history: list[dict[str, Any]] = []

        # All image file paths seen so far (original + zoomed), used to populate
        # multi_modal_data in observations and passed to the image tool.
        self._all_image_paths: list[str] = []

        # Tool call log for diagnostics
        self._tool_call_log: list[dict[str, Any]] = []

    def reset(self):
        self.turn = 0
        self._messages_history.clear()
        self._all_image_paths = list(self._initial_image_paths)
        self._tool_call_log.clear()
        observation: dict[str, Any] = {}
        reset_info: dict[str, Any] = {}
        return observation, reset_info

    def close(self):
        """Release tool resources if needed."""
        return

    # ------------------------------------------------------------------
    # Tool-call parsing
    # ------------------------------------------------------------------

    def _parse_tool_payload(self, raw_json: str) -> dict[str, Any] | None:
        """Parse tool payload strictly as JSON. Malformed payloads are rejected."""
        loader = orjson.loads if orjson is not None else json.loads
        try:
            return loader(raw_json)
        except Exception as exc:
            logger.warning("Failed to decode tool call payload: %s", exc)
            return None

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        """
        Parse the latest tool call payload from the assistant response.

        Supports the <tool_call>{...}</tool_call> convention used in SGLang
        multi-turn templates.
        """
        matches = list(TOOL_CALL_RE.finditer(text))
        if not matches:
            return None

        raw_json = matches[-1].group(1).strip()
        payload = self._parse_tool_payload(raw_json)
        if payload is None:
            return None

        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments") or payload.get("function", {}).get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("Tool call arguments are not valid JSON; rejecting tool call.")
                return None

        if not name:
            return None
        return {"name": name, "arguments": arguments}

    def _extract_answer(self, text: str) -> str | None:
        """Extract the answer from <answer>...</answer> tags."""
        match = ANSWER_RE.search(text)
        return match.group(1).strip() if match else None

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    @property
    def _image_tool(self):
        if self._img_tool is None:
            self._img_tool = ImageZoomInTool(work_dir=self.work_dir)
        return self._img_tool

    def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a parsed tool call and return the result."""
        name = tool_call["name"]
        arguments = tool_call.get("arguments") or {}

        try:
            if name == "image_zoom_in_tool":
                # Pass the full image list (original + zoomed) so img_idx can reference any image.
                result = self._image_tool.call(arguments=arguments, images=self._all_image_paths)
            else:
                return {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.warning("Tool '%s' raised: %s", name, exc)
            return {"error": f"Tool execution error: {exc}"}

        return result

    # ------------------------------------------------------------------
    # Environment step
    # ------------------------------------------------------------------

    def step(self, response_text: str) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        """
        Process a model (assistant) response.

        Args:
            response_text: the raw text generated by the model.

        Returns:
            observation: {"obs_str": str, "role": "tool"|"user", "multi_modal_data": {"images": [...]}}
            done: whether the episode has ended
            info: metadata including the parsed tool call and tool result
        """
        self.turn += 1
        is_final_turn = self.turn >= self.max_turns
        info: dict[str, Any] = {}

        # Record assistant message in conversation history for image tool access
        self._messages_history.append({"role": "assistant", "content": response_text})

        # Check if model provided an final answer
        answer = self._extract_answer(response_text)
        if answer:
            info["final_answer"] = answer
            info["tool_executed"] = False
            obs = {
                "obs_str": "",
                "role": "user",
                "multi_modal_data": {},
            }
            return obs, True, info

        # No <answer> tag, try to extract tool call
        tool_call = self._extract_tool_call(response_text)
        info["tool_call"] = deepcopy(tool_call)

        if not tool_call:
            # No tool call detected — prompt to use tool or provide answer
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    "No <tool_call> or <answer> detected. "
                    "If you have gathered enough visual information, put your answer in <answer></answer> tags. "
                    "Otherwise call the image_zoom_in_tool to continue reasoning."
                ),
                "role": "user",
                "multi_modal_data": {},
            }
            return obs, is_final_turn, info

        name = tool_call["name"]
        if name not in self.supported_tool_names:
            info["tool_executed"] = False
            obs = {
                "obs_str": (
                    f"Tool `{name}` is not supported. "
                    "Available tool: image_zoom_in_tool. "
                    "Call a tool via <tool_call>{\"name\": \"<name>\", \"arguments\": {...}}</tool_call>."
                ),
                "role": "user",
                "multi_modal_data": {},
            }
            return obs, is_final_turn, info

        # Execute the tool
        result = self._execute_tool(tool_call)
        info["tool_result"] = result
        info["tool_executed"] = True

        # Build the observation text from result
        new_image_paths = []
        if "image" in result:
            new_image_paths.append(result["image"])
            obs_str = (
                f"Tool `{name}` returned:\n"
                f"[Zoomed image saved: {os.path.basename(result['image'])}]"
            )
        elif "error" in result:
            obs_str = f"Tool `{name}` returned:\n{result['error']}"
        else:
            obs_str = f"Tool `{name}` returned:\n{str(result)}"

        # Record tool result message in conversation history
        self._messages_history.append({"role": "user", "content": obs_str})

        # Accumulate image paths for multi-modal encoding in the next generation turn
        self._all_image_paths.extend(new_image_paths)
        self._tool_call_log.append({"turn": self.turn, "name": name, "arguments": tool_call.get("arguments")})

        obs = {
            "obs_str": obs_str,
            "role": "tool",
            "multi_modal_data": {"images": new_image_paths} if new_image_paths else {},
        }
        return obs, is_final_turn, info


# ---------------------------------------------------------------------------
# Env factory (called by rollout.py via rollout_interaction_env_path)
# ---------------------------------------------------------------------------

def _extract_ground_truth(sample) -> str | None:
    """Resolve the ground-truth answer from label or metadata."""
    if sample is None:
        return None
    if sample.label is not None:
        return str(sample.label)
    return None


def build_env(sample=None, args=None, **_: Any):
    """
    Construct a VLMToolEnv from sample metadata and CLI args.

    Expected args attributes (set via --custom-config-path YAML):
        max_turns: int          — maximum assistant generations per episode
        vlm_rollout_work_dir: str — working directory for zoomed images (default /tmp/vlm_rollout)
    """
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError(
            "max_turns must be set via --custom-config-path in the custom config file."
        )
    work_dir = getattr(args, "vlm_rollout_work_dir", "/mnt/tidal-alsh01/dataset/redone/zengyu/xikun/slime-2.4/examples/think_with_image/vlm_rollout")
    initial_images = []
    if sample is not None and getattr(sample, "multimodal_inputs", None):
        initial_images = list(sample.multimodal_inputs.get("images") or [])
    return VLMToolEnv(max_turns=max_turns, work_dir=work_dir, initial_image_paths=initial_images)
