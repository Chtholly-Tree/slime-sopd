"""
Custom image zoom-in tool for VLM multi-turn tool-calling rollout.

This tool crops and resizes a region of an image to allow finer visual inspection.
Unlike the qwen-agent version, this tool directly receives the image list
(including newly zoomed images from previous tool calls) as a parameter,
avoiding issues with message history misalignment.

Coordinates are in 0-1000 scale relative to image size.
"""

from __future__ import annotations

import math
import os
import uuid
from io import BytesIO
from math import ceil, floor
from typing import Any, List, Union

import requests
from PIL import Image


class ImageZoomInTool:
    """
    Zoom in on a specific region of an image by cropping it based on a bounding box.

    Args:
        work_dir: Directory to save zoomed images.
        min_pixels: Minimum pixel count for resized images (default: 256 * 32 * 32).
        max_pixels: Maximum pixel count for resized images (default: 12845056).
    """

    def __init__(
        self,
        work_dir: str = "/tmp/vlm_rollout",
        min_pixels: int = 256 * 32 * 32,
        max_pixels: int = 12845056,
    ):
        self.work_dir = work_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        os.makedirs(work_dir, exist_ok=True)

    description = (
        "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox). "
        "Use this when you need to see finer details (small text, fine-grained objects, dense areas). "
        "Coordinates are in 0-1000 scale relative to the image size."
    )

    parameters = {
        "type": "object",
        "properties": {
            "bbox_2d": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": (
                    "The bounding box of the region to zoom in, as [x1, y1, x2, y2], "
                    "where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner. "
                    "Values are in 0-1000 scale (relative to image width/height)."
                ),
            },
            "label": {
                "type": "string",
                "description": "The name or label of the object/region you are zooming into.",
            },
            "img_idx": {
                "type": "number",
                "description": "The index of the image to zoom (starting from 0).",
            },
        },
        "required": ["bbox_2d", "label", "img_idx"],
    }

    # Image resizing functions (copied from qwen-vl-utils)
    def _round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    def _ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    def _floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    def _smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 32,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> tuple[int, int]:
        """Smart resize image dimensions based on factor and pixel constraints."""
        min_pixels = min_pixels or self.min_pixels
        max_pixels = max_pixels or self.max_pixels
        h_bar = max(factor, self._round_by_factor(height, factor))
        w_bar = max(factor, self._round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self._floor_by_factor(height / beta, factor)
            w_bar = self._floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self._ceil_by_factor(height * beta, factor)
            w_bar = self._ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def _maybe_resize_bbox(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        img_width: int,
        img_height: int,
    ) -> list[float]:
        """Resize bbox to ensure it's at least 32x32 pixels."""
        left = max(0, left)
        top = max(0, top)
        right = min(img_width, right)
        bottom = min(img_height, bottom)

        height = bottom - top
        width = right - left
        if height < 32 or width < 32:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = 32 / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)

            # Ensure the resized bbox is within image bounds
            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(img_width, new_right)
            new_bottom = min(img_height, new_bottom)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            if new_height > 32 and new_width > 32:
                return [new_left, new_top, new_right, new_bottom]
        return [left, top, right, bottom]

    def _load_image(self, image_arg: str) -> Image.Image | None:
        """Load an image from a file path, URL, or relative path."""
        try:
            if image_arg.startswith("file://"):
                image_arg = image_arg[len("file://") :]

            if image_arg.startswith("http"):
                response = requests.get(image_arg)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            elif os.path.exists(image_arg):
                return Image.open(image_arg)
            else:
                # Try relative to work_dir
                relative_path = os.path.join(self.work_dir, image_arg)
                if os.path.exists(relative_path):
                    return Image.open(relative_path)
                return None
        except Exception:
            return None

    def call(
        self,
        arguments: dict[str, Any],
        images: list[str],
    ) -> dict[str, Any]:
        """
        Execute the image zoom tool.

        Args:
            arguments: Tool arguments containing:
                - bbox_2d: [x1, y1, x2, y2] in 0-1000 scale
                - label: description of the region being zoomed
                - img_idx: index of the image in the images list
            images: List of image file paths/URLs (includes all original and zoomed images)

        Returns:
            dict with either:
                - "image": path to the zoomed image file (on success)
                - "error": error message (on failure)
        """
        img_idx = arguments.get("img_idx")
        bbox = arguments.get("bbox_2d")
        label = arguments.get("label", "unknown")

        # Validate inputs
        if img_idx is None or bbox is None:
            return {"error": "Missing required parameters: img_idx and bbox_2d are required."}

        if not isinstance(images, list) or img_idx >= len(images):
            return {"error": f"Invalid img_idx {img_idx}: image list has {len(images) if isinstance(images, list) else 0} images."}

        try:
            # Load the image
            image_arg = images[img_idx]
            image = self._load_image(image_arg)
            if image is None:
                return {"error": f"Failed to load image at index {img_idx}: {image_arg}"}

            # Validate and crop bbox
            img_width, img_height = image.size
            rel_x1, rel_y1, rel_x2, rel_y2 = bbox
            abs_x1 = rel_x1 / 1000.0 * img_width
            abs_y1 = rel_y1 / 1000.0 * img_height
            abs_x2 = rel_x2 / 1000.0 * img_width
            abs_y2 = rel_y2 / 1000.0 * img_height

            validated_bbox = self._maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
            left, top, right, bottom = validated_bbox

            # Crop the image
            cropped_image = image.crop((left, top, right, bottom))

            # Resize according to smart_resize logic
            new_w, new_h = self._smart_resize(
                int(bottom - top),
                int(right - left),
                factor=32,
            )
            cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)

            # Save the zoomed image
            output_path = os.path.abspath(os.path.join(self.work_dir, f"{uuid.uuid4()}.png"))
            cropped_image.save(output_path)

            return {"image": output_path, "label": label}

        except Exception as e:
            return {"error": f"Tool execution error: {str(e)}"}
