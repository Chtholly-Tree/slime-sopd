"""
Tool registry for VLM multi-turn tool-calling rollout.

Exposes tool schemas used by the dataset and the chat template,
and provides factory functions for instantiating tool objects.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI-compatible, used in apply_chat_template)
# ---------------------------------------------------------------------------

TOOL_IMAGE_ZOOM_IN = {
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": (
            "Zoom in on a specific region of an image by cropping and resizing it. "
            "Use this when you need to see finer visual details in a small area "
            "(e.g. reading small text, identifying small objects, counting密集的细节). "
            "Coordinates are in 0-1000 scale relative to image size."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": (
                        "Bounding box as [x1, y1, x2, y2]. "
                        "(x1, y1) is top-left, (x2, y2) is bottom-right. "
                        "Values are in 0-1000 scale (relative to image width/height)."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": "Name or label of the object/region you are zooming into.",
                },
                "img_idx": {
                    "type": "number",
                    "description": "Index of the image to zoom (starting from 0).",
                },
            },
            "required": ["bbox_2d", "label", "img_idx"],
        },
    },
}


def get_tools() -> list[dict[str, Any]]:
    """Return the list of tool schemas used in the dataset / chat template."""
    return [TOOL_IMAGE_ZOOM_IN]


def get_tool_names() -> list[str]:
    """Return the names of all registered tools."""
    return [TOOL_IMAGE_ZOOM_IN["function"]["name"]]


# ---------------------------------------------------------------------------
# Tool instances (lazily created in the env)
# ---------------------------------------------------------------------------

def make_image_zoom_tool(work_dir: str):
    """Instantiate the image zoom tool."""
    from examples.think_with_image.tools.image_zoom_in_tool import ImageZoomInTool
    return ImageZoomInTool(work_dir=work_dir)
