"""Qwen2-VL narrative generator backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from purrai_core.interfaces.narrative import NarrativeGenerator
from purrai_core.types import NarrativeOutput, Track


class Qwen2VLNarrative(NarrativeGenerator):
    """VLM-backed narrative generator using Qwen2-VL."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Initialise model and processor from config dict.

        Args:
            cfg: Section dict with keys: model_id, device, dtype, max_new_tokens,
                 temperature, system_prompt.
        """
        self.cfg = cfg
        self.device = str(cfg["device"])
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[str(cfg["dtype"])]
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(cfg["model_id"]), torch_dtype=dtype, device_map=self.device
        )
        processor_kwargs: dict[str, Any] = {}
        if "min_pixels" in cfg:
            processor_kwargs["min_pixels"] = int(cfg["min_pixels"])
        if "max_pixels" in cfg:
            processor_kwargs["max_pixels"] = int(cfg["max_pixels"])
        self.processor = AutoProcessor.from_pretrained(str(cfg["model_id"]), **processor_kwargs)
        self.max_new_tokens = int(cfg["max_new_tokens"])
        self.temperature = float(cfg["temperature"])
        self.system_prompt = str(cfg["system_prompt"])

    def describe(
        self,
        frames: list[np.ndarray],
        tracks_history: list[list[Track]],
    ) -> NarrativeOutput:
        """Generate a natural-language description of the pet state.

        Feeds all provided frames as a multi-image prompt so Qwen2-VL sees
        temporal context (e.g., begin/middle/end of a chapter), not just one
        instant. Callers control cost by limiting how many frames they pass.

        Args:
            frames: List of BGR frames as uint8 numpy arrays. All are shown
                to the model in order.
            tracks_history: Per-frame track lists (unused by this backend but
                required by the NarrativeGenerator interface).

        Returns:
            NarrativeOutput with generated text and backend metadata.
        """
        images = [Image.fromarray(f[..., ::-1]) for f in frames]  # BGR → RGB
        content: list[dict[str, str]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": "请描述这只宠物的状态。"})
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": content},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        output_text = self.processor.batch_decode(
            out_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0].strip()
        meta: dict[str, str] = {"backend": "qwen2-vl-2b", "num_frames": str(len(frames))}
        return NarrativeOutput(text=output_text, confidence=None, meta=meta)
