"""Qwen2-VL narrative backend tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Skip the whole module when the `[narrative]` extras (transformers +
# accelerate + qwen-vl-utils) is not installed. CI's default
# `[dev,detector,reid]` matrix does not pull transformers — intentional
# per pet-id `.github/workflows/ci.yml` header. Same pattern as
# test_bytetrack_tracker (boxmot) / test_mmpose_pose (mmpose) /
# test_osnet_reid (torchreid).
pytest.importorskip("transformers")

from purrai_core.types import NarrativeOutput  # noqa: E402

CFG = {
    "model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "device": "cpu",
    "dtype": "float32",
    "max_new_tokens": 64,
    "temperature": 0.7,
    "system_prompt": "你是一个懂宠物行为的观察者。",
}

_MODULE = "purrai_core.backends.qwen2vl_narrative"


def _make_mocks() -> tuple[MagicMock, MagicMock]:
    """Return (mock_model_cls, mock_autoproc_cls) ready for patching."""
    mock_model_cls = MagicMock()
    mock_model_inst = MagicMock()
    mock_model_inst.generate.return_value = torch.tensor([[100, 101, 102, 200, 201, 202]])
    mock_model_cls.from_pretrained.return_value = mock_model_inst

    mock_autoproc_cls = MagicMock()
    mock_processor_inst = MagicMock()
    mock_processor_inst.apply_chat_template.return_value = "chat text"
    mock_inputs = MagicMock()
    mock_inputs.input_ids.shape = (1, 3)
    mock_processor_inst.return_value.to.return_value = mock_inputs
    mock_processor_inst.batch_decode.return_value = ["猫咪正在休息"]
    mock_autoproc_cls.from_pretrained.return_value = mock_processor_inst

    return mock_model_cls, mock_autoproc_cls


def test_qwen2vl_describe_returns_narrative_output() -> None:
    """describe() returns NarrativeOutput with non-empty text and correct meta backend."""
    mock_model_cls, mock_autoproc_cls = _make_mocks()

    with (
        patch(f"{_MODULE}.Qwen2VLForConditionalGeneration", mock_model_cls),
        patch(f"{_MODULE}.AutoProcessor", mock_autoproc_cls),
    ):
        from purrai_core.backends.qwen2vl_narrative import Qwen2VLNarrative

        gen = Qwen2VLNarrative(CFG)
        frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
        result = gen.describe(frames, [])

    assert isinstance(result, NarrativeOutput)
    assert result.text  # non-empty
    assert result.meta is not None
    assert result.meta["backend"] == "qwen2-vl-2b"


def test_qwen2vl_passes_all_frames_as_multi_image() -> None:
    """describe() passes every frame (BGR→RGB) to Image.fromarray in order."""
    mock_model_cls, mock_autoproc_cls = _make_mocks()

    frames = [np.full((480, 640, 3), i * 10, dtype=np.uint8) for i in range(5)]

    with (
        patch(f"{_MODULE}.Qwen2VLForConditionalGeneration", mock_model_cls),
        patch(f"{_MODULE}.AutoProcessor", mock_autoproc_cls),
        patch(f"{_MODULE}.Image.fromarray") as mock_fromarray,
    ):
        mock_fromarray.return_value = MagicMock()

        from purrai_core.backends.qwen2vl_narrative import Qwen2VLNarrative

        gen = Qwen2VLNarrative(CFG)
        gen.describe(frames, [])

    assert mock_fromarray.call_count == len(frames)
    for i, call in enumerate(mock_fromarray.call_args_list):
        np.testing.assert_array_equal(call[0][0], frames[i][..., ::-1])
