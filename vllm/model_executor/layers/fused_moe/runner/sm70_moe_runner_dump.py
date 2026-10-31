# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 MoE-runner tensor dump diagnostics.

Owns the ``vllm.sm70_moe_runner_dump`` custom op and its layer/label gating.
The generic MoE runner only calls :func:`dump_moe_runner_tensor` at its
capture points; the call is a pass-through unless the ``moe_runner``
diagnostic channel requests the tensor.
"""

import torch

from vllm.diagnostics import (
    diagnostic_channel,
    flush_layer_buffers,
    layer_dump_requested,
    legacy_channel,
    record_layer_tensor,
)
from vllm.utils.torch_utils import direct_register_custom_op

_SM70_MOE_RUNNER_DUMP_BUFFERS = legacy_channel("moe_runner").buffers
_SM70_MOE_RUNNER_DUMP_META = legacy_channel("moe_runner").metadata


def _sm70_extract_moe_layer_idx(layer_name: str) -> int:
    parts = layer_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return -1
    return -1


def _sm70_moe_runner_dump_requested(layer_idx: int, label: str) -> bool:
    return layer_dump_requested(layer_idx, label, moe=True)


def _sm70_moe_runner_dump_token_count_allowed(tensor: torch.Tensor) -> bool:
    return diagnostic_channel("moe_runner").policy.allows_tokens(tensor)


def _sm70_moe_runner_dump_impl(
    tensor: torch.Tensor, label: str, layer_idx: int
) -> torch.Tensor:
    return record_layer_tensor(tensor, label, layer_idx, "moe_runner", moe=True)


def _sm70_moe_runner_dump_fake(
    tensor: torch.Tensor,
    label: str,
    layer_idx: int,
) -> torch.Tensor:
    return tensor


direct_register_custom_op(
    op_name="sm70_moe_runner_dump",
    op_func=_sm70_moe_runner_dump_impl,
    mutates_args=[],
    fake_impl=_sm70_moe_runner_dump_fake,
)


def dump_moe_runner_tensor(
    tensor: torch.Tensor,
    label: str,
    layer_name: str,
) -> torch.Tensor:
    """Return ``tensor``, recorded through the dump op when requested."""
    layer_idx = _sm70_extract_moe_layer_idx(layer_name)
    if _sm70_moe_runner_dump_requested(layer_idx, label):
        tensor = torch.ops.vllm.sm70_moe_runner_dump(tensor, label, layer_idx)
    return tensor


def dump_sm70_moe_runner_graph_buffers(step: int, stage: str) -> None:
    flush_layer_buffers(step, stage, moe=True)
