import sys
from pathlib import Path

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from shared_utils import get_layer_module  # noqa: E402


class SelfAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(4, 4, bias=False)


class LinearAttentionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_attn = nn.Module()
        self.linear_attn.out_proj = nn.Linear(4, 4, bias=False)


def test_attention_output_projection_self_attention() -> None:
    layer = SelfAttentionLayer()

    assert get_layer_module(layer, "attn.o_proj") is layer.self_attn.o_proj


def test_attention_output_projection_linear_attention() -> None:
    layer = LinearAttentionLayer()

    assert get_layer_module(layer, "attn.o_proj") is layer.linear_attn.out_proj


def test_missing_component_returns_none() -> None:
    layer = nn.Module()

    assert get_layer_module(layer, "attn.o_proj") is None
