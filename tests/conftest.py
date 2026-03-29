"""Shared fixtures for fused-turboquant tests."""

import pytest
import torch


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(params=[64, 128, 256])
def head_dim(request):
    return request.param


@pytest.fixture(params=[2, 3, 4])
def bits(request):
    return request.param
