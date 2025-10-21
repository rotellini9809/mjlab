"""Shared test fixtures and utilities."""

import os

import pytest
import torch
import warp as wp


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
  """Configure test environment settings automatically for all tests."""
  wp.config.quiet = True


def get_test_device() -> str:
  """Get device for testing, preferring CUDA if available.

  Can be overridden with FORCE_CPU=1 environment variable to test
  CPU-only behavior on GPU machines.
  """
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"
