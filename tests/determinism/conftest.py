"""CUDA determinism fixture — applied to every test in this directory."""
import pytest

from aiutopia.determinism.harness import configure_cuda_determinism


@pytest.fixture(autouse=True)
def _cuda_determinism() -> None:
    configure_cuda_determinism()
