"""Environment and System Introspection Tests.
Test & Validation Plan Section 2 (Environment).
"""

from src.utils.system_info import get_system_environment


def test_system_environment_detection():
    env = get_system_environment()
    assert "python_version" in env
    assert "platform" in env
    assert "cuda_available" in env
    assert "packages" in env
