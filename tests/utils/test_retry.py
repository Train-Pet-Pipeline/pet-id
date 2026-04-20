import pytest

from purrai_core.utils.retry import retrying


def test_retrying_succeeds_after_failures() -> None:
    counter = {"n": 0}

    @retrying(max_attempts=3, wait_seconds=0)
    def flaky() -> str:
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert counter["n"] == 3


def test_retrying_reraises_after_exhausted() -> None:
    @retrying(max_attempts=2, wait_seconds=0)
    def always_fails() -> str:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        always_fails()
