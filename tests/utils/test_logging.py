import json

from purrai_core.utils.logging import get_logger


def test_get_logger_emits_json(capsys) -> None:
    logger = get_logger("test-json-logger")
    logger.info("hello world", extra={"custom_key": "custom_value"})
    captured = capsys.readouterr()
    # Parse the first line as JSON
    lines = [ln for ln in captured.out.splitlines() if ln]
    assert len(lines) >= 1
    payload = json.loads(lines[0])
    assert payload["message"] == "hello world"
    assert payload["levelname"] == "INFO"
    assert payload["name"] == "test-json-logger"


def test_get_logger_idempotent() -> None:
    a = get_logger("test-idempotent")
    b = get_logger("test-idempotent")
    # Same logger, handlers not duplicated
    assert a is b
    assert len(a.handlers) == 1
