import pytest

from aerointercept.gazebo.world_control import request_pause_state


@pytest.mark.parametrize("paused", [False, True])
def test_missing_acknowledgement_retries_same_absolute_state(paused):
    calls = []

    def request(value):
        calls.append(value)
        return (len(calls) == 2, True)

    request_pause_state(request, paused)
    assert calls == [paused, paused]


def test_service_rejection_does_not_retry_or_report_success():
    calls = []

    def request(value):
        calls.append(value)
        return True, False

    with pytest.raises(RuntimeError, match="rejected"):
        request_pause_state(request, True)
    assert len(calls) == 1


def test_unavailable_service_remains_fatal_after_bounded_attempts():
    calls = []

    def request(value):
        calls.append(value)
        return False, False

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        request_pause_state(request, False)
    assert len(calls) == 3
