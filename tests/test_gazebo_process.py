from pathlib import Path
from unittest.mock import MagicMock
import pytest
from aerointercept.gazebo import process


def test_launcher_failure_does_not_wait_for_bridge(monkeypatch):
    child = MagicMock()
    child.poll.return_value = 1
    monkeypatch.setattr(process.subprocess, 'Popen', lambda *a, **kw: child)
    sleeps = []
    monkeypatch.setattr(process.time, 'sleep', sleeps.append)
    with pytest.raises(RuntimeError, match='save the world and close'):
        process.GazeboStack(headless=True, mode='stop_go', seed=0,
                            socket_path='/tmp/nonexistent-test.sock')
    assert sleeps == []


def test_distance_is_preserved_when_restarting(monkeypatch):
    child = MagicMock()
    child.poll.return_value = None
    calls = []
    def launch(command, **kwargs):
        calls.append(list(command))
        return child
    monkeypatch.setattr(process.subprocess, 'Popen', launch)
    monkeypatch.setattr(Path, 'is_socket', lambda self: True)
    times = iter([0., 1., 2., 3.])
    monkeypatch.setattr(process.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(process.GazeboStack, 'close', lambda self: None)
    stack = process.GazeboStack(headless=True, mode='stop_go', seed=0,
                                socket_path='/tmp/test.sock', initial_distance=30.)
    stack.restart()
    assert calls[0] == calls[1]
    assert calls[0][calls[0].index('--initial-distance')+1] == '30.0'
