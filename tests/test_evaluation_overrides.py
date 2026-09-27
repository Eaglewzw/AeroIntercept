"""Evaluation overrides must survive DotDict nested attribute reads."""
import sys
from unittest.mock import MagicMock
import pytest
from aerointercept.gazebo.scripts import evaluate


def test_distance_and_speed_reach_launcher_and_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--launch', '--config',
        'configs/gazebo_feedback.yaml', '--checkpoint', 'unused.pt',
        '--initial-distance', '30', '--max-speed', '5',
        '--output', str(tmp_path/'result.json')])
    stack = MagicMock()
    def launch(args, socket):
        assert args.initial_distance == 30.
        return stack
    class Verified(Exception):
        pass
    def environment(cfg, *args, **kwargs):
        assert cfg.gazebo.task.reset_target_distance_m == 30.
        assert cfg.gazebo.action.velocity_max == 5.
        assert cfg.gazebo.task.hit_radius == .5
        raise Verified()
    monkeypatch.setattr(evaluate, 'maybe_launch', launch)
    monkeypatch.setattr(evaluate, 'GazeboInterceptEnv', environment)
    with pytest.raises(Verified):
        evaluate.main()
    stack.close.assert_called_once()
