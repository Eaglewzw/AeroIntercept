"""Optional lifecycle manager for the project-local Gazebo launcher."""

from __future__ import annotations

import os
import math
import signal
import subprocess
import time
from pathlib import Path


LAUNCHER = Path(__file__).resolve().parent / "scripts" / "launch_gazebo.sh"


class GazeboStack:
    def __init__(self, *, headless: bool, mode: str, seed: int, socket_path: str,
                 initial_distance: float = 10.0):
        if not math.isfinite(initial_distance) or initial_distance <= 0:
            raise ValueError("initial_distance must be finite and positive")
        command = [
            "bash", str(LAUNCHER), "--mode", mode, "--seed", str(seed),
            "--socket", socket_path,
            "--initial-distance", str(initial_distance),
        ]
        if headless:
            command.append("--headless")
        self.command = command
        self.socket_path = Path(socket_path)
        self._start()

    def _start(self) -> None:
        self.process = subprocess.Popen(self.command, start_new_session=True)
        started = time.monotonic()
        try:
            while True:
                code = self.process.poll()
                if code is not None:
                    raise RuntimeError(
                        f"Gazebo launcher exited with code {code}; see the launch error above. "
                        "If a map editor is open, save the world and close it before using --launch."
                    )
                # Give preflight checks time to reject a conflicting stack before
                # accepting a socket that might belong to that existing stack.
                elapsed = time.monotonic() - started
                if elapsed >= 0.5 and self.socket_path.is_socket():
                    return
                if elapsed >= 120.0:
                    raise TimeoutError(
                        f"Gazebo launcher did not create {self.socket_path} within 120 seconds; "
                        "inspect /tmp/aerointercept_gazebo logs"
                    )
                time.sleep(0.1)
        except BaseException:
            self.close()
            raise

    def restart(self) -> None:
        self.close()
        self._start()

    def close(self) -> None:
        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5.0)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def maybe_launch(args, socket_path: str) -> GazeboStack | None:
    return GazeboStack(
        headless=bool(args.headless), mode=args.mode, seed=args.seed,
        socket_path=socket_path,
        initial_distance=getattr(args, "initial_distance", None) or 10.0,
    ) if args.launch else None
