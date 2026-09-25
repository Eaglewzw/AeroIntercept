"""Bounded retry for idempotent Gazebo pause state requests."""


def request_pause_state(request, paused, *, attempts=3):
    """Retry missing acknowledgements; an explicit rejection remains fatal.

    The caller supplies a 2 s transport timeout, keeping three attempts within
    the bridge client's 10 s timeout. Repeating an absolute pause state is safe
    even if a previous request executed but its acknowledgement was lost.
    """
    for attempt in range(1, attempts + 1):
        acknowledged, accepted = request(bool(paused))
        if acknowledged:
            if not accepted:
                raise RuntimeError(f"Gazebo rejected pause={bool(paused)}")
            return
        print(f"GAZEBO_CONTROL_TIMEOUT pause={bool(paused)} attempt={attempt}/{attempts}", flush=True)
    raise RuntimeError(f"Gazebo pause={bool(paused)} acknowledgement missing after {attempts} attempts")
