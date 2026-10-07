"""Gait force-replay: Wang dataset adapter, COP-inferred toe bending, chronological replay.

See ``docs/gait_replay.md`` for conventions, citations, and limitations.
"""

from __future__ import annotations

from compliance_fem.gait.replay import GaitReplayResult, replay_stance
from compliance_fem.gait.wang_io import (
    TrialSelection,
    discover_trials,
    parse_wang_trial_name,
    read_opensim_mot,
    read_opensim_trc,
)

__all__ = [
    "GaitReplayResult",
    "TrialSelection",
    "discover_trials",
    "parse_wang_trial_name",
    "read_opensim_mot",
    "read_opensim_trc",
    "replay_stance",
]
