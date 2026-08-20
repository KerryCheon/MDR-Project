from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("hybrid_lstm_sweep_core", EXPERIMENT_DIR / "core.py")
CORE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CORE
SPEC.loader.exec_module(CORE)


def test_candidate_catalog_has_one_entry_per_version_line():
    candidates = CORE.build_candidates()
    assert len(candidates) == 15
    assert len({candidate.name for candidate in candidates}) == len(candidates)
    assert {candidate.version for candidate in candidates} == {
        "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
        "v15", "v16", "v17", "v20", "v21", "v22", "v23",
    }


def test_sequence_builder_preserves_every_source_row():
    frame = pd.DataFrame(
        {
            "station_id": ["a", "a", "b"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-01"]),
            "feature": [1.0, 2.0, 7.0],
            "SMAP_sm_pm_interp": [0.1, 0.2, 0.3],
            "_row_id": [0, 1, 2],
        }
    )
    sequence, target, auxiliary, rows = CORE._build_arrays(
        frame, ("feature",), 3, np.asarray([0.4, 0.5, 0.6], dtype=np.float32)
    )
    assert sequence.shape == (3, 3, 1)
    assert sorted(rows.tolist()) == [0, 1, 2]
    assert np.allclose(sequence[0, :, 0], [1.0, 1.0, 1.0])
    assert len(target) == len(auxiliary) == len(frame)


def test_target_is_never_an_input_feature():
    for candidate in CORE.build_candidates():
        assert CORE.TARGET not in candidate.features
        assert not any(feature.lower() == CORE.TARGET.lower() for feature in candidate.features)
