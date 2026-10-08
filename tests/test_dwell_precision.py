"""Layby-Dwell precisions: the option is validated, and on a GPU every shipped 8-bit precision keeps the curves of
public-pool states close to bf16 (model/QUANT.md: max |dP| 0.031 for int8w and 0.082 for fp8w over 4,445 states;
fp32 itself differs from bf16 by up to 0.0135).

The GPU part needs LAYBY_DWELL_DIR (a local copy of the model) and LAYBY_STATES (a parquet of public-pool states with
columns state and kind); it is skipped without them.
"""
import os

import numpy as np
import pytest

from layby_dwell.model import PRECISIONS, QUANT

GATE = {"int8w": 0.05, "fp8w": 0.1}   # above the measured max on 4,445 states; 256 here


def test_precision_names():
    assert set(QUANT) <= set(PRECISIONS) and set(QUANT) == set(GATE)


def test_8bit_needs_gpu():
    from layby_dwell.model import Dwell
    with pytest.raises(ValueError):
        Dwell("/nonexistent", device="cpu", precision="int8w")


@pytest.mark.skipif(not (os.environ.get("LAYBY_DWELL_DIR") and os.environ.get("LAYBY_STATES")),
                    reason="needs LAYBY_DWELL_DIR and LAYBY_STATES")
def test_shipped_8bit_within_gate():
    import pandas as pd
    import torch
    from layby_dwell.bench import server
    from layby_dwell.model import Dwell
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    df = pd.read_parquet(os.environ["LAYBY_STATES"]).sample(256, random_state=0)
    states = [server(s) for s in df.state]
    kinds = np.where(df.kind.values == "workflow", "human", df.kind.values)
    ref = Dwell(os.environ["LAYBY_DWELL_DIR"], device="cuda", precision="bf16").survival(states, kinds)
    for prec in QUANT:
        S = Dwell(os.environ["LAYBY_DWELL_DIR"], device="cuda", precision=prec).survival(states, kinds)
        assert not np.isnan(S).any(), prec
        assert np.max(np.abs(S - ref)) <= GATE[prec], prec
