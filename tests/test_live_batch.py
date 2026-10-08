"""Live.end_idle_many matches end_idle event by event (same time)."""
import numpy as np
from layby.live import Live

a, b = Live(120), Live(120)
for L in (a, b):
    L.tick(100.0)
ages = np.random.default_rng(0).exponential(30, 500)
for x in ages:
    a.end_idle(200.0, "g", x, 16, True)
    a.end_idle(200.0, "c", x, 16, False)
b.end_idle_many(200.0, "g", ages, 16, True)
b.end_idle_many(200.0, "c", ages, 16, False)
for k in a.km:
    assert np.allclose(a.km[k], b.km[k]), k
for k in a.acc:
    assert np.isclose(a.acc[k], b.acc[k]), k
print("batch ok")
