import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def toy():
    """Two taste clusters: users 0-9 watch items 0-9, users 10-19 watch items 10-19."""
    rng = np.random.default_rng(0)
    rows = []
    t = 0
    for user in range(20):
        pool = range(0, 10) if user < 10 else range(10, 20)
        for item in rng.permutation(list(pool))[:7]:
            rows.append((user, int(item), 5, t))
            t += 1
    frame = pd.DataFrame(rows, columns=["user", "item", "rating", "timestamp"])
    frame["liked"] = 1
    genres = [("Comedy",)] * 10 + [("Horror",)] * 10
    return frame, genres, 20, 20
