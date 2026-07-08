"""Phase 3 — load recorded Go1 teacher rollouts into smotf's episode format.

Reads go1_data.npz (produced by collect_go1.py) and returns a list of episode
dicts of torch tensors — the SAME format generate_synthetic_data produces — so
PlayWindowDataset / train.py work unchanged, just on real data.
"""

import numpy as np
import torch


def load_go1_episodes(path="go1_data.npz"):
    d = np.load(path)
    fields = list(d.files)                         # base, legs, contacts, command, action, s_next
    n_traj = d[fields[0]].shape[0]
    return [
        {k: torch.tensor(d[k][i], dtype=torch.float32) for k in fields}
        for i in range(n_traj)
    ]
