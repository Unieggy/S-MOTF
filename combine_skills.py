"""Merge per-skill .npz files into go1_multiskill.npz (concatenate episodes).

Run in either env (numpy only):
  python combine_skills.py go1_walk.npz go1_footstand.npz go1_getup.npz
"""

import sys
import numpy as np

files = sys.argv[1:] or ["go1_walk.npz", "go1_footstand.npz", "go1_getup.npz"]
data = [np.load(f) for f in files]
keys = data[0].files
merged = {k: np.concatenate([d[k] for d in data], axis=0) for k in keys}  # stack episodes
np.savez("go1_multiskill.npz", **merged)
print("saved go1_multiskill.npz", {k: v.shape for k, v in merged.items()})
