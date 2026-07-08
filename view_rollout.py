"""Watch the s-motf Go1 rollout on your Mac (which has a display).

Replays rollout_qpos.npz (the recorded poses from rollout_go1.py) in the Go1
model. Run in a Mac env that has mujoco (e.g. `robotics`). Needs the Go1 model:
    git clone https://github.com/google-deepmind/mujoco_menagerie.git

Live viewer: `python view_rollout.py`
"""

import time
import numpy as np
import mujoco
import mujoco.viewer

MODEL_XML = "mujoco_menagerie/unitree_go1/scene.xml"
QPOS = np.load("rollout_qpos.npz")["qpos"]           # [T, 19]

model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)
assert model.nq == QPOS.shape[1], f"model nq={model.nq} != qpos dim {QPOS.shape[1]}"

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():                       # loop the playback
        for t in range(len(QPOS)):
            if not viewer.is_running():
                break
            data.qpos[:] = QPOS[t]
            mujoco.mj_forward(model, data)           # kinematics only — show recorded poses
            viewer.sync()
            time.sleep(1.0 / 50)                     # 50 Hz playback
