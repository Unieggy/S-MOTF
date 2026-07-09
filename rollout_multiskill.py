"""Phase 4 (multi-skill) — roll out the multi-skill s-motf for ONE skill and
record the pose trajectory (for rendering a per-skill gif).

Runs checkpoint_multi.pt in the given skill's env, telling it the skill via the
[velocity, skill-id] command. Saves rollout_<name>.npz (qpos over time).

Run in go2-rl (needs cpu torch):
  python rollout_multiskill.py Go1JoystickFlatTerrain 0 rollout_walk.npz
  python rollout_multiskill.py Go1Footstand           1 rollout_footstand.npz
  python rollout_multiskill.py Go1Getup               2 rollout_getup.npz
"""

import sys
import numpy as np
import jax
import jax.numpy as jp
import torch

from mujoco_playground import registry
from smotf import load_config
from smotf.model.smotf import SMoTF

ENV_NAME = sys.argv[1]
SKILL_ID = int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else "rollout_qpos.npz"
NUM_SKILLS = 3
STEPS = 300


def quat_to_rot(q):
    w, x, y, z = q
    return jp.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

def quat_to_rpy(q):
    w, x, y, z = q
    return jp.array([
        jp.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
        jp.arcsin(jp.clip(2*(w*y-z*x), -1.0, 1.0)),
        jp.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)),
    ])

def base_from_data(d):
    q = d.qpos[3:7]; R = quat_to_rot(q)
    return np.asarray(jp.concatenate([quat_to_rpy(q), d.qvel[3:6], R.T @ d.qvel[0:3], R.T @ jp.array([0., 0., -1.])]))

def legs_from_data(d):
    return np.asarray(jp.concatenate([d.qpos[7:19], d.qvel[6:18]]))


def main():
    env = registry.load(ENV_NAME)
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    cfg = load_config("configs/multiskill.yaml")
    model = SMoTF(cfg)
    model.load_state_dict(torch.load("checkpoint_multi.pt", map_location="cpu"))
    model.eval()
    stats = torch.load("norm_stats.pt")
    nrm = lambda k, x: (torch.tensor(np.asarray(x), dtype=torch.float32) - stats[k]["mean"]) / stats[k]["std"]
    dnrm = lambda k, x: x * stats[k]["std"] + stats[k]["mean"]

    onehot = np.zeros(NUM_SKILLS, np.float32); onehot[SKILL_ID] = 1.0

    state = reset(jax.random.PRNGKey(0))
    qpos_traj = []
    for t in range(STEPS):
        d = state.data
        qpos_traj.append(np.asarray(d.qpos))

        if "command" in state.info:
            vel = np.asarray(state.info["command"], np.float32)
            if vel.shape[-1] != 3:
                vel = np.zeros(3, np.float32)
        else:
            vel = np.zeros(3, np.float32)
        command = np.concatenate([vel, onehot])                     # [6]
        contacts = np.asarray(state.info["last_contact"], np.float32) if "last_contact" in state.info else np.zeros(4, np.float32)

        ctx = {
            "base":     nrm("base", base_from_data(d))[None],
            "legs":     nrm("legs", legs_from_data(d))[None],
            "contacts": nrm("contacts", contacts)[None],
            "command":  nrm("command", command)[None],
        }
        with torch.no_grad():
            a = model.act(ctx).squeeze(0)
        state = step(state, jp.asarray(dnrm("action", a).numpy()))

    np.savez(OUT, qpos=np.array(qpos_traj))
    print("saved", OUT, np.array(qpos_traj).shape)


if __name__ == "__main__":
    main()
