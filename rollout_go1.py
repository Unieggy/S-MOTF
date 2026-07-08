"""Phase 4 — deploy s-motf in the Go1 MuJoCo env and see how it does.

Runs in the go2-rl env (it needs the Go1 env). s-motf runs on CPU (tiny model),
so this env also needs CPU torch:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install imageio imageio-ffmpeg

Closed loop: observe -> normalize -> s-motf.act() -> de-normalize -> env.step().
Prints survival / forward-velocity metrics and renders rollout.mp4 (kubectl cp
it to your Mac to watch).
"""

import numpy as np
import jax
import jax.numpy as jp
import torch

from mujoco_playground import registry
from smotf import load_config
from smotf.model.smotf import SMoTF

ENV_NAME = "Go1JoystickFlatTerrain"
STEPS = 300
COMMAND = np.array([1.0, 0.0, 0.0], dtype=np.float32)   # walk forward (vx, vy, wz)


# --- base-frame extraction (same convention as collect_go1.py) ---
def quat_to_rot(q):
    w, x, y, z = q
    return jp.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
    ])

def quat_to_rpy(q):
    w, x, y, z = q
    roll  = jp.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = jp.arcsin(jp.clip(2*(w*y-z*x), -1.0, 1.0))
    yaw   = jp.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return jp.array([roll, pitch, yaw])

def base_from_data(data):
    q = data.qpos[3:7]; R = quat_to_rot(q)
    return np.asarray(jp.concatenate([
        quat_to_rpy(q), data.qvel[3:6], R.T @ data.qvel[0:3], R.T @ jp.array([0., 0., -1.])
    ]))

def legs_from_data(data):
    return np.asarray(jp.concatenate([data.qpos[7:19], data.qvel[6:18]]))


def main():
    env = registry.load(ENV_NAME)
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)

    cfg = load_config()
    model = SMoTF(cfg)
    model.load_state_dict(torch.load("checkpoint_go1.pt", map_location="cpu"))
    model.eval()
    stats = torch.load("norm_stats.pt")

    def nrm(k, x):
        return (torch.tensor(np.asarray(x), dtype=torch.float32) - stats[k]["mean"]) / stats[k]["std"]
    def dnrm(k, x):
        return x * stats[k]["std"] + stats[k]["mean"]

    state = reset(jax.random.PRNGKey(0))
    states = [state]
    heights, fwd_vels, qpos_traj = [], [], []

    for t in range(STEPS):
        d = state.data
        qpos_traj.append(np.asarray(d.qpos))
        base = base_from_data(d)
        ctx = {
            "base":     nrm("base", base)[None],
            "legs":     nrm("legs", legs_from_data(d))[None],
            "contacts": nrm("contacts", np.asarray(state.info["last_contact"], dtype=np.float32))[None],
            "command":  nrm("command", COMMAND)[None],
        }
        with torch.no_grad():
            a_norm = model.act(ctx).squeeze(0)          # normalized action [12]
        action = dnrm("action", a_norm).numpy()          # de-normalized -> env action space

        heights.append(float(d.qpos[2]))                 # base height
        fwd_vels.append(float(base[6]))                  # body-frame forward velocity
        state = step(state, jp.asarray(action))
        states.append(state)

    heights, fwd = np.array(heights), np.array(fwd_vels)
    print(f"steps: {STEPS}")
    print(f"upright steps (h>0.2): {int((heights > 0.2).sum())}/{STEPS}")
    print(f"mean base height: {heights.mean():.3f}  (min {heights.min():.3f})")
    print(f"mean forward vel: {fwd.mean():.3f}  (command vx={COMMAND[0]})")

    # save the pose trajectory so you can render on your Mac if the pod's GL won't cooperate
    np.savez("rollout_qpos.npz", qpos=np.array(qpos_traj))
    print("saved rollout_qpos.npz")

    # best-effort video on the pod (needs a working GL backend; metrics above hold regardless)
    try:
        frames = env.render(states)
        import imageio
        imageio.mimsave("rollout.mp4", frames, fps=50)
        print("saved rollout.mp4")
    except Exception as e:
        print("render skipped (use rollout_qpos.npz on your Mac instead):", repr(e))


if __name__ == "__main__":
    main()
