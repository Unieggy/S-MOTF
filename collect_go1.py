"""Phase 2 — roll out the Go1 teacher, record BC data in smotf's field format.

Runs go1_policy.pkl in Go1JoystickFlatTerrain across many parallel envs and
records per timestep: base[12], legs[24], contacts[4], command[3], action[12],
s_next[12]. Saves go1_data.npz (numpy, no torch) -> loaded into smotf in Phase 3.

Run in the go2-rl env.
"""

import pickle
import jax
import jax.numpy as jp
import numpy as np

from mujoco_playground import registry
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

N_ENVS = 256      # parallel episodes
T = 300           # steps per episode (~6 s at 50 Hz)
SEED = 0


# ---- base-frame extraction from a single env's mjx data ----
def quat_to_rot(q):                         # q = [w,x,y,z], body->world
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

def base_from_data(data):                   # -> [12]
    q = data.qpos[3:7]
    R = quat_to_rot(q)
    rpy     = quat_to_rpy(q)                 # 3
    ang_vel = data.qvel[3:6]                 # 3 (free-joint angular vel is body-frame)
    lin_vel = R.T @ data.qvel[0:3]           # 3 (world lin-vel -> body)
    proj_g  = R.T @ jp.array([0., 0., -1.])  # 3 (gravity in body frame)
    return jp.concatenate([rpy, ang_vel, lin_vel, proj_g])

def legs_from_data(data):                    # -> [24]
    return jp.concatenate([data.qpos[7:19], data.qvel[6:18]])


def load_policy():
    ckpt = pickle.load(open("go1_policy.pkl", "rb"))
    env = registry.load(ckpt["env"])
    ppo_net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **ckpt["network_factory"],
    )
    policy = ppo_networks.make_inference_fn(ppo_net)(ckpt["params"], deterministic=True)
    return env, policy


def main():
    env, policy = load_policy()
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    key = jax.random.PRNGKey(SEED)
    key, rk = jax.random.split(key)
    state = reset(jax.random.split(rk, N_ENVS))

    fields = {k: [] for k in ["base", "legs", "contacts", "command", "action", "s_next"]}

    for t in range(T):
        key, ak = jax.random.split(key)
        act, _ = policy(state.obs, ak)                      # [N,12] (deterministic)

        base = jax.vmap(base_from_data)(state.data)          # [N,12]
        legs = jax.vmap(legs_from_data)(state.data)          # [N,24]
        contacts = state.info["last_contact"].astype(jp.float32)  # [N,4]
        command = state.info["command"]                      # [N,3]

        nstate = step(state, act)
        s_next = jax.vmap(base_from_data)(nstate.data)        # [N,12]

        for k, v in [("base", base), ("legs", legs), ("contacts", contacts),
                     ("command", command), ("action", act), ("s_next", s_next)]:
            fields[k].append(np.asarray(v))
        state = nstate

    # [T,N,dim] -> [N,T,dim] (N episodes of length T)
    stacked = {k: np.stack(v, 0).transpose(1, 0, 2) for k, v in fields.items()}
    np.savez("go1_data.npz", **stacked)
    print("saved go1_data.npz")
    for k, v in stacked.items():
        print(f"  {k:10} {v.shape}")


if __name__ == "__main__":
    main()
