"""Phase 2 (multi-skill) — roll out a skill's teacher and record skill-tagged data.

Repurposes the 3-dim command as a SKILL-ID one-hot so one s-motf can be told
which skill to perform:
    walk=[1,0,0]  footstand=[0,1,0]  getup=[0,0,1]
Saves a per-skill .npz; combine_skills.py merges them into go1_multiskill.npz.

Run in go2-rl:
  python collect_multiskill.py Go1JoystickFlatTerrain go1_policy.pkl      0 go1_walk.npz
  python collect_multiskill.py Go1Footstand           footstand_policy.pkl 1 go1_footstand.npz
  python collect_multiskill.py Go1Getup               getup_policy.pkl     2 go1_getup.npz
"""

import sys
import pickle
import numpy as np
import jax
import jax.numpy as jp

from mujoco_playground import registry
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

ENV_NAME = sys.argv[1]
POLICY   = sys.argv[2]
SKILL_ID = int(sys.argv[3])
OUT      = sys.argv[4]

NUM_SKILLS = 3        # = command dim (one-hot skill selector)
N_ENVS = 256
T = 300


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
    return jp.concatenate([quat_to_rpy(q), d.qvel[3:6], R.T @ d.qvel[0:3], R.T @ jp.array([0., 0., -1.])])

def legs_from_data(d):
    return jp.concatenate([d.qpos[7:19], d.qvel[6:18]])


def load_policy():
    ck = pickle.load(open(POLICY, "rb"))
    env = registry.load(ck["env"])
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **ck["network_factory"])
    policy = ppo_networks.make_inference_fn(net)(ck["params"], deterministic=True)
    return env, policy


def main():
    print(f"collecting {ENV_NAME} (skill {SKILL_ID}) from {POLICY} -> {OUT}")
    env, policy = load_policy()
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    key = jax.random.PRNGKey(0)
    key, rk = jax.random.split(key)
    state = reset(jax.random.split(rk, N_ENVS))

    cmd = np.zeros((N_ENVS, NUM_SKILLS), np.float32)   # skill-id one-hot
    cmd[:, SKILL_ID] = 1.0

    fields = {k: [] for k in ["base", "legs", "contacts", "command", "action", "s_next"]}
    for t in range(T):
        key, ak = jax.random.split(key)
        act, _ = policy(state.obs, ak)
        base = jax.vmap(base_from_data)(state.data)
        legs = jax.vmap(legs_from_data)(state.data)
        if "last_contact" in state.info:                # robust: some skills may not expose it
            contacts = np.asarray(state.info["last_contact"], np.float32)
        else:
            contacts = np.zeros((N_ENVS, 4), np.float32)
        nstate = step(state, act)
        s_next = jax.vmap(base_from_data)(nstate.data)
        for k, v in [("base", base), ("legs", legs), ("contacts", contacts),
                     ("command", cmd), ("action", np.asarray(act)), ("s_next", s_next)]:
            fields[k].append(np.asarray(v))
        state = nstate

    stacked = {k: np.stack(v, 0).transpose(1, 0, 2) for k, v in fields.items()}
    np.savez(OUT, **stacked)
    print("saved", OUT, {k: v.shape for k, v in stacked.items()})


if __name__ == "__main__":
    main()
