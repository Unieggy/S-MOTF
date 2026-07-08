"""Evaluate s-motf, its ablations, an MLP baseline, and the RL teacher in the
Go1 env — the numbers for the README.

Rolls out each policy for K episodes in the SAME env and reports, as mean ± std:
  - episode return (env reward; the teacher was trained to maximize this)
  - survival (upright steps out of T)
  - command-tracking error (|forward vel - commanded vx|)

Rows whose checkpoint is missing are skipped, so you can run it after training
whichever variants you have.

Run in go2-rl (needs the env + cpu torch):  python eval_go1.py
"""

import os
import pickle
import numpy as np
import jax
import jax.numpy as jp
import torch

from mujoco_playground import registry
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics

from smotf import load_config
from smotf.model.smotf import SMoTF

ENV_NAME = "Go1JoystickFlatTerrain"
K = 20        # episodes per policy
T = 300       # steps per episode


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
    return np.asarray(jp.concatenate([
        quat_to_rpy(q), d.qvel[3:6], R.T @ d.qvel[0:3], R.T @ jp.array([0., 0., -1.])
    ]))

def legs_from_data(d):
    return np.asarray(jp.concatenate([d.qpos[7:19], d.qvel[6:18]]))


# ---- context builder shared by s-motf and the MLP (normalized, batch of 1) ----
def make_norm(stats):
    nrm = lambda k, x: (torch.tensor(np.asarray(x), dtype=torch.float32) - stats[k]["mean"]) / stats[k]["std"]
    dnrm = lambda k, x: x * stats[k]["std"] + stats[k]["mean"]
    def context(state):
        d = state.data
        return {
            "base":     nrm("base", base_from_data(d))[None],
            "legs":     nrm("legs", legs_from_data(d))[None],
            "contacts": nrm("contacts", np.asarray(state.info["last_contact"], np.float32))[None],
            "command":  nrm("command", np.asarray(state.info["command"], np.float32))[None],
        }
    return context, dnrm


def make_teacher(env):
    ck = pickle.load(open("go1_policy.pkl", "rb"))
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **ck["network_factory"])
    pol = ppo_networks.make_inference_fn(net)(ck["params"], deterministic=True)
    return lambda state, key: pol(state.obs, key)[0]


def make_smotf(ckpt, use_plan):
    cfg = load_config(); cfg.use_plan = use_plan
    m = SMoTF(cfg); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    context, dnrm = make_norm(torch.load("norm_stats.pt"))
    def act(state, key):
        with torch.no_grad():
            a = m.act(context(state)).squeeze(0)
        return jp.asarray(dnrm("action", a).numpy())
    return act


def make_mlp(ckpt):
    from mlp_baseline import MLPPolicy
    cfg = load_config()
    m = MLPPolicy(cfg); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    context, dnrm = make_norm(torch.load("norm_stats.pt"))
    def act(state, key):
        with torch.no_grad():
            a = m(context(state)).squeeze(0)
        return jp.asarray(dnrm("action", a).numpy())
    return act


def evaluate(action_fn, env, reset, step):
    rets, survs, trk = [], [], []
    for k in range(K):
        state = reset(jax.random.PRNGKey(1000 + k))
        key = jax.random.PRNGKey(k)
        ret, alive, errs = 0.0, 0, []
        for t in range(T):
            key, ak = jax.random.split(key)
            base = base_from_data(state.data)
            errs.append(abs(float(base[6]) - float(state.info["command"][0])))
            state = step(state, action_fn(state, ak))
            ret += float(state.reward)
            alive += int(float(state.data.qpos[2]) > 0.2)
            if bool(state.done):
                break
        rets.append(ret); survs.append(alive); trk.append(float(np.mean(errs)))
    return np.array(rets), np.array(survs), np.array(trk)


def main():
    env = registry.load(ENV_NAME)
    reset, step = jax.jit(env.reset), jax.jit(env.step)

    rows = [
        ("RL teacher",        lambda: make_teacher(env),                        "go1_policy.pkl"),
        ("s-motf (full)",     lambda: make_smotf("checkpoint_go1.pt", True),    "checkpoint_go1.pt"),
        ("  - no world model",lambda: make_smotf("checkpoint_nodyn.pt", True),  "checkpoint_nodyn.pt"),
        ("  - no latent plan",lambda: make_smotf("checkpoint_noplan.pt", False),"checkpoint_noplan.pt"),
        ("MLP baseline",      lambda: make_mlp("checkpoint_mlp.pt"),            "checkpoint_mlp.pt"),
    ]

    print(f"\n{'policy':20} {'return':>15} {'survival/'+str(T):>13} {'track err':>12}")
    print("-" * 62)
    for name, build, ckpt in rows:
        if not os.path.exists(ckpt):
            print(f"{name:20} {'(no ' + ckpt + ')':>40}")
            continue
        r, s, t = evaluate(build(), env, reset, step)
        print(f"{name:20} {r.mean():7.1f} ± {r.std():4.1f} {s.mean():11.0f}   {t.mean():10.3f}")


if __name__ == "__main__":
    main()
