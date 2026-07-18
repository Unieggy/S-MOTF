"""Phase E — per-skill evaluation of the multi-skill generalists vs specialists.

For each skill, runs the policy in that skill's env and reports return + survival.
s-motf / MLP are told which skill to do via the command one-hot; the RL specialist
is that skill's own teacher (the per-skill ceiling).

Run in go2-rl:  python eval_multiskill.py
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

# (env, skill_id, specialist_policy)
SKILLS = [
    ("Go1JoystickFlatTerrain", 0, "go1_policy.pkl"),
    ("Go1Footstand",           1, "footstand_policy.pkl"),
    ("Go1Getup",               2, "getup_policy.pkl"),
]
NUM_SKILLS = 3
K, T = 10, 500


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

def contacts_from(state):
    if "last_contact" in state.info:
        return np.asarray(state.info["last_contact"], np.float32)
    return np.zeros(4, np.float32)

def skill_onehot(skill_id):
    c = np.zeros(NUM_SKILLS, np.float32); c[skill_id] = 1.0
    return c


def command_vec(state, skill_id):
    # [velocity(3), skill-id(3)]: env's velocity for walk, zeros for balance skills
    if "command" in state.info:
        vel = np.asarray(state.info["command"], np.float32)
        if vel.shape[-1] != 3:
            vel = np.zeros(3, np.float32)
    else:
        vel = np.zeros(3, np.float32)
    return np.concatenate([vel, skill_onehot(skill_id)])   # [6]


def make_context(stats, skill_id):
    nrm = lambda k, x: (torch.tensor(np.asarray(x), dtype=torch.float32) - stats[k]["mean"]) / stats[k]["std"]
    dnrm = lambda k, x: x * stats[k]["std"] + stats[k]["mean"]
    def context(state):
        d = state.data
        return {
            "base":     nrm("base", base_from_data(d))[None],
            "legs":     nrm("legs", legs_from_data(d))[None],
            "contacts": nrm("contacts", contacts_from(state))[None],
            "command":  nrm("command", command_vec(state, skill_id))[None],   # velocity + skill
        }
    return context, dnrm


def make_smotf(ckpt, use_plan, skill_id):
    cfg = load_config("configs/multiskill.yaml"); cfg.use_plan = use_plan
    m = SMoTF(cfg); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    context, dnrm = make_context(torch.load("norm_stats.pt"), skill_id)
    def act(state, key):
        with torch.no_grad():
            a = m.act(context(state)).squeeze(0)
        return jp.asarray(dnrm("action", a).numpy())
    return act


def make_mlp(ckpt, skill_id):
    from mlp_baseline import MLPPolicy
    cfg = load_config("configs/multiskill.yaml")
    m = MLPPolicy(cfg); m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    context, dnrm = make_context(torch.load("norm_stats.pt"), skill_id)
    def act(state, key):
        with torch.no_grad():
            a = m(context(state)).squeeze(0)
        return jp.asarray(dnrm("action", a).numpy())
    return act


def make_specialist(policy_file, env):
    ck = pickle.load(open(policy_file, "rb"))
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize, **ck["network_factory"])
    pol = ppo_networks.make_inference_fn(net)(ck["params"], deterministic=True)
    return lambda state, key: pol(state.obs, key)[0]


def evaluate(action_fn, env, reset, step):
    rets, survs = [], []
    for k in range(K):
        state = reset(jax.random.PRNGKey(1000 + k))
        key = jax.random.PRNGKey(k)
        ret, alive = 0.0, 0
        for t in range(T):
            key, ak = jax.random.split(key)
            state = step(state, action_fn(state, ak))
            ret += float(state.reward)
            alive += int(float(state.data.qpos[2]) > 0.2)
            if bool(state.done):
                break
        rets.append(ret); survs.append(alive)
    return np.mean(rets), np.mean(survs)

def make_planner(ckpt, skill_id, N=16, H=3):
    """Action provider that PLANS: policy proposes N actions, world model imagines
    them H steps, reward head scores, best is executed. Needs 3 trained files:
    the policy (ckpt), world.pt, reward.pt."""
    from smotf.model.world import WorldModel, RewardHead
    from smotf.model.planner import Planner

    cfg = load_config("configs/multiskill.yaml")               # command=6

    # load the 3 trained modules onto CPU
    smotf = SMoTF(cfg);  smotf.load_state_dict(torch.load(ckpt,        map_location="cpu")); smotf.eval()
    world = WorldModel(cfg); world.load_state_dict(torch.load("world.pt",  map_location="cpu")); world.eval()
    rew   = RewardHead(cfg); rew.load_state_dict(torch.load("reward.pt", map_location="cpu")); rew.eval()

    planner = Planner(smotf, world, rew, cfg, N=N, H=H)         # the MPC wrapper

    # same normalized-context builder the other providers use (velocity + skill-id command)
    context, dnrm = make_context(torch.load("norm_stats.pt"), skill_id)

    def act(state, key):                                       # called each env step
        with torch.no_grad():
            a = planner.plan(context(state))                  # [12] normalized best action
        return jp.asarray(dnrm("action", a).numpy())          # denormalize -> jax action for env.step
    return act


def main():
    # load + jit each env ONCE (not per policy) -> 3 compiles instead of 12
    print("compiling envs (one-time)...", flush=True)
    ENVS = {}
    for env_name, _, _ in SKILLS:
        e = registry.load(env_name)
        ENVS[env_name] = (e, jax.jit(e.reset), jax.jit(e.step))

    # rows = policy family; measured per skill
    print(f"\n{'policy':18} | " + " | ".join(f"{s[0][3:]:>22}" for s in SKILLS))
    print("-" * 90)

    for name, builder in [
        ("RL specialist", "spec"),
        ("s-motf multi", "smotf"),
        ("  - no plan", "smotf_noplan"),
        ("s-motf + plan", "planner"),  
        ("MLP multi", "mlp"),
    ]:
        cells = []
        for env_name, skill_id, spec in SKILLS:
            env, reset, step = ENVS[env_name]
            if builder == "spec":
                ok = os.path.exists(spec); fn = (lambda: make_specialist(spec, env)) if ok else None
            elif builder == "smotf":
                ok = os.path.exists("checkpoint_multi.pt"); fn = (lambda sid=skill_id: make_smotf("checkpoint_multi.pt", True, sid)) if ok else None
            elif builder == "smotf_noplan":
                ok = os.path.exists("checkpoint_multi_noplan.pt"); fn = (lambda sid=skill_id: make_smotf("checkpoint_multi_noplan.pt", False, sid)) if ok else None
            elif builder == "planner":
                # needs all three: policy + world model + reward head
                ok = all(os.path.exists(f) for f in
                         ["checkpoint_multi.pt", "world.pt", "reward.pt"])
                fn = (lambda sid=skill_id: make_planner("checkpoint_multi.pt", sid)) if ok else None

            else:
                ok = os.path.exists("checkpoint_mlp.pt"); fn = (lambda sid=skill_id: make_mlp("checkpoint_mlp.pt", sid)) if ok else None
            if fn is None:
                cells.append("      (no ckpt)     ")
            else:
                print(f"  running {name.strip()} on {env_name} ...", flush=True)
                r, s = evaluate(fn(), env, reset, step)
                cells.append(f"ret {r:6.1f}  surv {s:4.0f}")
        print(f"{name:18} | " + " | ".join(cells), flush=True)


if __name__ == "__main__":
    main()
