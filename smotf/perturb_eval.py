"""Perturbation eval: does planning recover better than BC under pushes?
Runs BC and the N=16,H=3 planner on walk with STRONG perturbations, K=20 episodes,
reports mean±std survival/return. Two SEPARATE runs — the controllers diverge."""
import numpy as np, jax
from mujoco_playground import registry
import eval_multiskill as em

ENV, SKILL, K, T = "Go1JoystickFlatTerrain", 0, 20, 300

cfg = registry.get_default_config(ENV)
# >>> FILL FROM STEP 1: enable + strengthen pushes, e.g.
cfg.pert_config.enable = True
cfg.pert_config.velocity_kick = [3.0, 5.0]     # strong, consistent pushes (default max was 3.0)
cfg.pert_config.kick_wait_times = [1.0, 2.0]   # push every 1–2 s (more frequent than default)
env = registry.load(ENV, config=cfg)
reset, step = jax.jit(env.reset), jax.jit(env.step)

def eval_full(fn):
    rets, survs = [], []
    for k in range(K):
        state = reset(jax.random.PRNGKey(1000 + k))
        key = jax.random.PRNGKey(k)
        ret, alive = 0.0, 0
        for t in range(T):
            key, ak = jax.random.split(key)
            state = step(state, fn(state, ak))
            ret += float(state.reward)
            alive += int(float(state.data.qpos[2]) > 0.2)
            if bool(state.done): break
        rets.append(ret); survs.append(alive)
    return np.array(rets), np.array(survs)

for name, fn in [
    ("BC",          em.make_smotf("checkpoint_multi.pt", True, SKILL)),
    ("plan N16 H3", em.make_planner("checkpoint_multi.pt", SKILL, N=16, H=3)),
]:
    r, s = eval_full(fn)
    print(f"{name:12} ret {r.mean():6.2f} ± {r.std():4.2f}   surv {s.mean():5.1f} ± {s.std():4.1f}", flush=True)
