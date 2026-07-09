"""Phase 1 — train the RL TEACHER (Go1 locomotion) in MuJoCo Playground.

Run this in the `go2-rl` env (jax + mujoco + brax + playground), NOT the smotf env.

MuJoCo Playground ships the Unitree Go1 quadruped (not Go2), which is the same
12-joint quadruped and maps onto smotf's dims exactly (base 12 / legs 24 /
contacts 4 / command 3 / action 12), so it is a drop-in teacher.

This is reinforcement learning: the policy learns to walk FROM SCRATCH by
interacting with the MuJoCo physics and a locomotion reward — there is no
dataset. Brax PPO runs the Go1 in the sim, rewards forward locomotion, and
improves the policy through interaction.

Output: go1_policy.pkl — the trained teacher (params + network config), used in
Phase 2 to record the real demonstration data that s-motf is cloned on.
"""

import functools
import pickle
import sys

import jax
from mujoco_playground import registry, wrapper
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

# usage: python train_go1.py [ENV_NAME] [OUTPUT.pkl] [NUM_TIMESTEPS]
#   e.g. python train_go1.py Go1Footstand footstand_policy.pkl
#        python train_go1.py Go1Getup     getup_policy.pkl     300000000   # train 300M steps
ENV = sys.argv[1] if len(sys.argv) > 1 else "Go1JoystickFlatTerrain"
OUT = sys.argv[2] if len(sys.argv) > 2 else "go1_policy.pkl"
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else None   # optional: override num_timesteps


def main():
    print("JAX devices:", jax.devices())          # expect [CudaDevice(id=0)]
    print("Training env:", ENV, "-> saving", OUT)

    env = registry.load(ENV)
    ppo_params = dict(locomotion_params.brax_ppo_config(ENV))
    print("default num_timesteps:", ppo_params.get("num_timesteps"))
    if STEPS is not None:
        ppo_params["num_timesteps"] = STEPS
        print("overriding num_timesteps ->", STEPS)

    # Playground nests the network sizes under 'network_factory'; brax's ppo.train
    # wants an actual factory callable, so pop it out and build the partial.
    net_cfg = dict(ppo_params.pop("network_factory"))
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        print(f"step {step:>10}  reward {r:8.2f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,   # Playground's MjxEnv wrapper (not brax's default)
        network_factory=network_factory,
        progress_fn=progress,
        **ppo_params,
    )

    with open(OUT, "wb") as f:
        pickle.dump({"env": ENV, "params": params, "network_factory": net_cfg}, f)
    print(f"saved {OUT}  (teacher for {ENV})")


if __name__ == "__main__":
    main()
