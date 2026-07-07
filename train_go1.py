"""Phase 1 — train the RL TEACHER (Go1 locomotion) in MuJoCo Playground.

Run this in the `go2-rl` env (jax + mujoco + brax + playground), NOT the smotf env.

MuJoCo Playground ships the Unitree Go1 quadruped (not Go2), which is the same
12-joint quadruped and maps onto smotf's dims exactly (base 12 / legs 24 /
contacts 4 / command 3 / action 12), so it is a drop-in teacher.

This is reinforcement learning: the policy learns to walk FROM SCRATCH by
interacting with the MuJoCo physics and a locomotion reward — there is no
dataset. Brax PPO runs the Go1 in the sim, rewards forward locomotion, and
improves the policy through interaction.

Output: go1_policy.pkl — the trained teacher, used in Phase 2 to record the
real demonstration data that s-motf (the student) will be behavior-cloned on.

NOTE: MuJoCo Playground's API is version-sensitive. If an import or call below
does not match your installed version, run:
    python -c "import mujoco_playground as mp; print(mp.registry.ALL_ENVS)"
and check Playground's example training script, then adjust the marked lines.
"""

import pickle

import jax
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params          # <-- version-sensitive
from brax.training.agents.ppo import train as ppo               # <-- version-sensitive

ENV = "Go1JoystickFlatTerrain"      # flat-ground velocity-commanded Go1 walking


def main():
    print("JAX devices:", jax.devices())          # expect [CudaDevice(id=0)] on the A800
    print("Training env:", ENV)

    env = registry.load(ENV)
    ppo_cfg = locomotion_params.brax_ppo_config(ENV)   # PPO hyperparams for this env

    def progress(step, metrics):
        r = metrics.get("eval/episode_reward", float("nan"))
        print(f"step {step:>10}  reward {r:8.2f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        **ppo_cfg,
    )

    with open("go1_policy.pkl", "wb") as f:
        pickle.dump({"env": ENV, "params": params}, f)
    print("saved go1_policy.pkl  (the teacher)")


if __name__ == "__main__":
    main()
