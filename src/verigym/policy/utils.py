import numpy as np

import verigym

def evaluate_policy(model: verigym.VeriGymEnv, policy: verigym.PolicyClass, n_steps: int) -> tuple[float, float]:
    """Takes a policy and an environment and does the rollouts on that environment and checks its performance wrt average return."""
    trajectories = model.simulate(
        policy=policy,
        n_steps=n_steps,
        verbose=False
    )
    mean, std = mean_reward_trajectories(trajectories)
    return mean, std


def mean_reward_trajectories(trajectories: list) -> tuple[float, float]:
    returns = [sum(reward for _, _, reward, *_ in trajectory) for trajectory in trajectories]
    return np.mean(returns), np.std(returns)
