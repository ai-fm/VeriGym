from verigym.environments.verigymenv import VeriGymEnv
from verigym.policy.policy import PolicyClass
from verigym.environments.explicitmodelenv import ExplicitModelEnv

import time
import numpy as np

class Evaluator:
    """An abstract class for providing the inferface for policy evaluation."""

    def evaluate_rollout(self, policy: PolicyClass , env: VeriGymEnv, 
                         nmbr_rollouts = 1_000,
                         max_steps = 1_000, 
                         timeout = None,
                         seed = 0
                         ):
        
        all_rewards = np.zeros(nmbr_rollouts)
        start_time = time.time()

        for rollout in range(nmbr_rollouts):
            obs, _ = env.reset(seed+rollout)
            policy.reset()
            cumulative_reward = 0

            for step in range(max_steps):
                action = policy.get_action(obs)
                obs, rew, is_terminated, is_truncated, _info = env.step(action)
                cumulative_reward += rew
                if (is_terminated or is_truncated):
                    break
            
            all_rewards[rollout] = cumulative_reward
            if timeout is not None and time.time() - start_time > timeout:
                print("Error: policy evaluation timed out.")
                all_rewards = all_rewards[:rollout]
                break

        return all_rewards

    def evaluate_rollout_vectorized(policy: PolicyClass , env: VeriGymEnv, 
                         nmbr_rollouts = 1_000,
                         max_steps = 1_000, 
                         timeout = None,
                         seed = 0
                         ):
        raise NotImplementedError



