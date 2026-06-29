import pytest
import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C

import verigym
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
from verigym.frameworks.stable_baselines3.policy import SB3Policy

global model_dict

model_dict = {}

ENVIRONMENT_NAMES = [
    "CartPole-v1",
    "MountainCarContinuous-v0",
    "Acrobot-v1",
]  # TODO: add envs with continous action space
POLICIES = (PPO, DQN, A2C)


def get_basics_helper(
    total_timesteps=int(10e2), env_name="CartPole-v1", sb3_policy=PPO
):
    """Helper function. creates environment and trains a policy. Returns both."""
    global model_dict
    env_name = "CartPole-v1"
    gym_env = gym.make(env_name)

    if model_dict.get(sb3_policy.__name__) is None:
        model = sb3_policy("MlpPolicy", gym_env, verbose=1)
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        model_dict[sb3_policy.__name__] = model

    return gym_env, model_dict[sb3_policy.__name__]


@pytest.mark.parametrize("env_name", ENVIRONMENT_NAMES)
def test_instantiation(env_name):
    _gym_env, model = get_basics_helper(env_name=env_name)
    _verigym_policy = SB3Policy(model)


@pytest.mark.parametrize("env_name", ENVIRONMENT_NAMES)
@pytest.mark.parametrize("sb3_policy", POLICIES)
def test_verigym_pol_on_gym_env(env_name, sb3_policy):
    gym_env, model = get_basics_helper(env_name=env_name, sb3_policy=sb3_policy)
    verigym_policy = SB3Policy(model)
    # Test the verigym_policy on gym_env
    obs, info = gym_env.reset()
    for _ in range(10):
        action = verigym_policy.get_action(obs)
        assert action in gym_env.action_space, (
            f"{action = } not in {gym_env.action_space = }!"
        )
        obs, reward, terminated, truncated, info = gym_env.step(action)
        if terminated or truncated:
            obs, info = gym_env.reset()


@pytest.mark.parametrize("env_name", ENVIRONMENT_NAMES)
@pytest.mark.parametrize("sb3_policy", POLICIES)
def test_verigym_pol_on_verigym_env(env_name, sb3_policy):
    gym_env, model = get_basics_helper(env_name=env_name, sb3_policy=sb3_policy)
    verigym_policy = SB3Policy(model)
    # test verigym_policy on VeriGymEnv
    gym_env_finite = ReplaceInfObservation(gym_env, neg_inf=-10, pos_inf=10)
    verigym_env = verigym.GenerativeEnv.from_gymnasium(gym_env_finite)
    obs, info = verigym_env.reset()
    for _ in range(10):
        action = verigym_policy.get_action(obs)
        assert action in verigym_env.action_space, (
            f"{action = } not in {verigym_env.action_space = }!"
        )
        obs, reward, terminated, truncated, info = gym_env.step(action)
        if terminated or truncated:
            obs, info = gym_env.reset()


@pytest.mark.parametrize("env_name", ENVIRONMENT_NAMES)
@pytest.mark.parametrize("sb3_policy", POLICIES)
def test_verigym_simulation_on_verigym_env(env_name, sb3_policy):
    gym_env, model = get_basics_helper(env_name=env_name, sb3_policy=sb3_policy)
    verigym_policy = SB3Policy(model)
    # simulation with verigym_policy on verigym_env
    gym_env_finite = ReplaceInfObservation(gym_env, neg_inf=-10, pos_inf=10)
    verigym_env = verigym.GenerativeEnv.from_gymnasium(gym_env_finite)
    verigym_env.simulate(policy=verigym_policy, n_steps=int(10e1))
