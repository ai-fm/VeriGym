import gymnasium as gym
from stable_baselines3 import PPO

import verigym
from verigym.abstraction.gym_utils.transform_observation import ReplaceInfObservation
from verigym.frameworks.stable_baselines3.policy import SB3Policy

global model

model = None


def get_basics_helper(total_timesteps=int(10e2)):
    global model
    env_name = "CartPole-v1"
    gym_env = gym.make(env_name)

    if model is None:
        model = PPO("MlpPolicy", gym_env, verbose=1)
        model.learn(total_timesteps=total_timesteps, progress_bar=False)

    return gym_env, model


def test_instantiation():

    _gym_env, model = get_basics_helper()
    _verigym_policy = SB3Policy(model)


def test_verigym_pol_on_gym_env():
    gym_env, model = get_basics_helper()
    verigym_policy = SB3Policy(model)
    # Test the verigym_policy on gym_env
    obs, info = gym_env.reset()
    for _ in range(10):
        action = verigym_policy.get_action(obs)
        assert action in gym_env.action_space, (
            f"{action = } not in {gym_env.action_space = }!"
        )
        obs, reward, terminated, truncated, info = gym_env.step(action)


def test_verigym_pol_on_verigym_env():
    gym_env, model = get_basics_helper()
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


def test_verigym_simulation_on_verigym_env():
    gym_env, model = get_basics_helper()
    verigym_policy = SB3Policy(model)
    # simulation with verigym_policy on verigym_env
    gym_env_finite = ReplaceInfObservation(gym_env, neg_inf=-10, pos_inf=10)
    verigym_env = verigym.GenerativeEnv.from_gymnasium(gym_env_finite)
    verigym_env.simulate(policy=verigym_policy, n_steps=int(10e3))
