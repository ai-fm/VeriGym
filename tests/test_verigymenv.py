import gymnasium as gym

from verigym.environments.verigymenv import verigymenv

def test_simulate():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = verigymenv.(env)