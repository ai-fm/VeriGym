from verigym.environments.explicitenv import BaseExplicitEnv

def preprocess_explicit_env(env: BaseExplicitEnv) -> BaseExplicitEnv:
    assert issubclass(type(env), BaseExplicitEnv) \
        or issubclass(type(env.unwrapped), BaseExplicitEnv)
    if not issubclass(type(env), BaseExplicitEnv):
        return env.unwrapped
    return env