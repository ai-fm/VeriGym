import stable_baselines3 as sb3
import os
import gymnasium as gym
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from verigym.policy.policy import PolicyClass

# Some functions that are used in several workflows

def train_with_sb3_ppo(
    env: gym.Env,
    eval_env: gym.Env,
    outpath: str,
    n_steps: int,
    eval_freq: int,
    seed: int = 42,
) -> None:
    """
    Trains a Stable-Baselines3 PPO agent and saves it to disk.
    The best model found during evaluation is stored in `outpath + "best/"`, the evaluation logs
    in `outpath + "log/"` and the final model in `outpath + "ppo_model"`.

    Parameters
    ----------
    env: gym.Env
        The environment to train on.
    eval_env: gym.Env
        The environment used for periodic evaluation.
    outpath: str
        Output directory, including a trailing slash.
    n_steps: int
        Total number of training timesteps.
    eval_freq: int
        Number of timesteps between evaluations.
    seed: int
        Seed for the PPO model.
    """
    monitor_env = sb3.common.monitor.Monitor(env)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    save_path = outpath + "best/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = outpath + "log/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    model_path = outpath + "ppo_model"

    eval_env = sb3.common.monitor.Monitor(eval_env)
    eval_callback = sb3.common.callbacks.EvalCallback(eval_env, 
                                                        best_model_save_path=save_path,
                                                        log_path=log_path,
                                                        eval_freq=eval_freq,
                                                        deterministic=True,
                                                        render=False)

    # hyperparameters fit through trial and error
    model = sb3.PPO(
        "MlpPolicy",
        monitor_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(net_arch=[64, 64]),
        verbose=0,
        seed=seed
    )

    model.learn(total_timesteps=n_steps,
                callback=eval_callback,
                progress_bar=True)
    
    model.save(model_path)
    monitor_env.close()

def train_with_sb3_dqn(
    env: gym.Env,
    eval_env: gym.Env,
    outpath: str,
    n_steps: int,
    seed: int = 42,
) -> None:
    """
    Trains a Stable-Baselines3 DQN agent and saves it to disk, evaluating 25 times during training.
    The best model found during evaluation is stored in `outpath + "best/"`, the evaluation logs
    in `outpath + "log/"` and the final model in `outpath + "dqn_model"`.

    Parameters
    ----------
    env: gym.Env
        The environment to train on.
    eval_env: gym.Env
        The environment used for periodic evaluation.
    outpath: str
        Output directory, including a trailing slash.
    n_steps: int
        Total number of training timesteps.
    seed: int
        Seed for the DQN model.
    """
    monitor_env = sb3.common.monitor.Monitor(env)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    save_path = outpath + "best/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = outpath + "log/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    model_path = outpath + "dqn_model"

    eval_env = sb3.common.monitor.Monitor(eval_env)
    eval_callback = sb3.common.callbacks.EvalCallback(eval_env, 
                                                        best_model_save_path=save_path,
                                                        log_path=log_path,
                                                        eval_freq=int(n_steps / 25),
                                                        deterministic=True,
                                                        render=False)

    # hyperparameters fit through trial and error for the 
    # workflow_compare_rl_model_checking.ipynb notebook
    model = sb3.DQN(
        "MlpPolicy",
        monitor_env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=500,
        train_freq=1,
        target_update_interval=250,
        exploration_fraction=0.2,
        policy_kwargs=dict(
            net_arch=[64, 64],
        ),
        verbose=0,
        seed=seed
    )

    model.learn(total_timesteps=n_steps,
                callback=eval_callback,
                progress_bar=True)
    
    model.save(model_path)
    monitor_env.close()

def train_with_sb3_sac(env: gym.Env, outpath: str, seed: int = 42) -> None:
    """
    Trains a Stable-Baselines3 SAC agent with default hyperparameters for 1M timesteps
    and saves it to disk, evaluating on the training environment every 10k timesteps.
    The best model found during evaluation is stored in `outpath + "best/"`, the evaluation logs
    in `outpath + "log/"` and the final model in `outpath + "sac_model"`.

    Parameters
    ----------
    env: gym.Env
        The environment to train and evaluate on.
    outpath: str
        Output directory, including a trailing slash.
    seed: int
        Seed for the SAC model.
    """
    monitor_env = sb3.common.monitor.Monitor(env)
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    save_path = outpath + "best/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    log_path = outpath + "log/"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    model_path = outpath + "sac_model"

    eval_callback = sb3.common.callbacks.EvalCallback(monitor_env, 
                                                      best_model_save_path=save_path,
                                                      log_path=log_path,
                                                      eval_freq=10000,
                                                      deterministic=True,
                                                      render=False)

    # using standard sb3 parameters
    model = sb3.SAC(
        policy="MlpPolicy",
        env=monitor_env,
        verbose=0,
        seed=seed
    )

    model.learn(total_timesteps=1_000_000,
                callback=eval_callback,
                progress_bar=True)
    
    model.save(model_path)
    monitor_env.close()

def rolling_mean(x: npt.ArrayLike, window: int) -> npt.NDArray[np.float64]:
    """
    Computes the trailing rolling mean of `x` over the last `window` values, ignoring NaNs.
    Positions whose window contains only NaNs are NaN in the output.

    Parameters
    ----------
    x: npt.ArrayLike
        One-dimensional input data.
    window: int
        Number of values (up to and including the current one) to average over.

    Returns
    -------
    npt.NDArray[np.float64]
        Array of the same length as `x` containing the rolling means.
    """
    x =np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    end = np.arange(1, x.size + 1)
    start = np.maximum(end - window, 0)
    sums, counts = csum[end] - csum[start], ccnt[end] - ccnt[start]
    out = np.full(x.size, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out

def plot_logged_data(
    log_paths: list[str],
    labels: list[str],
    key: str,
    title: str,
    running_mean: int = 1,
    ylabel: str | None = None,
) -> None:
    """
    Plots the evaluation logs (`evaluations.npz`) written by SB3's EvalCallback of several runs.
    For each run, the mean over the evaluation episodes is plotted per evaluation point,
    with a shaded band between the minimum and maximum. All three curves are smoothed
    over the evaluation points with a rolling mean of size `running_mean`.
    The x-axis uses the training timesteps stored in the log.

    Parameters
    ----------
    log_paths: list[str]
        Paths to the `.npz` log files, one per run.
    labels: list[str]
        Legend label for each run.
    key: str
        Key of the logged quantity to plot, e.g. "results" or "ep_lengths".
    title: str
        Title of the plot.
    running_mean: int
        Window size (in evaluation points) of the rolling mean used to smooth the curves.
    ylabel: str | None
        Label of the y-axis. Defaults to `key`.
    """
    plt.title(title)
    for log_path, label in zip(log_paths, labels):
        data = np.load(log_path)
        log = data[key]
        x = data["timesteps"]
        plt.plot(x, rolling_mean(log.mean(axis=1), window=running_mean), label=label)
        plt.fill_between(x,
                         rolling_mean(log.min(axis=1), window=running_mean),
                         rolling_mean(log.max(axis=1), window=running_mean),
                         alpha=0.3)
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel(key)
    plt.xlabel("Train steps")
    plt.legend()
    plt.show()

def run_eval_episodes(env: gym.Env, policy: PolicyClass) -> list[float]:
    """
    Runs 100 evaluation episodes of `policy` in `env`.

    Parameters
    ----------
    env: gym.Env
        The environment to evaluate in.
    policy: PolicyClass
        The policy to evaluate.

    Returns
    -------
    list[float]
        The total (undiscounted) reward of each episode.
    """
    eval_rewards = []
    for _ in range(100):
        episode_reward = 0
        obs, _ = env.reset()

        while True:
            action = policy.get_action(obs)
            obs, rew, term, trunc, _ = env.step(action)
            
            episode_reward += rew
            if term or trunc:
                break

        eval_rewards.append(episode_reward)
    return eval_rewards

def plot_compare_policy_eval(rewards: list[list[float]], labels: list[str]) -> None:
    """
    Draws a box plot (with means) comparing the evaluation rewards of several policies.

    Parameters
    ----------
    rewards: list[list[float]]
        Episode rewards per policy, e.g. as returned by `run_eval_episodes`.
    labels: list[str]
        Name of each policy.
    """
    plt.title(f"Comparing policies in evaluation: {", ".join(labels)}")
    plt.boxplot(
        rewards,
        tick_labels=labels,
        showmeans=True
    )
    plt.ylabel("Reward")
    plt.xlim(0.5, len(rewards)+0.5)
    plt.show()