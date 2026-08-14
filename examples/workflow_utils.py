import stable_baselines3 as sb3
import os
import numpy as np
import matplotlib.pyplot as plt

# Some functions that are used in several workflows

def train_with_sb3_dqn(env, eval_env, outpath, n_steps):
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
        env,
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
    )

    model.learn(total_timesteps=n_steps,
                callback=eval_callback,
                progress_bar=True)
    
    model.save(model_path)
    monitor_env.close()

def train_with_sb3_sac(env, outpath):
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
        verbose=0
    )

    model.learn(total_timesteps=1_000_000,
                callback=eval_callback,
                progress_bar=True)
    
    model.save(model_path)
    monitor_env.close()


def plot_logged_data(log_paths, labels, key):
    logs = []
    for lp in log_paths:
        log = np.load(lp)
        logs.append(log[key])

    plt.title(f"Training {key} using DQN")
    for log, label in zip(logs, labels):
        plt.plot([i*10000 for i in range(len(log))], [np.mean(log_dp) for log_dp in log], label=label)
        plt.fill_between([i*10000 for i in range(len(log))], [np.mean(log_dp)+np.std(log_dp) for log_dp in log], [np.mean(log_dp) - np.std(log_dp) for log_dp in log], alpha=0.3)
    plt.ylabel(key)
    plt.xlabel("timesteps")
    plt.legend()
    plt.show()

def run_eval_episodes(env, policy):
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

def plot_compare_policy_eval(rewards, labels):
    plt.title(f"Comparing policies in evaluation: {", ".join(labels)}")
    for i, (rew, label) in enumerate(zip(rewards, labels)):
        plt.scatter(i, np.mean(rew), label=label)
        plt.plot([i, i], [np.mean(rew) + np.std(rew), np.mean(rew) - np.std(rew)])
    plt.legend()
    plt.xticks([i for i in range(len(rewards))])
    plt.xlim(-0.5, len(rewards)-0.5)
    plt.show()