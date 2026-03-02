from time import sleep

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium.wrappers import NormalizeObservation
from gymnasium.wrappers.vector import NormalizeObservation as VecNormalizeObservation
from tqdm import tqdm

from rl_training.agents.ppo import PPO


@click.group()
def cli():
    pass


@cli.command()
@click.option("--agent-horizon", default=50, help="Horizon for the agent.")
@click.option("--total-episodes", default=500, help="Total number of episodes.")
@click.option("--use-gpu", default=True, help="Use GPU for training.")
@click.option("--num-agents", default=4, help="Number of parallel agents.")
def run_cartpole_ppo(
    agent_horizon: int, total_episodes: int, use_gpu: bool, num_agents: int
) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    envs = gym.make_vec("CartPole-v1", num_envs=num_agents, render_mode=None)
    envs = VecNormalizeObservation(envs)

    state_dim = (
        envs.single_observation_space.shape[0]
        if hasattr(envs, "single_observation_space")
        else envs.observation_space.shape[1]
    )
    action_dim = (
        envs.single_action_space.n
        if hasattr(envs, "single_action_space")
        else envs.action_space[0].n
    )

    model = PPO(state_dim=state_dim, action_dim=action_dim, device=device)

    rewards_per_episode = []

    progress_bar = tqdm(
        total=total_episodes,
        desc="Agent is currently learning ...",
    )

    completed_episodes = 0
    s, _ = envs.reset()
    episode_rewards = np.zeros(num_agents)

    while completed_episodes < total_episodes:
        for t in range(agent_horizon):
            probs = model.policy_old.act(s)
            a = probs[0]
            a_log_prob = probs[1]

            s_prime, r, term, trunc, info = envs.step(a)
            done = term | trunc

            for i in range(num_agents):
                model.put_data(
                    (s[i], a[i], r[i] / 100.0, s_prime[i], a_log_prob[i], done[i])
                )
                episode_rewards[i] += r[i]

                if done[i]:
                    rewards_per_episode.append(episode_rewards[i])
                    progress_bar.set_postfix({"episode reward": episode_rewards[i]})
                    episode_rewards[i] = 0.0
                    completed_episodes += 1
                    progress_bar.update(1)

            s = s_prime

            if completed_episodes >= total_episodes:
                break

        model.update()

    envs.close()
    progress_bar.close()

    plt.figure()
    plt.plot(rewards_per_episode)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("rewards_per_episode.png")
    plt.show()

    env_show = gym.make("CartPole-v1", render_mode="human")
    env_show = NormalizeObservation(env_show)

    env_show.obs_rms.mean = envs.obs_rms.mean.copy()
    env_show.obs_rms.var = envs.obs_rms.var.copy()
    env_show.update_running_mean = False

    s, _ = env_show.reset()
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        a, _ = model.policy_old.act(s)
        s, r, done, truncated, _ = env_show.step(a)
        total_reward += r
    print(f"Score de la démonstration : {total_reward}")
    env_show.close()


if __name__ == "__main__":
    cli()
