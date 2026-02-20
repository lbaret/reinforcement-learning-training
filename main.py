from time import sleep

import click
import gymnasium as gym
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from rl_training.agents.ppo import PPO


@click.command()
@click.option("--agent-horizon", default=50, help="Horizon for the agent.")
@click.option("--total-episodes", default=500, help="Total number of episodes.")
def main(agent_horizon: int, total_episodes: int) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1", render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PPO(state_dim=state_dim, action_dim=action_dim, device=device)

    rewards_per_episode = []
    score = 0.0
    progress_bar = tqdm(
        range(total_episodes),
        total=total_episodes,
        desc="Agent is currently learning ...",
    )
    for episode_number in progress_bar:
        s, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0

        while not done and not truncated:
            for t in range(agent_horizon):
                prob = model.policy_old.act(s)
                a = prob[0]
                a_log_prob = prob[1]
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data(
                    (s, a, r / 100.0, s_prime, a_log_prob, done or truncated)
                )
                s = s_prime
                score += r
                episode_reward += r
                if done or truncated:
                    break

            model.update()
            score = 0.0

        rewards_per_episode.append(episode_reward)
        progress_bar.set_postfix({"episode reward": episode_reward})

    env.close()

    plt.figure()
    plt.plot(rewards_per_episode)
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("rewards_per_episode.png")
    plt.show()

    env_show = gym.make("CartPole-v1", render_mode="human")
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
    main()
