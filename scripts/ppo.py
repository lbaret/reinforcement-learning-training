import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from rl_training.agents import PPO

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # Extract parameters from config
    learning_rate = cfg.agent.learning_rate
    gamma         = cfg.agent.gamma
    eps_clip      = cfg.agent.eps_clip
    K_epochs      = cfg.agent.k_epochs
    T_horizon     = cfg.agent.t_horizon
    
    env_name       = cfg.env.name
    max_episodes   = cfg.train.max_episodes
    print_interval = cfg.train.print_interval

    env = gym.make(env_name, render_mode=None) 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = PPO(
        state_dim, 
        action_dim, 
        lr=learning_rate, 
        gamma=gamma, 
        eps_clip=eps_clip, 
        k_epochs=K_epochs, 
        device=device
    )
    
    score = 0.0
    
    for n_epi in range(max_episodes):
        s, _ = env.reset()
        done = False
        truncated = False
        
        while not done and not truncated:
            for t in range(T_horizon):
                prob = model.policy_old.act(s)
                a = prob[0]
                a_log_prob = prob[1]
                s_prime, r, done, truncated, info = env.step(a)
                model.put_data((s, a, r/100.0, s_prime, a_log_prob, done or truncated))
                s = s_prime
                score += r
                if done or truncated:
                    break
            model.update()

        if n_epi % print_interval == 0 and n_epi != 0:
            print(f"Épisode: {n_epi}, Score moyen: {score/print_interval:.1f}")
            if score/print_interval > 450:
                print(f"Environnement résolu ! Score final : {score/print_interval}")
                break
            score = 0.0

    env.close()
    
    env_show = gym.make(env_name, render_mode=None)
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

if __name__ == '__main__':
    main()