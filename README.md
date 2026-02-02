# RL Training - PPO with Hydra

This repository contains a clean implementation of the Proximal Policy Optimization (PPO) algorithm using [Gymnasium](https://gymnasium.farama.org/), [PyTorch](https://pytorch.org/), and [Hydra](https://hydra.cc/) for configuration management.

## 🚀 Getting Started

### Prerequisites

This project uses `uv` for dependency management. If you don't have it installed, you can find instructions [here](https://github.com/astral-sh/uv).

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rl-training
   ```

2. Install dependencies and create a virtual environment:
   ```bash
   uv sync
   ```

## 🛠 Project Structure

```text
rl-training/
├── configs/                # Hydra configuration files
│   ├── agent/              # Agent-specific configs (PPO)
│   ├── env/                # Environment-specific configs (CartPole)
│   └── config.yaml         # Main configuration entry point
├── scripts/
│   └── ppo.py              # Main training script
├── src/rl_training/
│   ├── agents/             # PPO implementation
│   └── models/             # Neural network architectures (Actor-Critic)
├── pyproject.toml          # Project metadata and dependencies
└── README.md
```

## 📈 Usage

### Running the Training

To start training with the default parameters:
```bash
python scripts/ppo.py
```

### Overriding Parameters

Hydra allows you to easily override any configuration parameter from the command line:

- **Change Learning Rate:**
  ```bash
  python scripts/ppo.py agent.learning_rate=0.001
  ```

- **Change Number of Episodes:**
  ```bash
  python scripts/ppo.py train.max_episodes=2000
  ```

- **Modify Multiple Parameters:**
  ```bash
  python scripts/ppo.py agent.gamma=0.98 agent.k_epochs=10
  ```

### Configuration Hierarchy

The project uses a hierarchical YAML configuration:
- `configs/agent/ppo.yaml`: Contains PPO hyperparameters like `learning_rate`, `gamma`, `eps_clip`, etc.
- `configs/env/cartpole.yaml`: Defines the environment name.
- `configs/config.yaml`: Orchestrates the defaults and defines training loop parameters.

## 🤖 Algorithm: PPO

Proximal Policy Optimization (PPO) is a popular Reinforcement Learning algorithm that aims to strike a balance between ease of implementation, sample efficiency, and ease of tuning. This implementation includes:
- Actor-Critic architecture.
- Clipped objective function.
- Support for Gymnasium environments.

## 📄 License

MIT License
