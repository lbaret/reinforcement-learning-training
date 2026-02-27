# RL Training

This repository contains, only for the moment, a clean implementation of the Proximal Policy Optimization (PPO) algorithm using [Gymnasium](https://gymnasium.farama.org/) and [PyTorch](https://pytorch.org/).

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
├── src/rl_training/
│   ├── agents/             # PPO implementation
│   └── models/             # Neural network architectures (Actor-Critic)
├── pyproject.toml          # Project metadata and dependencies
├── main.py                 # Main training script
└── README.md
```

## 📈 Usage

### Running the Training

To start training with the default parameters (500 episodes, horizon of 50):
```bash
python main.py run-cartpole-ppo
```

### Overriding Parameters

You can easily override parameters using CLI options provided by Click:

- **Change Agent Horizon:**
  ```bash
  python main.py run-cartpole-ppo --agent-horizon 100
  ```

- **Change Number of Episodes:**
  ```bash
  python main.py run-cartpole-ppo --total-episodes 2000
  ```

- **Modify Multiple Parameters:**
  ```bash
  python main.py run-cartpole-ppo --agent-horizon 100 --total-episodes 2000
  ```

## 🤖 Algorithm: PPO

Proximal Policy Optimization (PPO) is a popular Reinforcement Learning algorithm that aims to strike a balance between ease of implementation, sample efficiency, and ease of tuning. This implementation includes:
- Actor-Critic architecture.
- Clipped objective function.
- Support for Gymnasium environments.

### 1. Overview and Classification

PPO is an **"on-policy"** reinforcement learning algorithm belonging to the **Actor-Critic** family. It has become the default algorithm for many academic and industrial applications (such as robotics or LLM alignment) due to its exceptional balance between ease of implementation, training stability, and performance.

### 2. The Fundamental Problem: Stability

Traditional policy gradient methods (like REINFORCE) suffer from significant instability. They are highly sensitive to the **learning rate**: an update to the parameters that is too large can lead to **"policy drift"** and a **catastrophic collapse** in performance from which the agent never recovers.

### 3. The Core Principle: The Trust Region

The central idea of PPO is to limit the magnitude of the policy update at each training step. It ensures that the new policy does not deviate too radically from the old one.  

While PPO is inspired by **TRPO (Trust Region Policy Optimization)**, which uses complex constraints (KL divergence) and second-order mathematics (conjugate gradient), PPO considerably simplifies this process by using a first-order method that is easier to implement and tune.

### 4. Key Mechanism: The "Clipped" Surrogate Objective

To enforce this stability, PPO uses a specific objective function called the **"clipped surrogate objective."** Here is how it works in broad terms:

1. **Probability Ratio:** The algorithm calculates a ratio $r_t(\theta)$ that compares the probability of taking an action under the new policy versus the old policy: $r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$.
2. **Clipping:** If this ratio moves too far from 1 (meaning the new policy changes the behavior too much), the objective is "cut" or "clipped." Specifically, the ratio is constrained within an interval $[1-\epsilon, 1+\epsilon]$ (where $\epsilon$ is often 0.2).
3. **Loss Function:** The optimized equation is the minimum between the normal objective and the clipped objective:
$$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t [\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$

This ensures that the agent is not incentivized to make excessively large updates, even if they appear beneficial in the short term.

### 5. Architecture and Components

* **Actor-Critic:** PPO uses two neural networks (or two heads of the same network). The **Actor** decides on actions (the policy), and the **Critic** estimates the state value to guide the actor.
* **Advantage Estimation:** It often uses **GAE (Generalized Advantage Estimation)** to calculate the advantage function $\hat{A}_t$, which measures how much better or worse an action was than expected.
* **Entropy:** To encourage exploration and prevent premature convergence to a sub-optimal solution, an **entropy bonus** is often added to the loss function.

## 📄 License

MIT License
