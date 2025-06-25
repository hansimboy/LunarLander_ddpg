# DDPG for LunarLanderContinuous-v3

This repository implements the Deep Deterministic Policy Gradient (DDPG) algorithm and applies it to the OpenAI Gym/Gymnasium LunarLanderContinuous-v3 environment.  
Based on the original paper: Lillicrap et al., “Continuous Control with Deep Reinforcement Learning” (2015) ([arXiv:1509.02971](https://arxiv.org/abs/1509.02971)).


---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation & Setup](#installation--setup)  
4. [Usage](#usage)  
   - [Training](#training)  
   - [Recording Video](#recording-video)  
5. [Hyperparameters](#hyperparameters)  
6. [Example Results](#example-results)  
7. [References](#references)  
8. [Contributing & License](#contributing--license)  

---

## Project Overview
This project provides:
- A PyTorch-based implementation of DDPG from scratch.
- Application to the LunarLanderContinuous-v3 environment so that an agent learns to land successfully.
- Actor-Critic network architecture.
- Automatic saving of the best model during training.
- Scripts for training and recording agent behavior as video.

---

## Features
- Pure PyTorch implementation of DDPG (no high-level RL libraries).
- Configurable hyperparameters via `argparse`
- Scripts:
  - `train.py` for training.
  - `record_video.py` for recording an episode as an MP4.

---

## Installation & Setup
- pip install --upgrade pip
- pip install -r requirements.txt

## Usage
Training:

```bash
python src/train.py
```
Recoding Video

```bash
python src/record_video.py\
  --actor-model-path models/best_actor_path.pth
```
## Hyperparameters
The default hyperparameters used for PPO training (as defined in `train.py`):

| Parameter          | Description                                          | Default                             |
| ------------------ | ---------------------------------------------------- | ----------------------------------- |
| `--env-name`       | Gym environment name                                 | `LunarLanderContinuous-v3`          |
| `--episodes`       | Number of training episodes                          | `500`                               |
| `--rollout-length` | Maximum steps per episode before reset               | `1000`                              |
| `--batch-size`     | Batch size for sampling from replay buffer           | `128`                               |
| `--memory-size`    | Memory size of replay buffer                         | `100000`                            |
| `--start-memory`   | Minimum replay memory size before starting training  | `10000`                             |
| `--lr`             | Learning rate for actor and critic optimizers        | `1e-3`                              |
| `--gamma`          | Discount factor                                      | `0.99`                              |
| `--tau`            | Soft update coefficient for target networks (Polyak) | `0.001`                             |
| `--noise-std`      | Initial standard deviation of exploration noise      | `1.0`                               |
| `--device`         | Torch device to use (`cpu` or `cuda`)                | `"cuda"` if available, else `"cpu"` |
| `--save-dir`       | Directory to save model checkpoints                  | `models`                            |
| `--seed`           | Random seed for reproducibility (optional)           | `None`                              |

You can override these when running `train.py`, for example:

```bash
python src/train.py \
  --iteration 200 \
  --actors 8 \
  --rollout-length 500 \
  --epochs 10 \
  --batch-size 64 \
  --clip-eps 0.2 \
  --lr 3e-4 \
  --gamma 0.99 \
  --lamb 0.95 \
  --vf-coef 0.05 \
  --entropy-bonus-coef 0.005 \
  --device cpu \
  --save-dir models \
  --seed 42
```

## References
- Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T., Tassa, Y., Silver, D., & Wierstra, D. (2015). Continuous Control with Deep Reinforcement Learning. [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)
- Gymnasium LunarLanderContinuous-v3 documentation
- PyTorch documentation

## Contributing & License
Contributions welcome.  
Licensed under MIT License. See [LICENSE](LICENSE.txt) for details.
