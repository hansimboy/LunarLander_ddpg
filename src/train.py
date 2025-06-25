import gymnasium as gym
import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.model import CriticNetwork, ActorNetwork
from src.memory import ReplayBuffer, soft_update
from src.env_utils import make_env, preprocess_obs

def parse_args():
    parser = argparse.ArgumentParser(description="DDPG for LunarLanderContinuous-v3")
    parser.add_argument("--env-name", type=str, default="LunarLanderContinuous-v3")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rollout-length", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--start-memory", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)
    # 환경 생성
    env = make_env(args.env_name, args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    critic = CriticNetwork(obs_dim, act_dim).to(device)
    target_critic = CriticNetwork(obs_dim, act_dim).to(device)
    actor = ActorNetwork(obs_dim, act_dim).to(device)
    target_actor = ActorNetwork(obs_dim, act_dim).to(device)
    
    optimizer_critic = optim.Adam(critic.parameters(), lr=args.lr)
    optimizer_actor = optim.Adam(actor.parameters(), lr=args.lr)

    memory = ReplayBuffer(args.memory_size)

    best_reward = -float("inf")

    os.makedirs(args.save_dir, exist_ok=True)

    for episode in range(args.episodes):
        total_reward = 0

        noise = torch.rand(args.rollout_length, 2, device=device) * np.maximum(0.4, args.noise_std * 0.99 ** (episode // 5))
        obs_np, _ = env.reset()

        for t in range(args.rollout_length):
            obs = preprocess_obs(obs_np, device)

            with torch.no_grad():
                action = actor(obs).squeeze(0) + noise[t]
                action = torch.clamp(action, -1, 1)

            action_np = action.detach().cpu().numpy()

            next_obs, reward, done, truncated, _ = env.step(action_np)
            real_done = done or truncated
            memory.push((obs_np, action_np, reward, next_obs, real_done))
            obs_np = next_obs

            total_reward += reward

            if len(memory) >= args.start_memory:
                states_np, actions_np, rewards_np, next_states_np, dones_np = memory.sample(args.mini_batch)
                states = torch.from_numpy(np.array(states_np)).float().to(device)
                actions = torch.from_numpy(np.array(actions_np)).float().to(device)
                rewards = torch.tensor(rewards_np, dtype=torch.float32).to(device)
                next_states = torch.from_numpy(np.array(next_states_np)).float().to(device)
                dones = torch.tensor(dones_np, dtype=torch.float32).to(device)
                
                with torch.no_grad(): # target network 계산할 때는 gradient 연결 x
                    next_actions = target_actor(next_states) # (batch,1)
                    target_Q = target_critic(next_states, next_actions) # (batch)
                    y = rewards + args.gamma * (1-dones) * target_Q # (batch)
                Q = critic(states, actions) # (batch)

                loss_critic = F.mse_loss(Q, y)
                optimizer_critic.zero_grad()
                loss_critic.backward()
                optimizer_critic.step()

                loss_actor = -critic(states, actor(states)).mean()

                optimizer_actor.zero_grad()
                loss_actor.backward()
                optimizer_actor.step()

                soft_update(target_critic, critic, args.tau)
                soft_update(target_actor, actor, args.tau)

            if real_done:
                obs_np, _ = env.reset()

                if total_reward > best_reward:
                    best_reward = total_reward
                    actor_save_path = os.path.join(args.save_dir, "best_actor_path.pth")
                    critic_save_path = os.path.join(args.save_dir, "best_critic_path.pth")
                    torch.save(actor.state_dict(), actor_save_path)
                    torch.save(critic.state_dict(), critic_save_path)
                    print(f"{episode}th episode, best reward: {np.ceil(best_reward)}")

                if episode % 50 == 0:
                    print(f"{episode}th episode, reward: {np.ceil(total_reward)}")

                total_reward = 0
                break
    env.close()

if __name__ == "__main__":
    main()