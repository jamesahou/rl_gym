import os
import random
import time
from dataclasses import dataclass

import tyro
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch
import gymnasium as gym

from gymnasium.wrappers import RecordVideo

import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    track: bool = False
    torch_deterministic: bool = False
    wandb_project_name: str = "rl_gym"
    epochs: int = 50
    max_timesteps: int=5000
    lr: float = 1e-2

class Policy(nn.Module):
    def __init__(self, env: RecordVideo):
        super(Policy, self).__init__()
        obs_dim = env.unwrapped.observation_space.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.Tanh(),
            nn.Linear(32, env.unwrapped.action_space.n),
        )
    
    def forward(self, x):
        logits = self.fc(x)
        distribution = Categorical(logits=logits)
        return distribution
    
    def get_action(self, obs):
        distribution = self(obs)
        action = distribution.sample()
        action_probs = distribution.log_prob(action)

        return action.item(), action_probs

def compute_loss(policy, obs, act, weights):
    logp = policy(obs).log_prob(act)
    return -(logp * weights).mean()


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"CartPole_VPG_{args.seed}_{int(time.time())}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    RECORDING_DIR = f"../recordings/{run_name}"
    os.makedirs(RECORDING_DIR, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env = gym.make("CartPole-v1", render_mode="rgb_array")

    train_env = RecordVideo(env, RECORDING_DIR, name_prefix="train")
    
    policy_net = Policy(train_env)
    policy_net = policy_net.to(device)
    # value_net = Value(train_env)
    optimizer = Adam(policy_net.parameters(), lr=args.lr)
    for k in range(args.epochs):
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_returns = []
        batch_lens = []

        batch_logprobs = []
        
        done = False
        episode_rewards = []
        obs, _ = env.reset()

        total_timesteps = 0
        while True:
            batch_obs.append(obs.copy())

            action, log_prob = policy_net.get_action(torch.as_tensor(obs, dtype=torch.float32).to(device))
            obs, reward, done, _, _ = env.step(action)
        
            batch_actions.append(action)
            batch_logprobs.append(log_prob)
            episode_rewards.append(reward)

            if done:
                episode_return = sum(episode_rewards)
                episode_length = len(episode_rewards)

                batch_returns += episode_length * [episode_return]
                batch_lens.append(episode_length)

                obs, _ = env.reset()
                episode_rewards = []
                done = False

                total_timesteps += episode_length
                writer.add_scalar("charts/episodic_return", episode_return, total_timesteps)
                writer.add_scalar("charts/episodic_length", episode_length, total_timesteps)

                if total_timesteps > args.max_timesteps:
                    print(f"Reached max time steps with {len(batch_lens)} episodes recorded")
                    break

        optimizer.zero_grad()    
        batch_loss = compute_loss(policy=policy_net, obs=torch.as_tensor(batch_obs, dtype=torch.float32).to(device),
                                  act=torch.as_tensor(batch_actions, dtype=torch.int32).to(device),
                                  weights=torch.as_tensor(batch_returns, dtype=torch.float32).to(device)
                                  )
        batch_loss.backward()
        optimizer.step()
        
        print(f"epoch {k}: \t loss: {batch_loss} \t avg return {np.mean(batch_returns)} \t avg episode length {np.mean(batch_lens)}")

    


        
