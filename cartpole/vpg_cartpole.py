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

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    track: bool = False
    torch_deterministic: bool = True
    wandb_project_name: str = "rl_gym"
    epochs: int = 1

class Value(nn.Module):
    def __init__(self, env: RecordVideo):
        super(Value, self).__init__()
        obs_dim = env.unwrapped.observation_space.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax()
        )

    def forward(self, x):
        return self.fc(x)


class Policy(nn.Module):
    def __init__(self, env: RecordVideo):
        super(Policy, self).__init__()
        obs_dim = env.unwrapped.observation_space.shape[0]

        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env.unwrapped.action_space.shape[0]),
            nn.Softmax()
        )
    
    def forward(self, x):
        logits = self.fc(x)
        return logits
    
    def get_action(self, obs):
        logits = self(obs)
        distribution = Categorical(logits=logits)

        action = distribution.sample()
        action_probs = distribution.probs

        return action, action_probs


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

    env = gym.make("CartPole-v1")

    train_env = RecordVideo(env, RECORDING_DIR, name_prefix="train")
    
    for k in range(args.epochs):
        trajectories = []
        for i in range(args.max_trajectory_steps):
            