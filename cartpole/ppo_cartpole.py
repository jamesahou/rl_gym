import tyro
from dataclasses import dataclass
import time
import os
import random

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from torch.utils.tensorboard import SummaryWriter

@dataclass
class Args:
    seed: int = 1
    cuda: bool = True
    torch_deterministic: bool = False
    num_iterations: int = 1000
    gamma: float = 0.99
    num_envs: int = 4
    num_steps: int = 128
    op_lr: float = 2.5e-4
    op_eps: float = 1e-5
    num_minibatches: int = 4
    epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    total_timesteps: int = 500000
    update_epochs: int = 4
    lam: float = 0.95

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()

        input_dim = env.single_observation_space.shape[0]
        out_dim = env.single_action_space.n
        self.policy = nn.Sequential(
            layer_init(nn.Linear(input_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, out_dim), std=0.01)
        )
        
        self.value = nn.Sequential(
            layer_init(nn.Linear(input_dim, 32)),
            nn.Tanh(),
            layer_init(nn.Linear(32, 1), std=1.0)
        )
    
    def get_value(self, obs):
        return self.value(obs)
    
    def get_action_and_value(self, obs, action=None):
        logits = self.policy(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.get_value(obs)

def make_env(gym_id, seed, idx, capture_video, recording_dir):
    def f():
        env = gym.make(gym_id, render_mode="rgb_array")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:    
                env = RecordVideo(env, recording_dir, name_prefix="train")
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return f

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"Acrobot_PPO_{args.seed}_{int(time.time())}"

    writer = SummaryWriter("runs/" + run_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    RECORDING_DIR = f"../recordings/{run_name}"
    os.makedirs(RECORDING_DIR, exist_ok=True)

    # set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    envs = gym.vector.AsyncVectorEnv([make_env("Acrobot-v1", args.seed + i, i, True, RECORDING_DIR) for i in range(args.num_envs)])

    agent = Agent(envs).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.op_lr, eps=args.op_eps)

    # obs, actions, logprobs, rewards, dones, values
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()

    next_obs, _ = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    args.batch_size = int(args.num_envs * args.num_steps)   # N * T
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps) # for K in iterations
    
    for update in range(1, num_updates+1):
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = np.logical_or(terminations, truncations) # torch.Tensor (alias for float), torch.tensor infers
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                        writer.add_scalar("charts/episode_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episode_length", info["episode"]["l"], global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.lam * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten all to (TOTAL TIMESTEPS, ...)
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                minibatch_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[minibatch_inds], b_actions[minibatch_inds])

                logratio = (newlogprob - b_logprobs[minibatch_inds])
                ratio = logratio.exp()

                mb_advantages = b_advantages[minibatch_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                clipped_obj = torch.clamp(ratio, 1-args.epsilon, 1+args.epsilon) * mb_advantages
                loss_clip = torch.min(ratio * mb_advantages, clipped_obj).mean()
                
                newvalue = newvalue.view(-1)
                loss_value = 0.5 * ((newvalue - b_returns[minibatch_inds])**2).mean()

                loss_entropy = entropy.mean()

                optimizer.zero_grad()
                # gradient ascent instead of descent
                loss = - (loss_clip - loss_value * args.value_coef + loss_entropy * args.entropy_coef)
                loss.backward()
                optimizer.step()

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", loss_value.item(), global_step)
        writer.add_scalar("losses/policy_loss", loss_clip.item(), global_step)
        writer.add_scalar("losses/entropy", loss_entropy.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()