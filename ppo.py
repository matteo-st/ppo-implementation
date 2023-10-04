# !pip install gymnasium[atari]
# !pip install gymnasium[accept-rom-license]

import collections
import gymnasium as gym # Version 0.28.1
import numpy as np # Version 1.23.5
import random 
import time
from torchvision import transforms
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCriticPyTorch(nn.Module):
    def __init__(self, env):
        super(ActorCriticPyTorch, self).__init__()

        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, env.env.action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.network(x)
        policy = self.softmax(self.actor(hidden))
        value = self.critic(hidden)
        
        return policy, value
    
    def reproduce_action_value(self, x, action):

        hidden = self.network(x)
        policy = self.softmax(self.actor(hidden))
        value = self.critic(hidden)
        probs = Categorical(policy)
        return action, probs.log_prob(action), probs.entropy(), value
    


class FrameProcessor:
    def __init__(self, frame_height=84, frame_width=84):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def process(self, frame) -> torch.Tensor:
        
        frame = torch.tensor(frame).to(device)
        frame = frame.permute(2, 0, 1) # Convert from (H,W,C) to (C,H,W) format
        frame = frame.float() / 255 # Convert from Integer fromat to float format
        frame = frame[:,34:-16, :] # Cropping useless parts for training
        transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), # RGB to grzy scale
                transforms.Resize((84, 84)),
            ])
        
        return transform(frame)
            

class Atari(object):
    """Wrapper for the environment provided by gym"""
    def __init__(self, envName, no_op_steps=10, agent_history_length=4):
        self.env = gym.make(envName, render_mode="rgb_array")
        self.frame_processor = FrameProcessor()
        self.observation_space_shape = (4, 84, 84)
        self.state = None
        self.lives = 0
        self.no_op_steps = no_op_steps
        self.agent_history_length = agent_history_length
        self.action_probs = []
        self.was_real_terminated = True # To check if the game was really over


    def reset(self, evaluation=False):
        """
        Args:
            evaluation: A boolean saying whether the agent is evaluating or training
        Resets the environment and stacks four frames ontop of each other to 
        create the first state
        """
        if self.was_real_terminated:
            
            frame, info = self.env.reset()
        else: 
            # no-op step to advance from terminal/lost life state
            frame, _, terminal,_, info = self.env.step(0)
            # The no-op step can lead to a game over, so we need to check it again
            # to see if we should reset the environment
            if terminal:
                frame, info = self.env.reset()
            
        self.lives = info['lives']

        processed_frame = self.frame_processor.process(frame).unsqueeze(0)   # (★★★)
        self.state = torch.repeat_interleave(processed_frame, repeats=self.agent_history_length, dim=1)

        return self.state, info
        

    def step(self, action):
        """
        Args:
            action: Integer, action the agent performs
        Performs an action and observes the reward and terminal state from the environment
        """
        new_frame, reward, terminal,_, info = self.env.step(action)  # (5★)
        self.was_real_terminated = terminal
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        if 0 < info['lives'] < self.lives:

            terminal = True

        self.lives = info['lives']
        
        processed_new_frame = self.frame_processor.process(new_frame).unsqueeze(0)   # (6★)
        new_state = torch.cat((self.state[:,1:, :, :], processed_new_frame), dim=1)  

        self.state = new_state
        
        return new_state, reward, terminal,_, info
    
    def clip_reward(self, reward):
    # Clip the reward to {+1, 0, -1} by its sign.
        return np.sign(reward) 


ENV_NAME = "ALE/Breakout-v5"
EXP_NAME = "1"

SEED = 7

# Control parameters
DISCOUNT_FACTOR = 0.99           
GAE_LAMBDA = 0.95               
MAX_STEPS = 100000          
NO_OP_STEPS = 10                                 
UPDATE_FREQ = 128                  
UPDATE_EPOCHS = 4
LEARNING_RATE = 0.00001          
ANNEAL_LR = True                         
BATCH_SIZE = UPDATE_FREQ                        
NUM_MINIBATCH = 4
MINIBATCH_SIZE = int(BATCH_SIZE // NUM_MINIBATCH)
NUM_UPDATES = MAX_STEPS // BATCH_SIZE
CLIP_COEF = 0.1
CLIP_VLOSS = True
VF_COEF = 0.5
ENT_COEF = 0.01
TARGET_KL = None
NORM_ADV = True
MAX_GRAD_NORM = 0.5


run_name = f"{EXP_NAME}__{SEED}__{int(time.time())}"

writer = SummaryWriter(f"runs/{run_name}")

# seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

atari = Atari(ENV_NAME, NO_OP_STEPS)
agent = ActorCriticPyTorch(atari).to(device)
optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

obs = torch.zeros((UPDATE_FREQ, ) + atari.observation_space_shape).to(device)
actions = torch.zeros((UPDATE_FREQ, ) + atari.env.action_space.shape).to(device)
logprobs = torch.zeros(UPDATE_FREQ).to(device)
rewards = torch.zeros(UPDATE_FREQ).to(device)
terminales = torch.zeros(UPDATE_FREQ).to(device)
values = torch.zeros(UPDATE_FREQ).to(device)

start_time = time.time()

global_step = 0
next_state, _ = atari.reset()
next_terminal = torch.zeros(1).to(device)
episode_reward = 0

for update in range(NUM_UPDATES):

    if ANNEAL_LR:
        frac = 1.0 - (update - 1.0) / NUM_UPDATES
        lrnow = frac * LEARNING_RATE
        optimizer.param_groups[0]["lr"] = lrnow
    
    for step in range(UPDATE_FREQ):
        
        global_step += 1
        obs[step] = next_state
        terminales[step] = next_terminal
        

        with torch.no_grad():

            probs, value = agent(next_state)
            probs = Categorical(probs)
            action = probs.sample()
            logprob = probs.log_prob(action)
            values[step] = value.flatten() 
        
        actions[step] = action
        logprobs[step] = logprob

        next_state, reward, next_terminal,_, info = atari.step(action.item())
        episode_reward += reward 
        reward = atari.clip_reward(reward)
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_terminal = torch.tensor(next_terminal).to(device)
        
        if next_terminal:

            next_state, _ = atari.reset()
            if atari.was_real_terminated:
                print(f"global_step={global_step}, episodic_return={episode_reward}")
                writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                episode_reward = 0

    with torch.no_grad():

        _, next_value = agent(next_state)
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0

        for t in reversed(range(UPDATE_FREQ)):

            if t == (UPDATE_FREQ - 1):
                nextnonterminal = 1 - next_terminal.float()
                nextvalues = next_value
            else:
                nextnonterminal = 1 - terminales[t + 1]
                nextvalues = values[t+1]
            delta = rewards[t] + DISCOUNT_FACTOR * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + DISCOUNT_FACTOR * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + values

    b_inds = np.arange(BATCH_SIZE)
    clipfracs = []

    for epoch in range(UPDATE_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
            end = start + MINIBATCH_SIZE
            mb_inds = b_inds[start:end]

            _, new_logprob, entropy, newvalue  = agent.reproduce_action_value(obs[mb_inds], actions.long()[mb_inds])
            logratio = new_logprob - logprobs[mb_inds]
            ratio = logratio.exp()

            with torch.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > CLIP_COEF).float().mean().item()]

            mb_advantages = advantages[mb_inds]
            if NORM_ADV:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)

            if CLIP_VLOSS:
                v_loss_unclipped = (newvalue - returns[mb_inds]) ** 2
                v_clipped = values[mb_inds] + torch.clamp(
                    newvalue - values[mb_inds],
                    -CLIP_COEF,
                    CLIP_COEF,
                    )
                v_loss_clipped = (v_clipped - returns[mb_inds]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()

            else:
                v_loss = 0.5 * ((newvalue - returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
            optimizer.step()

        if TARGET_KL is not None:
            if approx_kl > TARGET_KL:
                break

    y_pred, y_true = values.cpu().numpy(), returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
writer.close()