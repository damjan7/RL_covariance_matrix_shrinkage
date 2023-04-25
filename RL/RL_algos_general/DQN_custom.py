import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import gym
from gym.wrappers import Monitor
from collections import deque


class DQN(nn.Module):
    '''
    A Network to approximate the Q-function (= Q value for every state action pair)
    '''

    def __init__(self, num_features, num_actions):
        super(DQN, self).__init__()
        self.state_space = num_features
        self.action_space = num_actions
        self.hidden = self.state_space * 2
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=True)
        self.l2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.l3 = nn.Linear(self.hidden, self.action_space, bias=True)  # output Q for every possible action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


def select_epsilon_greedy_action(state, epsilon, env, main_network):
    """Take random action with probability epsilon, else take best action."""
    result = np.random.uniform()
    if result < epsilon:
        return env.action_space.sample()  # Random action
    else:
        qs = main_network(state).cpu().data.numpy()
        return np.argmax(qs)  # Greedy action for state.


def train_step(states, actions, rewards, next_states, dones, env, main_network, target_network, loss_fn, optimizer):
    """Perform a training iteration on a batch of data sampled from the experience
    replay buffer.
    """
    # Calculate targets.
    DISCOUNT = 0.99
    # next states need to provide the covmat and the target
    max_next_q_vals = target_network(next_states).max(-1).values  # according to our target network
    target = rewards + (1.0 - dones) * DISCOUNT * max_next_q_vals
    q_vals_current = main_network(states)
    action_masks = F.one_hot(actions.to(torch.int64), env.action_space.n)
    masked_q_vals = (action_masks * q_vals_current).sum(dim=-1)
    loss = loss_fn(masked_q_vals, target.detach())
    # nn.utils.clip_grad_norm_(loss, max_norm=10)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # optimizes weights of main network as only it isy
    return loss


def main(env):
    steps = 2500
    epsilon = 0.3  # for exploration
    gamma = 0.99
    loss_history = []
    reward_history = []
    num_episodes = 1000
    max_position = -0.4
    learning_rate = 0.03
    successes = 0
    batch_size = 32
    position = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    buffer = ReplayBuffer(1000000, device=device)  # changing replay buffer size added quite some stability
    cur_frame = 0

    # Initialize Policy
    # env = gym.make('CartPole-v1')
    state = env.reset()
    # as in the paper I need two networks
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n  # should be 100 in discretized action space
    main_network = DQN(num_features, num_actions)
    target_network = DQN(num_features, num_actions)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(main_network.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    # Start training. Play game once and then train with a batch.
    last_100_ep_rewards = []
    for episode in range(num_episodes + 1):
        state = env.reset().astype(np.float32)  # returns starting current state
        #  The position of the car is assigned a uniform random value in [-0.6 , -0.4]
        ep_reward, done = 0, False
        while not done:

            # actually the state should be the covmat and the shrinkage target
            # such that the network makes a prediction based on them...
            state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)


            action = select_epsilon_greedy_action(state_in, epsilon, env, main_network)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.astype(np.float32)
            ep_reward += reward
            # Save to experience replay.
            # run one episode and add it to the buffer so we can replay it
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            cur_frame += 1

            # Copy main_nn weights to target_nn every 2000 frames
            # this should be every XY episodes
            if cur_frame % 2000 == 0:  # change to 10'000
                target_network.load_state_dict(
                    main_network.state_dict())  # copying weights from main network to target network

            # Train neural network.
            if len(buffer) > batch_size:  # if buffer has large enought size --> sample from it
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                loss = train_step(states, actions, rewards, next_states, dones, env, main_network, target_network,
                                  loss_fn, optimizer)  # updates weights of nn for each batch

        # scheduler outside of train loop
        scheduler.step()  # let's see if this improves performance

        if episode < 2000 and epsilon > 0.05:
            epsilon -= 0.0005
        if 0 < epsilon <= 0.05:
            epsilon -= 0.0001
        if epsilon < 0:
            epsilon = 0

        if len(last_100_ep_rewards) == 100:
            last_100_ep_rewards = last_100_ep_rewards[1:]
        last_100_ep_rewards.append(ep_reward)

        if episode % 50 == 0:
            print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}.'
                  f' Average Reward in Last 100 Episodes: {np.mean(last_100_ep_rewards):.2f}')
            print(f"learning rate: {optimizer.param_groups[0]['lr']}")

    env.close()


#### should not need replay buffer or rewrite it
class ReplayBuffer:
    def __init__(self, buffer_size, device="cpu"):
        self.buffer = deque(maxlen=buffer_size)  # see collections docs
        self.device = device

    def add(self, state, action, rew, next_state, done):
        self.buffer.append((state, action, rew, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = torch.as_tensor(np.array(states), device=self.device)
        actions = torch.as_tensor(np.array(actions), device=self.device)
        rewards = torch.as_tensor(
            np.array(rewards, dtype=np.float32), device=self.device
        )
        next_states = torch.as_tensor(np.array(next_states), device=self.device)
        dones = torch.as_tensor(np.array(dones, dtype=np.float32), device=self.device)
        return states, actions, rewards, next_states, dones
