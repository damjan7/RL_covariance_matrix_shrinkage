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
        self.hidden = 100
        self.l1 = nn.Linear(self.state_space, self.hidden, bias=True)
        self.l2 = nn.Linear(self.hidden, self.hidden, bias=True)
        self.l3 = nn.Linear(self.hidden, self.action_space, bias=True)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

# Parameters
steps = 2500
epsilon = 0.3  # for exploration
gamma = 0.99
loss_history = []
reward_history = []
episodes = 1000
max_position = -0.4
learning_rate = 0.03
successes = 0
position = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize Policy
env = gym.make('CartPole-v1')
state = env.reset()
# as in the paper I need two networks
num_features = env.observation_space.shape[0]
num_actions = env.action_space.n
main_network = DQN(num_features, num_actions)
target_network = DQN(num_features, num_actions)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(main_network.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

# also like in the 2015 paper, we need to create a replac buffer to eliminate correlation
# between samples as we will randomly sample from the replay buffer
# an obs in the buffer consists of (s, a, r, s') s=state, a=action
# for the buffer we use a
class ReplayBuffer:
    def __init__(self, buffer_size, device="cpu"):
        self.buffer = deque(maxlen=buffer_size)  # see collections docs
        self.device = device

    def add(self, state, action, rew, next_state, done):
        self.buffer.append((state, action, rew, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        #### may need to change s.t. it returns tensors not sure
        ### MY SOL DOESN'T WORK YET BUT I THINK HIS IS REALLY INEFFICIENT!
    #    indices = np.random.choice(len(self.buffer), num_samples)  # the random samples we should take
    #    states, actions, rewards, next_states, dones = \
    #        self.buffer[0][indices], self.buffer[1][indices],\
    #        self.buffer[2][indices], self.buffer[3][indices], self.buffer[4][indices]
    #    return states, actions, rewards, next_states, dones
            ######
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

def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""
  result = np.random.uniform()
  if result < epsilon:
    return env.action_space.sample() # Random action (left or right).
  else:
    qs = main_network(state).cpu().data.numpy()
    return np.argmax(qs) # Greedy action for state.


def train_step(states, actions, rewards, next_states, dones):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer.
  """
  # Calculate targets.
  DISCOUNT = 0.99

  max_next_q_vals = target_network(next_states).max(-1).values  # according to our target network
  target = rewards + (1.0 - dones) * DISCOUNT * max_next_q_vals
  q_vals_current = main_network(states)
  action_masks = F.one_hot(actions.to(torch.int64), num_actions)
  masked_q_vals = (action_masks * q_vals_current).sum(dim=-1)
  loss = loss_fn(masked_q_vals, target.detach())
  #nn.utils.clip_grad_norm_(loss, max_norm=10)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()  # optimizes weights of main network as only it isy
  return loss



# Hyperparameters.
num_episodes = 4000
epsilon = 1.0
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(1000000, device=device)  # changing replay buffer size added quite some stability
cur_frame = 0

# Start training. Play game once and then train with a batch.
last_100_ep_rewards = []
for episode in range(num_episodes+1):
  state = env.reset().astype(np.float32)  # returns starting current state
  #  The position of the car is assigned a uniform random value in [-0.6 , -0.4]
  ep_reward, done = 0, False
  while not done:

    state_in = torch.from_numpy(np.expand_dims(state, axis=0)).to(device)
    action = select_epsilon_greedy_action(state_in, epsilon)
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
      target_network.load_state_dict(main_network.state_dict())  # copying weights from main network to target network

    # Train neural network.
    if len(buffer) > batch_size:  # if buffer has large enought size --> sample from it
      states, actions, rewards, next_states, dones = buffer.sample(batch_size)
      loss = train_step(states, actions, rewards, next_states, dones)  # updates weights of nn for each batch

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






'''
OLD CODE IMPLEMENTATION FROM INTERNET
for episode in range(episodes):
    print("We are at episode: ", episode)
    episode_loss = 0
    episode_reward = 0
    state = env.reset()

    for s in range(steps):
        if episode % 200 == 0 and episode > 0:
            env.render()

        # Get first action value function
        if type(state) is tuple:
            state = state[0]
        Q = pol(Variable(torch.from_numpy(state).type(torch.FloatTensor)))

        # Choose epsilon-greedy action
        if np.random.rand(1) < epsilon:
            action = np.random.randint(0, 3)  # sample from left, right, do nothing
        else:
            _, action = torch.max(Q, -1)
            action = action.item()

        # Step forward and receive next state and reward
        new_state, reward, done, _ = env.step(action)

        new_Q = pol(Variable(torch.from_numpy(new_state).type(torch.FloatTensor)))
        max_new_Q, _ = torch.max(new_Q, -1)

        # Create target Q value for training the policy
        Q_target = Q.clone()
        Q_target = Variable(Q_target.data)
        Q_target[action] = reward + torch.mul(max_new_Q.detach(), gamma)

        # Calculate loss
        loss = loss_fn(Q, Q_target)

        # Update policy
        pol.zero_grad()
        loss.backward()
        optimizer.step()

        # Record history
        episode_loss += loss.item()
        episode_reward += reward

        # Keep track of max position
        if new_state[0] > max_position:
            max_position = new_state[0]

        if done:
            if new_state[0] >= 0.5:
                print("Success at episode ", episode)
                epsilon *= .99                 # Adjust epsilon
                scheduler.step()                # Adjust learning rate
                successes += 1                # Record successful episode

            # Record history
            loss_history.append(episode_loss)
            reward_history.append(episode_reward)
            weights = np.sum(np.abs(pol.l2.weight.data.numpy()))+np.sum(np.abs(pol.l1.weight.data.numpy()))
            position.append(new_state[0])

            break
        else:
            state = new_state

print('successful episodes: {}'.format(successes))

print("Printing loss history and reward history")
print(loss_history)
print(reward_history)
'''