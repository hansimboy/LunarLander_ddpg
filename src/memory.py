import random
from collections import deque

class ReplayBuffer:
  def __init__(self, capacity):
    self.memory = deque(maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    batch = random.sample(self.memory, batch_size)
    states, action, rewards, next_states, dones = zip(*batch)
    return states, action, rewards, next_states, dones

  def __len__(self):
    return len(self.memory)

  def clear(self):
    self.memory.clear()

def soft_update(target, source, tau):
  for target_param, param in zip(target.parameters(), source.parameters()):
    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)