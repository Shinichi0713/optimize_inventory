# Importing the model (function approximator for Q-table)
# from model import QNetwork
from typing import Any, Mapping
import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import namedtuple, deque


BUFFER_SIZE = int(5 * 1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
gamma = 0.99            # discount factor
tau = 1e-3              # for soft update of target parameters
LR = 1e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network


## エージェントのNNモデル
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_unit=128, fc2_unit=128):
        super(QNetwork, self).__init__()     # calls __init__ method of nn.Module class
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_unit)
        self.fc2 = nn.Linear(fc1_unit, fc2_unit)
        self.fc3 = nn.Linear(fc2_unit, action_size)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._load_state_dict()

    # 順伝播
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save_to_state_dict(self):
        print("save network parameters")
        torch.save(self.state_dict(), 'model.pth')

    def _load_state_dict(self):
        if os.path.exists('model.pth'):
            print("load existing network parameters")
            self.load_state_dict(torch.load("model.pth"))
        self.to(self.device)


# エージェントの定義
class Agent():
    def __init__(self, state_size, action_size, seed):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        
        # Q- Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        
        # 行動ヒストリーの初期化
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # 時間ステップの初期化
        self.t_step = 0
    
    # 行動の実行→学習
    def step(self, state, action, reward, next_step, done):
        # 行動のメモリ
        self.memory.add(state, action, reward, next_step, done)

        # 学習
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:

            if len(self.memory) > BATCH_SIZE:
                experience = self.memory.sample()
                self.learn(experience, gamma)
    
    # 行動選択
    def act(self, state, eps=0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon greedy 行動選択
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    ## NNの学習
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        ## TODO: compute and minimize the loss
        criterion = torch.nn.MSELoss()
        # 学習
        self.qnetwork_local.train()
        # ターゲットは固定
        self.qnetwork_target.eval()
        # 出力の最大値を取得
        predicted_targets = self.qnetwork_local(states).gather(1,actions)
        with torch.no_grad():
            labels_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # ラベルの計算
        labels = rewards + (gamma * labels_next * (1 - dones))
        loss = criterion(predicted_targets, labels).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # NNの更新
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    # NNの更新 
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1 - tau) * target_param.data)
       

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experiences = namedtuple("Experience", field_names=["state",
                                                               "action",
                                                               "reward",
                                                               "next_state",
                                                               "done"])
        self.seed = random.seed(seed)
        
    def add(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)