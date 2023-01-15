import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym


class ReplayBuffer:
    """docstring"""

    def __init__(self, max_size, input_shape):

        self.mem_size = int(max_size)
        self.mem_cntr = 0

        self.state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)

        self.action_mem = np.zeros((self.mem_size), dtype=np.int64)
        self.reward_mem = np.zeros((self.mem_size), dtype=np.float32)
        self.terminal_mem = np.zeros((self.mem_size), dtype=np.uint8)

    def store(self, state, action, reward, state_, done):
        """docstring"""

        i = self.mem_cntr % self.mem_size
        self.state_mem[i] = state
        self.new_state_mem[i] = state_
        self.reward_mem[i] = reward
        self.action_mem[i] = action
        self.terminal_mem[i] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """docstring"""

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        terminal = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminal


class DDDQN(nn.Module):
    """dueling double dqn"""

    def __init__(self, lr, nactions, idim, name, ckpt_dir):
        super(DDDQN, self).__init__()

        self.ckpt_dir = ckpt_dir
        self.ckpt_file = os.path.join(self.ckpt_dir, name)

        self.fc1 = nn.Linear(*idim, 512)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, nactions)

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        """docstring"""

        x = F.relu(self.fc1(state))
        V = self.V(x)
        A = self.A(x)
        return V, A

    def save(self):
        torch.save(self.state_dict(), self.ckpt_file)

    def load(self):
        self.load_state_dict(torch.load(self.ckpt_file))


class Agent():
    """agent ... contains dddqn
    gamma = discount factor
    eps = epsilon
    """

    def __init__(
        self,
        gamma,
        eps,
        lr,
        nactions,
        idim,
        mem_size,
        batch_size,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        ckpt_dir=".",
    ):

        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.nactions = nactions
        self.idim = idim
        self.memsz = mem_size
        self.bsz = batch_size

        self.eps_min = eps_min
        self.eps_dec = eps_dec

        self.replace = replace
        self.ckpt_dir = ckpt_dir

        self.learn_step_counter = 0
        self.action_space = [i for i in range(self.nactions)]

        self.mem = ReplayBuffer(mem_size, idim)

        self.online = DDDQN(self.lr, self.nactions, idim, "lunar_land_dddqn1", self.ckpt_dir)
        self.target = DDDQN(self.lr, self.nactions, idim, "lunar_land_dddqn2", self.ckpt_dir)

    def choose_action(self, observation):
        """docstring"""

        if np.random.random() > self.eps:
            state = torch.tensor(np.array([observation]), dtype=torch.float, device=self.online.device)
            # print(state, state.view(-1).shape)
            _, advantage = self.online(state.view(-1))
            action = torch.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store(self, state, action, reward, state_, done):
        """docstring"""
        self.mem.store(state, action, reward, state_, done)

    def replace_target(self):
        """docstring"""

        if self.learn_step_counter % self.replace == 0:
            self.target.load_state_dict(self.online.state_dict())

    def decrement_epsilon(self):
        self.eps -= self.eps_dec if self.eps > self.eps_min else 0

    def save(self):
        self.online.save()
        self.target.save()

    def load(self):
        self.online.load()
        self.target.load()

    def learn(self):
        """docstring"""

        if self.mem.mem_cntr < self.bsz:
            return

        self.online.optim.zero_grad()
        self.replace_target()

        state, action, reward, new_state, done = self.mem.sample_buffer(self.bsz)
        states = torch.tensor(state, device=self.online.device)
        actions = torch.tensor(action, device=self.online.device)
        dones = torch.tensor(done, device=self.online.device)
        rewards = torch.tensor(reward, device=self.online.device)
        states_ = torch.tensor(new_state, device=self.online.device)

        idxs = np.arange(self.bsz)

        Vs, As = self.online(states)
        Vs2, As2 = self.target(states_)

        Vs_eval, As_eval = self.online(states_)

        qpred = torch.add(Vs, (As + As.mean(dim=1, keepdim=True)))[idxs, actions]
        qnext = torch.add(Vs2, (As2 + As2.mean(dim=1, keepdim=True)))
        qeval = torch.add(Vs_eval, (As_eval + As_eval.mean(dim=1, keepdim=True)))

        max_actions = torch.argmax(qeval, dim=1)
        qnext[dones] = 0.0
        qtarget = rewards + self.gamma * qnext[idxs, max_actions]

        loss = self.online.loss(qtarget, qpred).to(self.online.device)
        loss.backward()
        self.online.optim.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()


def main():
    """docstring"""

    env = gym.make("LunarLander-v2")
    ngames = 500
    load_ckpt = False

    agent = Agent(
        gamma=0.99,
        eps=1.0,
        lr=5e-4,
        idim=[8],
        nactions=4,
        mem_size=1e6,
        eps_min=0.01,
        batch_size=64,
        eps_dec=1e-3,
        replace=100,
    )

    if load_ckpt:
        agent.load()

    fname = 'temp'
    scores, eps_hist = [], []

    for i in range(ngames):
        done=False
        observation = env.reset()
        observation = observation[0] if type(observation) == tuple else observation
        score=0

        while not done:
            action = agent.choose_action(observation) # call something better
            observation_, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score+= reward
            agent.store(observation, action, reward, observation_, int(done))
            agent.learn()
            observation = observation_

        scores.append(score)
        avg = np.mean(scores[-10:])
        print(f'episode: {i:3d} | score: {score:4.1f} | avg: {avg:4.1f} | eps: {agent.eps:.2f}')

        if not i % 10:
            agent.save()

        eps_hist.append(agent.eps)

    plot_loss(list(range(ngames)), scores, eps_hist, fname)

if __name__ == "__main__":
    main()
