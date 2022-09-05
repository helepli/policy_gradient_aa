""" Monte-Carlo Policy Gradient """
# from https://github.com/sonic1sonic/Monte-Carlo-Policy-Gradient-REINFORCE/blob/master/REINFORCE.py
from __future__ import print_function
import os
import sys
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

from torch.autograd import Variable
import argparse

DEVICE = torch.device('cpu')

MAX_TIMESTEPS = 108000

ALPHA = 1e-6
GAMMA = 0.99


gym.envs.registration.register(
    id='MyGrid-v0',
    entry_point='myGrid:myGrid',
    kwargs={'y' : 11, 'x' : 9} # 20 18
)

def get_probas(state, agent):
    probs = agent.forward(state)
    probs = torch.squeeze(probs, 0)
    return probs


class reinforce(nn.Module):

    def __init__(self, advisor, args, env):
        super(reinforce, self).__init__()
        # policy network

        state_shape = env.observation_space.shape[0]
        print('State shape:', state_shape)

        aspace = env.action_space
        aspace = [aspace]
        num_actions = int(np.prod([a.n for a in aspace]))
        print('Number of actions:', num_actions)
        self.num_actions = num_actions


        self.fc1 = nn.Linear(state_shape, 128)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, num_actions)
        self.softmax = nn.Softmax()

        self.advisor = advisor
        self.args = args

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def get_action(self, state):
        state = Variable(torch.Tensor(state))
        state = torch.unsqueeze(state, 0)

        actor = get_probas(state, self)
        actor = actor.detach().numpy()
        if self.advisor != None:
            advice = get_probas(state, self.advisor)
            advice+=0.001
            advice = advice.detach().numpy()
        else:
            advice = np.ones(self.num_actions)
        if self.args.intersection:
            mixed = actor*advice # policy intersection
        else:
            mixed = actor+advice # policy union
        mixed /= mixed.sum()
        action = np.random.choice([a for a in range(self.num_actions)], p=mixed)

        return action

    def pi(self, s, a):
        s = Variable(torch.Tensor([s]))
        probs = self.forward(s)
        probs = torch.squeeze(probs, 0)
        return probs[a]


    def update_weight(self, states, actions, rewards, optimizer):
        G = Variable(torch.Tensor([0]))
        # for each step of the episode t = T - 1, ..., 0
        # r_tt represents r_{t+1}
        for s_t, a_t, r_tt in zip(states[::-1], actions[::-1], rewards[::-1]):
            G = Variable(torch.Tensor([r_tt])) + GAMMA * G
            # learning correction
            if self.advisor != None and self.args.lc:
                s_t = Variable(torch.Tensor([s_t]))
                advice = get_probas(s_t, self.advisor)
                actor = get_probas(s_t, self)
                if self.args.union:
                    mixed = actor+advice
                else:
                    mixed = actor*advice
                mixed /= mixed.sum()
                mixed_proba = mixed[a_t]
                loss = (-1.0) * G * torch.log(mixed_proba)
            else:
                if self.args.bad: # only concerns training advisors
                    loss = G * torch.log(self.pi(s_t, a_t)) # update policy parameter \theta -> makes good advisor
                else:
                    loss = (-1.0) * G * torch.log(self.pi(s_t, a_t)) #-> makes bad advisor
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():


    parser = argparse.ArgumentParser(description="Transfer Learning with Policy Gradient: advisors are supported")

    parser.add_argument("--env", required=True, type=str, help="Gym environment to use")
    parser.add_argument("--episodes", default=50000, type=int, help="episodes")
    parser.add_argument("--name", type=str, default='', help="Experiment name")
    parser.add_argument("--save", type=str, help="Basename of saved weight files. If not given, nothing is saved")
    parser.add_argument("--advisor", type=str, default=None, help="model zip file of the policy that is going to be loaded and used as advisors")
    parser.add_argument("--lc", type=int, default=0, help="using learnign correctio or not")
    parser.add_argument("--bad", type=int, default=0, help="learning a bad advisor instead of a good one")
    parser.add_argument("--union", type=int, default=0, help="policy union used, rather than policy union")

    args = parser.parse_args()

    f = open(args.name+str(random.random()).strip('0.'), 'w')
    env = gym.make(args.env)
    if args.advisor != None:
        advisor = torch.load(args.advisor, map_location=DEVICE)
    else:
        advisor = None

    agent = reinforce(advisor, args, env)
    optimizer = optim.Adam(agent.parameters(), lr=ALPHA)


    for i_episode in range(args.episodes):

        state = env.reset()

        states = []
        actions = []
        rewards = [0]   # no reward at t = 0
        advice = []
        cumulative = 0.0

        for timesteps in range(MAX_TIMESTEPS):

            action = agent.get_action(state)

            states.append(state)
            actions.append(action)

            state, reward, done, _ = env.step(action)

            rewards.append(reward)
            cumulative += reward

            if done:
                print("Episode {} finished after {} timesteps, rewards: {}".format(i_episode, timesteps+1, cumulative))
                print(cumulative, file=f)
                f.flush()
                break

        agent.update_weight(states, actions, rewards, optimizer)
        if args.save:
            torch.save(agent, args.save)

    env.close()

if __name__ == "__main__":
    main()
