# coding: utf-8
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from envs import IPD
from trainer import train


class Hp():
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.lr_v = 0.1
        self.gamma = 0.96
        self.n_update = 200
        self.len_rollout = 150
        self.batch_size = 128
        self.use_baseline = True
        self.seed = 42


hp = Hp()
ipd = IPD(hp.len_rollout, hp.batch_size)


class Memory():
    def __init__(self, args):
        self.args = args

        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(self.args.gamma * torch.ones(*rewards.size()), dim=1) / self.args.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(self.magic_box(dependencies) * discounted_rewards, dim=1))

        if self.args.use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - self.magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)

    def magic_box(self, x):
        return torch.exp(x - x.detach())


class Agent():
    def __init__(self, args):
        self.args = args

        self.theta = nn.Parameter(torch.zeros(5, requires_grad=True))
        self.values = nn.Parameter(torch.zeros(5, requires_grad=True))

        self.theta_optimizer = torch.optim.Adam((self.theta,), lr=args.lr_out)
        self.value_optimizer = torch.optim.Adam((self.values,), lr=args.lr_v)

    def theta_update(self, objective):
        self.theta_optimizer.zero_grad()
        objective.backward(retain_graph=True)
        self.theta_optimizer.step()

    def value_update(self, loss):
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

    def in_lookahead(self, other_theta, other_values):
        (s1, s2), _ = ipd.reset()
        other_memory = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = self.act(s1, self.theta, self.values)
            a2, lp2, v2 = self.act(s2, other_theta, other_values)
            (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        other_objective = other_memory.dice_objective()
        grad = self.get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta, other_values):
        (s1, s2), _ = ipd.reset()
        memory = Memory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = self.act(s1, self.theta, self.values)
            a2, lp2, v2 = self.act(s2, other_theta, other_values)
            (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
            memory.add(lp1, lp2, v1, torch.from_numpy(r1).float())

        # update self theta
        objective = memory.dice_objective()
        self.theta_update(objective)
        # update self value:
        v_loss = memory.value_loss()
        self.value_update(v_loss)

    def act(self, batch_states, theta, values):
        batch_states = torch.from_numpy(batch_states).long()
        probs = torch.sigmoid(theta)[batch_states]
        m = Bernoulli(1 - probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions, values[batch_states]

    def get_gradient(self, objective, theta):
        # create differentiable gradient for 2nd orders:
        grad_objective = torch.autograd.grad(objective, (theta), create_graph=True)[0]
        return grad_objective


if __name__ == "__main__":
    colors = ['b', 'c', 'm', 'r']

    for i in range(4):
        torch.manual_seed(hp.seed)
        scores = train(Agent(hp), Agent(hp), i, hp, ipd)
        plt.plot(scores, colors[i], label=str(i) + " lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()
