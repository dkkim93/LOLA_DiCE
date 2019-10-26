import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from misc.replay_memory import ReplayMemory


class Agent():
    def __init__(self, env, args):
        self.env = env
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
        (s1, s2), _ = self.env.reset()
        other_memory = ReplayMemory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = self.act(s1, self.theta, self.values)
            a2, lp2, v2 = self.act(s2, other_theta, other_values)
            (s1, s2), (r1, r2), _, _ = self.env.step((a1, a2))
            other_memory.add(lp2, lp1, v2, torch.from_numpy(r2).float())

        other_objective = other_memory.dice_objective()
        grad = self.get_gradient(other_objective, other_theta)
        return grad

    def out_lookahead(self, other_theta, other_values):
        (s1, s2), _ = self.env.reset()
        memory = ReplayMemory(self.args)
        for t in range(self.args.len_rollout):
            a1, lp1, v1 = self.act(s1, self.theta, self.values)
            a2, lp2, v2 = self.act(s2, other_theta, other_values)
            (s1, s2), (r1, r2), _, _ = self.env.step((a1, a2))
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
