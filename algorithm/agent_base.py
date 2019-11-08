import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class AgentBase(object):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(AgentBase, self).__init__()

        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name
        self.i_agent = i_agent

    def _set_dim(self):
        self.actor_input_dim = self.env.state_space[0].n
        self.critic_input_dim = self.actor_input_dim

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))

    def set_policy(self):
        self.log[self.args.log_name].info("[{}] Set policy".format(self.name))

        self.theta = nn.Parameter(torch.zeros(self.actor_input_dim, requires_grad=True))
        self.critic = nn.Parameter(torch.zeros(self.critic_input_dim, requires_grad=True))

        self.actor_optimizer = torch.optim.Adam((self.theta,), lr=self.args.actor_lr_outer)
        self.critic_optimizer = torch.optim.Adam((self.critic,), lr=self.args.critic_lr)

    def _update(self, optimizer, loss, is_actor):
        optimizer.zero_grad()
        if is_actor:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        optimizer.step()

    def act_(self, obs, actor, critic):
        prob = torch.sigmoid(actor)[obs]
        bernoulli = Bernoulli(1 - prob)
        action = bernoulli.sample()
        logprob = bernoulli.log_prob(action)

        return action.numpy().astype(int), logprob, critic[obs]

    def get_action_prob(self, obs):
        raise ValueError("used?")
        obs = torch.from_numpy(obs).long()
        prob = torch.sigmoid(self.theta)[obs]
        cooperate_prob = 1. - prob

        return cooperate_prob.data.numpy().flatten()[0]

    @staticmethod
    def act(obs, actor, critic):
        obs = torch.from_numpy(obs).long()
        prob = torch.sigmoid(actor)[obs]
        bernoulli = Bernoulli(1 - prob)
        action = bernoulli.sample()
        logprob = bernoulli.log_prob(action)

        return action.numpy().astype(int), logprob, critic[obs]
