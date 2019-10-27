import torch
import torch.nn as nn
from torch.distributions import Bernoulli
from misc.replay_memory import ReplayMemory


class Agent(object):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name
        self.i_agent = i_agent

        self._set_dim()
        self._set_policy()

    def _set_dim(self):
        self.actor_input_dim = self.env.state_space[0].n
        self.critic_input_dim = self.actor_input_dim

        self.log[self.args.log_name].info("[{}] Actor input dim: {}".format(
            self.name, self.actor_input_dim))
        self.log[self.args.log_name].info("[{}] Critic input dim: {}".format(
            self.name, self.critic_input_dim))

    def _set_policy(self):
        self.actor = nn.Parameter(torch.zeros(self.actor_input_dim, requires_grad=True))
        self.critic = nn.Parameter(torch.zeros(self.critic_input_dim, requires_grad=True))

        self.actor_optimizer = torch.optim.Adam((self.actor,), lr=self.args.actor_lr_outer)
        self.critic_optimizer = torch.optim.Adam((self.critic,), lr=self.args.critic_lr)

    def actor_update(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def critic_update(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

    def in_lookahead(self, other_actor, other_critic):
        other_memory = ReplayMemory(self.args)

        obs1, obs2 = self.env.reset()
        for timestep in range(self.args.ep_max_timesteps):
            # Get actions
            action1, logprob1, value1 = self.act(obs1, self.actor, self.critic)
            action2, logprob2, value2 = self.act(obs2, other_actor, other_critic)

            # Take step in the environment
            (next_obs1, next_obs2), (reward1, reward2), _, _ = self.env.step((action1, action2))

            # Add to memory
            other_memory.add(logprob2, logprob1, value2, torch.from_numpy(reward2).float())

            # For next timestep
            obs1, obs2 = next_obs1, next_obs2

        # Sample experiences from memory
        self_logprob, other_logprob, value, reward = other_memory.sample()
        other_actor_loss = self.get_dice_loss(self_logprob, other_logprob, value, reward)
        grad = torch.autograd.grad(other_actor_loss, (other_actor), create_graph=True)[0]
        return grad

    def out_lookahead(self, other_actor, other_critic):
        memory = ReplayMemory(self.args)

        obs1, obs2 = self.env.reset()
        for timestep in range(self.args.ep_max_timesteps):
            # Get actions
            action1, logprob1, value1 = self.act(obs1, self.actor, self.critic)
            action2, logprob2, value2 = self.act(obs2, other_actor, other_critic)

            # Take step in the environment
            (next_obs1, next_obs2), (reward1, reward2), _, _ = self.env.step((action1, action2))

            # Add to memory
            memory.add(logprob1, logprob2, value1, torch.from_numpy(reward1).float())

            # For next timestep
            obs1, obs2 = next_obs1, next_obs2

        # Sample experiences from memory
        self_logprob, other_logprob, value, reward = memory.sample()

        # Update actor
        actor_loss = self.get_dice_loss(self_logprob, other_logprob, value, reward)
        self.actor_update(actor_loss)

        # Update critic
        critic_loss = self.get_critic_loss(value, reward)
        self.critic_update(critic_loss)

    def act(self, obs, actor, critic):
        obs = torch.from_numpy(obs).long()
        prob = torch.sigmoid(actor)[obs]
        m = Bernoulli(1 - prob)
        action = m.sample()
        logprob = m.log_prob(action)

        return action.numpy().astype(int), logprob, critic[obs]

    def get_critic_loss(self, value, reward):
        value = torch.stack(value, dim=1)
        reward = torch.stack(reward, dim=1)
        return torch.mean(pow((reward - value), 2))

    def get_dice_loss(self, self_logprob, other_logprob, value, reward):
        self_logprob = torch.stack(self_logprob, dim=1)
        other_logprob = torch.stack(other_logprob, dim=1)
        value = torch.stack(value, dim=1)
        reward = torch.stack(reward, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(self.args.discount * torch.ones(*reward.size()), dim=1) / self.args.discount
        discounted_rewards = reward * cum_discount
        discounted_values = value * cum_discount

        # stochastics nodes involved in reward dependencies:
        dependencies = torch.cumsum(self_logprob + other_logprob, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprob + other_logprob

        # dice objective:
        dice_loss = torch.mean(torch.sum(self.magic_box(dependencies) * discounted_rewards, dim=1))

        if self.args.use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - self.magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_loss = dice_loss + baseline_term

        return -dice_loss

    def magic_box(self, x):
        return torch.exp(x - x.detach())
