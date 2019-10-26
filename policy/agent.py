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

        other_objective = other_memory.dice_objective()
        grad = self.get_gradient(other_objective, other_actor)
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

        # Update actor
        objective = memory.dice_objective()
        self.actor_update(objective)

        # Update critic
        critic_loss = memory.critic_loss()
        self.critic_update(critic_loss)

    def act(self, batch_states, actor, critic):
        batch_states = torch.from_numpy(batch_states).long()
        probs = torch.sigmoid(actor)[batch_states]
        m = Bernoulli(1 - probs)
        actions = m.sample()
        logprobs = m.log_prob(actions)

        return actions.numpy().astype(int), logprobs, critic[batch_states]

    def get_gradient(self, objective, actor):
        # create differentiable gradient for 2nd orders:
        grad_objective = torch.autograd.grad(objective, (actor), create_graph=True)[0]
        return grad_objective
