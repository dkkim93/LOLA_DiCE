import torch
from dice.agent_base import AgentBase


class Agent(AgentBase):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, name=name, i_agent=i_agent)

        self._set_dim()
        self.set_policy()

    def in_lookahead(self, memory, iteration, logging=True):
        # Sample experiences from memory
        _, logprob, opponent_logprob, value, reward = memory.sample(self.i_agent)

        # Get actor grad and update
        actor_loss = self.get_dice_loss(logprob, opponent_logprob, value, reward, inner=True)
        actor_grad = torch.autograd.grad(actor_loss, (self.actor), create_graph=True)[0]
        phi = self.actor - self.args.actor_lr_inner * actor_grad

        # For logging
        if logging:
            self.tb_writer.add_scalars(
                "debug/inner_loss", {str(self.i_agent): actor_loss.data.numpy()}, iteration)

        return phi

    def in_lookahead_(self, memory, phi):
        # Sample experiences from memory
        obs, logprob, opponent_logprob, value, reward = memory.sample(self.i_agent)

        # Get actor grad and update
        actor_loss = self.get_dice_loss(logprob, opponent_logprob, value, reward, inner=True)

        # Get the importance sampling
        obs = torch.stack(obs, dim=1)
        _, logprob_theta, _ = self.act_(obs, self.actor, self.critic)
        logprob_phi = torch.stack(logprob, dim=1)
        sampling = torch.div(torch.exp(logprob_theta), torch.exp(logprob_phi)).mean().detach()
        actor_grad = torch.autograd.grad(actor_loss * sampling, (phi), create_graph=True)[0]
        new_phi = phi - self.args.actor_lr_inner * actor_grad

        return new_phi

    def out_lookahead(self, memory, iteration):
        # Sample experiences from memory
        _, logprob, opponent_logprob, value, reward = memory.sample(self.i_agent)

        # Update actor
        actor_loss = self.get_dice_loss(logprob, opponent_logprob, value, reward, inner=False)
        self._update(self.actor_optimizer, actor_loss, is_actor=True)

        # Update critic
        critic_loss = self.get_critic_loss(value, reward)
        self._update(self.critic_optimizer, critic_loss, is_actor=False)

        # For logging
        self.tb_writer.add_scalars(
            "debug/outer_loss", {str(self.i_agent): actor_loss.data.numpy()}, iteration)
        self.tb_writer.add_scalars(
            "debug/critic_loss", {str(self.i_agent): critic_loss.data.numpy()}, iteration)

    def get_dice_loss(self, logprob, opponent_logprob, value, reward, inner):
        # Process data
        logprob = torch.stack(logprob, dim=1)
        opponent_logprob = torch.stack(opponent_logprob, dim=1)
        value = torch.stack(value, dim=1)
        reward = torch.stack(reward, dim=1)

        # Apply discount
        cum_discount = torch.cumprod(self.args.discount * torch.ones(*reward.size()), dim=1) / self.args.discount
        discounted_rewards = reward * cum_discount
        discounted_values = value * cum_discount

        # Stochastic nodes involved in reward dependencies
        if self.args.opponent_shaping or inner:
            dependencies = torch.cumsum(logprob + opponent_logprob, dim=1)
            stochastic_nodes = logprob + opponent_logprob
        else:
            dependencies = torch.cumsum(logprob, dim=1)
            stochastic_nodes = logprob

        # Get Dice loss
        dice_loss = torch.mean(torch.sum(self.magic_box(dependencies) * discounted_rewards, dim=1))

        if self.args.use_baseline:
            baseline_term = torch.mean(torch.sum((1 - self.magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_loss = dice_loss + baseline_term

        return -dice_loss

    def magic_box(self, x):
        return torch.exp(x - x.detach())

    def get_critic_loss(self, value, reward):
        value = torch.stack(value, dim=1)
        reward = torch.stack(reward, dim=1)
        return torch.mean(pow((reward - value), 2))
