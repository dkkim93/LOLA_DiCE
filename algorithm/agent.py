import torch
from algorithm.agent_base import AgentBase


class Agent(AgentBase):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, name=name, i_agent=i_agent)

        self._set_dim()
        self.set_policy()

    def in_lookahead(self, memory, total_iteration, log):
        phis = []
        for i_task in range(memory.n_task):
            # Sample experiences from memory
            _, logprob, opponent_logprob, value, reward = memory.sample(self.i_agent, i_task)

            # Get actor grad and update
            actor_loss = self.get_dice_loss(logprob, opponent_logprob, value, reward)
            actor_grad = torch.autograd.grad(actor_loss, (self.theta), create_graph=True)[0]
            phi = self.theta - self.args.actor_lr_inner * actor_grad

            # For logging
            if i_task == 0 and log:
                actor_loss_numpy = actor_loss.cpu().detach().numpy().flatten()[0]
                self.log[self.args.log_name].info(
                    "Inner loop loss {:.2f} for agent {} at {}".format(
                        actor_loss_numpy, self.i_agent, total_iteration))
                self.tb_writer.add_scalars("debug/inner_loop", {str(self.i_agent): actor_loss_numpy}, total_iteration)
    
            phis.append(phi)

        return phis

    def out_lookahead(self, memory, total_iteration, log):
        actor_loss, critic_loss = 0, 0
        for i_task in range(memory.n_task):
            # Sample experiences from memory
            _, logprob, opponent_logprob, value, reward = memory.sample(self.i_agent, i_task)

            # Get actor loss
            actor_loss += self.get_dice_loss(logprob, opponent_logprob, value, reward)

            # Get critic loss
            critic_loss += self.get_critic_loss(value, reward)

        # Normalize loss
        actor_loss /= float(self.args.n_task)
        critic_loss /= float(self.args.n_task)

        # Perform update
        self._update(self.actor_optimizer, actor_loss, is_actor=True)
        self._update(self.critic_optimizer, critic_loss, is_actor=False)

        # For logging
        if log:
            actor_loss_numpy = actor_loss.cpu().detach().numpy().flatten()[0]
            critic_loss_numpy = critic_loss.cpu().detach().numpy().flatten()[0]
            
            self.log[self.args.log_name].info(
                "Outer loop actor loss {:.2f} for agent {} at {}".format(
                    actor_loss_numpy, self.i_agent, total_iteration))
            self.log[self.args.log_name].info(
                "Outer loop critic loss {:.2f} for agent {} at {}".format(
                    critic_loss_numpy, self.i_agent, total_iteration))
            self.tb_writer.add_scalars("debug/outer_loop_actor", {str(self.i_agent): actor_loss_numpy}, total_iteration)
            self.tb_writer.add_scalars("debug/outer_loop_critic", {str(self.i_agent): critic_loss_numpy}, total_iteration)

    # def in_lookahead_with_importance(self, memory, phi):
    #     # Sample experiences from memory
    #     obs, logprob, opponent_logprob, value, reward = memory.sample(self.i_agent)

    #     # Get actor grad and update
    #     actor_loss = self.get_dice_loss(logprob, opponent_logprob, value, reward)

    #     # Get the importance sampling
    #     obs = torch.stack(obs, dim=1)
    #     _, logprob_theta, _ = self.act_(obs, self.actor, self.critic)
    #     logprob_phi = torch.stack(logprob, dim=1)
    #     sampling = torch.div(torch.exp(logprob_theta), torch.exp(logprob_phi)).mean().detach()
    #     actor_grad = torch.autograd.grad(actor_loss * sampling, (phi), create_graph=True)[0]
    #     new_phi = phi - self.args.actor_lr_inner * actor_grad

    #     return new_phi

    def get_dice_loss(self, logprob, opponent_logprob, value, reward):
        logprob = torch.stack(logprob, dim=1)
        opponent_logprob = torch.stack(opponent_logprob, dim=1)
        value = torch.stack(value, dim=1)
        reward = torch.stack(reward, dim=1)

        # Apply discount
        cum_discount = torch.cumprod(self.args.discount * torch.ones(*reward.size()), dim=1) / self.args.discount
        discounted_rewards = reward * cum_discount
        discounted_values = value * cum_discount

        # Stochastic nodes involved in reward dependencies
        dependencies = torch.cumsum(logprob + opponent_logprob, dim=1)
        # dependencies = torch.cumsum(logprob, dim=1)  # TODO Use flag

        # Logprob of each stochastic nodes
        stochastic_nodes = logprob + opponent_logprob
        # stochastic_nodes = logprob  # TODO Use flag

        # Dice loss
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
