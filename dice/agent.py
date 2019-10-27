import torch
from misc.replay_memory import ReplayMemory
from dice.agent_base import AgentBase


class Agent(AgentBase):
    def __init__(self, env, log, tb_writer, args, name, i_agent):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, name=name, i_agent=i_agent)

        self._set_dim()
        self._set_policy()

    def in_lookahead(self, opponent_actor, opponent_critic):
        """Perform inner update
        Note that from agent 1's perspective, its opponent is agent 2.
        Similarly, from agent 2's perspective, its opponent is agent 1.
        """
        opponent_memory = ReplayMemory(self.args)

        obs, opponent_obs = self.env.reset()
        for timestep in range(self.args.ep_max_timesteps):
            # Get actions
            action, logprob, value = self.act(obs, self.actor, self.critic)
            opponent_action, opponent_logprob, opponent_value = self.act(
                opponent_obs, opponent_actor, opponent_critic)

            # Take step in the environment
            (next_obs, next_opponent_obs), (reward, opponent_reward), _, _ = \
                self.env.step((action, opponent_action))

            # Add to memory
            # Note that this is memory from opponent's perspective
            # thus opponent's opponent is myself
            opponent_memory.add(
                logprob=opponent_logprob, 
                opponent_logprob=logprob, 
                value=opponent_value, 
                reward=torch.from_numpy(opponent_reward).float())

            # For next timestep
            obs, opponent_obs = next_obs, next_opponent_obs

        # Sample opponent experiences from memory
        opponent_logprob, logprob, opponent_value, opponent_reward = opponent_memory.sample()

        # Get actor grad and update
        opponent_actor_loss = self.get_dice_loss(opponent_logprob, logprob, opponent_value, opponent_reward)
        actor_grad = torch.autograd.grad(opponent_actor_loss, (opponent_actor), create_graph=True)[0]
        opponent_actor = opponent_actor - self.args.actor_lr_inner * actor_grad

        return opponent_actor

    def out_lookahead(self, opponent_actor, opponent_critic):
        memory = ReplayMemory(self.args)

        obs, opponent_obs = self.env.reset()
        for timestep in range(self.args.ep_max_timesteps):
            # Get actions
            action, logprob, value = self.act(obs, self.actor, self.critic)
            opponent_action, opponent_logprob, opponent_value = self.act(
                opponent_obs, opponent_actor, opponent_critic)

            # Take step in the environment
            (next_obs, next_opponent_obs), (reward, opponent_reward), _, _ = \
                self.env.step((action, opponent_action))

            # Add to memory
            memory.add(
                logprob=logprob, 
                opponent_logprob=opponent_logprob, 
                value=value, 
                reward=torch.from_numpy(reward).float())

            # For next timestep
            obs, opponent_obs = next_obs, next_opponent_obs

        # Sample experiences from memory
        logprob, opponent_logprob, value, reward = memory.sample()

        # Update actor
        actor_loss = self.get_dice_loss(logprob, opponent_logprob, value, reward)
        self._update(self.actor_optimizer, actor_loss, is_actor=True)

        # Update critic
        critic_loss = self.get_critic_loss(value, reward)
        self._update(self.critic_optimizer, critic_loss, is_actor=False)

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

        # Logprob of each stochastic nodes
        stochastic_nodes = logprob + opponent_logprob

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
