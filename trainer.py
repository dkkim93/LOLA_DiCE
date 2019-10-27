import torch
import numpy as np


def evaluate(agent1, agent2, env, args):
    actor1, critic1 = agent1.actor, agent1.critic
    actor2, critic2 = agent2.actor, agent2.critic

    score1, score2 = 0., 0.
    obs1, obs2 = env.reset()
    for timestep in range(args.ep_max_timesteps):
        # Get actions
        action1, logprop1, value1 = agent1.act(obs1, actor1, critic1)
        action2, logprop2, value2 = agent2.act(obs2, actor2, critic2)
        
        # Take step in the environment
        (next_obs1, next_obs2), (reward1, reward2), _, _ = env.step((action1, action2))

        # For next timestep
        obs1, obs2 = next_obs1, next_obs2

        # Accumulate reward
        score1 += np.mean(reward1) / float(args.ep_max_timesteps)
        score2 += np.mean(reward2) / float(args.ep_max_timesteps)

    return score1, score2


def train(agent1, agent2, env, log, tb_writer, args):
    for iteration in range(200):
        # Copy other agent's parameters
        actor2_ = torch.tensor(agent2.actor.detach(), requires_grad=True)
        critic2_ = torch.tensor(agent2.critic.detach(), requires_grad=True)

        actor1_ = torch.tensor(agent1.actor.detach(), requires_grad=True)
        critic1_ = torch.tensor(agent1.critic.detach(), requires_grad=True)

        # Perform inner-loop update
        for _ in range(args.n_lookahead):
            actor2_ = agent1.in_lookahead(
                opponent_actor=actor2_, 
                opponent_critic=critic2_)
            actor1_ = agent2.in_lookahead(
                opponent_actor=actor1_, 
                opponent_critic=critic1_)

        # Perform outer-loop update
        agent1.out_lookahead(actor2_, critic2_)
        agent2.out_lookahead(actor1_, critic1_)

        # Evaluate progress
        score1, score2 = evaluate(agent1, agent2, env, args)

        # Log performance
        if iteration % 10 == 0:
            log[args.log_name].info("At iteration {}, returns: {:.3f}, {:.3f}".format(
                iteration, score1, score2))

            # # Tit-for-tat
            # TODO
            # prob1 = [p.item() for p in torch.sigmoid(agent1.actor)]
            # prob2 = [p.item() for p in torch.sigmoid(agent2.actor)]

            # print('policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (prob1[0], prob1[1], prob1[2], prob1[3], prob1[4]))
            # print('(agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (prob2[0], prob2[1], prob2[2], prob2[3], prob2[4]))
