import torch
import numpy as np
from misc.replay_memory import ReplayMemory

train_iteration = 0
test_iteration = 0


def tit_for_tat(agent1, agent2, env, log, tb_writer, args):
    # Log tit-for-tat
    for key, value in env.states_dict.items():
        # Agent 1
        cooperate_prob = agent1.get_action_prob(value[0])
        log[args.log_name].info(
            "Agent1 at {}: Cooperate prob {:.2f} at iteration {}".format(key, cooperate_prob, train_iteration))
        tb_writer.add_scalars("TFT/" + key, {"agent1": cooperate_prob}, train_iteration)

        # Agent 2
        cooperate_prob = agent2.get_action_prob(value[1])
        log[args.log_name].info(
            "Agent2 at {}: Cooperate prob {:.2f} at iteration {}".format(key, cooperate_prob, train_iteration))
        tb_writer.add_scalars("TFT/" + key, {"agent2": cooperate_prob}, train_iteration)


def evaluate(agent1, agent2, env, args, iteration):
    if iteration > 0:
        # Perform inner-loop update
        memory_theta = collect_trajectory(
            actor1=agent1.actor, actor2=agent2.actor,
            critic1=agent1.critic, critic2=agent2.critic,
            act=agent1.act, env=env, args=args)

        phi1 = agent1.in_lookahead(memory_theta, train_iteration, logging=False)
        phi2 = agent2.in_lookahead(memory_theta, train_iteration, logging=False)
    else:
        phi1 = agent1.actor
        phi2 = agent2.actor

    score1, score2 = 0., 0.
    obs1, obs2 = env.reset()
    for timestep in range(args.ep_max_timesteps):
        # Get actions
        action1, logprop1, value1 = agent1.act(obs1, phi1, agent1.critic)
        action2, logprop2, value2 = agent2.act(obs2, phi2, agent2.critic)
        
        # Take step in the environment
        (next_obs1, next_obs2), (reward1, reward2), _, _ = env.step((action1, action2))

        # For next timestep
        obs1, obs2 = next_obs1, next_obs2

        # Accumulate reward
        score1 += np.mean(reward1) / float(args.ep_max_timesteps)
        score2 += np.mean(reward2) / float(args.ep_max_timesteps)

    return score1, score2


def meta_test(agent1, agent2, env, args, log):
    global test_iteration
    phi1 = torch.tensor(agent1.actor.detach(), requires_grad=True)
    phi2 = torch.tensor(agent2.actor.detach(), requires_grad=True)

    for iteration in range(200):
        # Perform inner-loop update
        memory_theta = collect_trajectory(
            actor1=phi1, actor2=phi2,
            critic1=agent1.critic, critic2=agent2.critic,
            act=agent1.act, env=env, args=args)

        new_phi1 = agent1.in_lookahead_(memory_theta, phi1)
        new_phi2 = agent2.in_lookahead_(memory_theta, phi2)
        
        phi1 = torch.tensor(new_phi1.detach(), requires_grad=True)
        phi2 = torch.tensor(new_phi2.detach(), requires_grad=True)

        # Measure performance
        score1, score2 = evaluate(agent1, agent2, env, args, iteration)

        # Log performance
        log[args.log_name].info("[META-TEST] At iteration {}, returns: {:.3f}, {:.3f}".format(
            test_iteration, score1, score2))

        # For next iteration
        test_iteration += 1


def collect_trajectory(actor1, actor2, critic1, critic2, act, env, args):
    memory = ReplayMemory(args)

    obs1, obs2 = env.reset()
    for timestep in range(args.ep_max_timesteps):
        # Get actions
        action1, logprob1, value1 = act(obs1, actor1, critic1)
        action2, logprob2, value2 = act(obs2, actor2, critic2)

        # Take step in the environment
        (next_obs1, next_obs2), (reward1, reward2), _, _ = env.step((action1, action2))

        # Add to memory
        memory.add(
            obs1=torch.from_numpy(obs1).long(), 
            obs2=torch.from_numpy(obs2).long(),
            logprob1=logprob1, 
            logprob2=logprob2, 
            value1=value1, 
            value2=value2, 
            reward1=torch.from_numpy(reward1).float(),
            reward2=torch.from_numpy(reward2).float())

        # For next timestep
        obs1, obs2 = next_obs1, next_obs2

    return memory


def train(agent1, agent2, env, log, tb_writer, args):
    global train_iteration

    while True:
        for iteration in range(200):
            # Perform inner-loop update
            memory_theta = collect_trajectory(
                actor1=agent1.actor, actor2=agent2.actor,
                critic1=agent1.critic, critic2=agent2.critic,
                act=agent1.act, env=env, args=args)

            phi1 = agent1.in_lookahead(memory_theta, train_iteration)
            phi2 = agent2.in_lookahead(memory_theta, train_iteration)

            # Perform outer-loop update
            memory_phi = collect_trajectory(
                actor1=phi1, actor2=phi2,
                critic1=agent1.critic, critic2=agent2.critic,
                act=agent1.act, env=env, args=args)

            agent1.out_lookahead(memory_phi, train_iteration)
            agent2.out_lookahead(memory_phi, train_iteration)

            # Measure performance
            score1, score2 = evaluate(agent1, agent2, env, args, iteration)
            
            # Log performance
            if iteration % 10 == 0:
                log[args.log_name].info("At iteration {}, returns: {:.3f}, {:.3f}".format(
                    train_iteration, score1, score2))
                tb_writer.add_scalars("train_reward", {"agent1": score1}, iteration)
                tb_writer.add_scalars("train_reward", {"agent2": score2}, iteration)
                
                tit_for_tat(agent1, agent2, env, log, tb_writer, args)

            # For next iteration
            train_iteration += 1

        meta_test(agent1, agent2, env, args, log)
