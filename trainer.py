from misc.sampler import Sampler

total_iteration = 0


def tit_for_tat(agent1, agent2, env, log, tb_writer, args, iteration):
    # Log tit-for-tat
    for key, value in env.states_dict.items():
        # Agent 1
        cooperate_prob = agent1.get_action_prob(value[0])
        log[args.log_name].info(
            "Agent1 at {}: Cooperate prob {:.2f} at iteration {}".format(key, cooperate_prob, iteration))
        tb_writer.add_scalars("debug/" + key, {"agent1": cooperate_prob}, iteration)

        # Agent 2
        cooperate_prob = agent2.get_action_prob(value[1])
        log[args.log_name].info(
            "Agent2 at {}: Cooperate prob {:.2f} at iteration {}".format(key, cooperate_prob, iteration))
        tb_writer.add_scalars("debug/" + key, {"agent2": cooperate_prob}, iteration)


def evaluate(agent1, agent2, sampler, args):
    # Sample inner-loop trajectories 
    memory_theta = sampler.collect_trajectories(
        agent1, agent2, is_inner=True, n_task=1)

    # Perform inner-loop update
    phis1 = agent1.in_lookahead(memory_theta, total_iteration, log=False)
    phis2 = agent2.in_lookahead(memory_theta, total_iteration, log=False)

    # Sample outer-loop trajectories 
    memory_phi = sampler.collect_trajectories(
        agent1, agent2, is_inner=False, n_task=1, phis=[phis1, phis2])

    return memory_phi.get_average_reward(i_task=0)

# def meta_test(agent1, agent2, env, args, log):
#     phi1 = torch.tensor(agent1.theta.detach(), requires_grad=True)
#     phi2 = torch.tensor(agent2.theta.detach(), requires_grad=True)
# 
#     for iteration in range(200):
#         # Perform inner-loop update
#         memory_theta = collect_trajectory(env, agent1, agent2, is_inner=False)
# 
#         new_phi1 = agent1.in_lookahead_(memory_theta, phi1)
#         new_phi2 = agent2.in_lookahead_(memory_theta, phi2)
#         
#         phi1 = torch.tensor(new_phi1.detach(), requires_grad=True)
#         phi2 = torch.tensor(new_phi2.detach(), requires_grad=True)
# 
#         # Measure performance
#         score1, score2 = evaluate(agent1, agent2, env, args, iteration)
# 
#         # Log performance
#         log[args.log_name].info("[META-TEST] At iteration {}, returns: {:.3f}, {:.3f}".format(
#             iteration, score1, score2))


# def collect_trajectory(actor1, actor2, critic1, critic2, act, env, args):
#     memory = Memory(args)
# 
#     obs1, obs2 = env.reset()
#     for timestep in range(args.ep_max_timesteps):
#         # Get actions
#         action1, logprob1, value1 = act(obs1, actor1, critic1)
#         action2, logprob2, value2 = act(obs2, actor2, critic2)
# 
#         # Take step in the environment
#         (next_obs1, next_obs2), (reward1, reward2), _, _ = env.step((action1, action2))
# 
#         # Add to memory
#         memory.add(
#             obs1=torch.from_numpy(obs1).long(), 
#             obs2=torch.from_numpy(obs2).long(),
#             logprob1=logprob1, 
#             logprob2=logprob2, 
#             value1=value1, 
#             value2=value2, 
#             reward1=torch.from_numpy(reward1).float(),
#             reward2=torch.from_numpy(reward2).float())
# 
#         # For next timestep
#         obs1, obs2 = next_obs1, next_obs2
# 
#     return memory


def train(agent1, agent2, env, log, tb_writer, args):
    global total_iteration

    while True:
        # Initialize sampler
        sampler = Sampler(log=log, args=args)

        for iteration in range(200):
            # Sample inner-loop trajectories 
            memory_theta = sampler.collect_trajectories(
                agent1, agent2, is_inner=True, n_task=args.n_task)

            # Perform inner-loop update
            phis1 = agent1.in_lookahead(memory_theta, total_iteration, log=True)
            phis2 = agent2.in_lookahead(memory_theta, total_iteration, log=True)

            # Sample outer-loop trajectories 
            memory_phi = sampler.collect_trajectories(
                agent1, agent2, is_inner=False, n_task=args.n_task, phis=[phis1, phis2])

            # Perform outer-loop update
            agent1.out_lookahead(memory_phi, total_iteration, log=True)
            agent2.out_lookahead(memory_phi, total_iteration, log=True)

            # Log performance
            if iteration % 10 == 0:
                score1, score2 = evaluate(agent1, agent2, sampler, args)
                log[args.log_name].info("At iteration {}, returns: {:.3f}, {:.3f}".format(
                    iteration, score1, score2))
                tb_writer.add_scalars("train_reward", {"agent1": score1}, total_iteration)
                tb_writer.add_scalars("train_reward", {"agent2": score2}, total_iteration)

            # For next iteration
            total_iteration += 1

        # meta_test(agent1, agent2, env, args, log)
