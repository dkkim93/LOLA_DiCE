import torch
from misc.sampler import Sampler

total_iteration, total_test_iteration = 0, 0


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


def evaluate(agent1, agent2, sampler, args, iteration):
    # Sample inner-loop trajectories 
    memory_theta = sampler.collect_trajectories(
        agent1, agent2, use_theta=True, n_task=1)

    if iteration == 0:
        return memory_theta.get_average_reward(i_task=0)
    else:
        # Perform inner-loop update
        phis1 = agent1.in_lookahead(memory_theta, total_iteration, log_result=False)
        phis2 = agent2.in_lookahead(memory_theta, total_iteration, log_result=False)

        # Sample outer-loop trajectories 
        memory_phi = sampler.collect_trajectories(
            agent1, agent2, use_theta=False, n_task=1, phis=[phis1, phis2])

        return memory_phi.get_average_reward(i_task=0)


def meta_test(agent1, agent2, log, tb_writer, args):
    global total_test_iteration

    sampler = Sampler(log=log, args=args)
    phi1 = torch.tensor(agent1.theta.detach(), requires_grad=True)
    phi2 = torch.tensor(agent2.theta.detach(), requires_grad=True)

    for iteration in range(args.n_iteration):
        # Sample inner-loop trajectories
        memory_theta = sampler.collect_trajectories(
            agent1, agent2, use_theta=False, n_task=1, 
            phis=[[phi1], [phi2]])

        # Perform inner-loop update
        # but with importance sampling for meta-test
        new_phi1 = agent1.in_lookahead_with_importance(memory_theta, phi1, total_test_iteration)
        new_phi2 = agent2.in_lookahead_with_importance(memory_theta, phi2, total_test_iteration)
        
        phi1 = torch.tensor(new_phi1.detach(), requires_grad=True)
        phi2 = torch.tensor(new_phi2.detach(), requires_grad=True)

        # Log performance
        if iteration % 10 == 0:
            score1, score2 = memory_theta.get_average_reward(i_task=0)
            log[args.log_name].info("[META-TEST] At iteration {}, returns: {:.3f}, {:.3f}".format(
                total_test_iteration, score1, score2))
            tb_writer.add_scalars("meta_test/train_reward", {"agent1": score1}, total_test_iteration)
            tb_writer.add_scalars("meta_test/train_reward", {"agent2": score2}, total_test_iteration)

        # For next iteration
        total_test_iteration += 1


def meta_train(agent1, agent2, log, tb_writer, args):
    global total_iteration

    while True:
        # Initialize sampler
        sampler = Sampler(log=log, args=args)

        for iteration in range(args.n_iteration):
            # Sample inner-loop trajectories 
            memory_theta = sampler.collect_trajectories(
                agent1, agent2, use_theta=True, n_task=args.n_task)

            # Perform inner-loop update
            phis1 = agent1.in_lookahead(memory_theta, total_iteration, log_result=True)
            phis2 = agent2.in_lookahead(memory_theta, total_iteration, log_result=True)

            # Sample outer-loop trajectories 
            memory_phi = sampler.collect_trajectories(
                agent1, agent2, use_theta=False, n_task=args.n_task, phis=[phis1, phis2])

            # Perform outer-loop update
            agent1.out_lookahead(memory_phi, total_iteration)
            agent2.out_lookahead(memory_phi, total_iteration)

            # Log performance
            if iteration % 10 == 0:
                score1, score2 = evaluate(agent1, agent2, sampler, args, iteration)
                log[args.log_name].info("At iteration {}, returns: {:.3f}, {:.3f}".format(
                    iteration, score1, score2))
                tb_writer.add_scalars("train_reward", {"agent1": score1}, total_iteration)
                tb_writer.add_scalars("train_reward", {"agent2": score2}, total_iteration)

            # For next iteration
            total_iteration += 1
        
        # Perform meta-test 
        meta_test(agent1, agent2, log, tb_writer, args)

        if total_iteration > 1000:
            import sys
            sys.exit()
