import torch
import numpy as np


def evaluate(agent1, agent2, ipd, args):
    # just to evaluate progress:
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0

    theta1, values1 = agent1.theta, agent1.values
    theta2, values2 = agent2.theta, agent2.values
    for t in range(args.ep_max_timesteps):
        a1, lp1, v1 = agent1.act(s1, theta1, values1)
        a2, lp2, v2 = agent2.act(s2, theta2, values2)
        (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1) / float(args.ep_max_timesteps)
        score2 += np.mean(r2) / float(args.ep_max_timesteps)
    return (score1, score2)


def train(agent1, agent2, n_lookaheads, args, ipd):
    joint_scores = []
    print("start iterations with", n_lookaheads, "lookaheads:")

    for update in range(200):
        # copy other's parameters:
        theta1_ = torch.tensor(agent1.theta.detach(), requires_grad=True)
        values1_ = torch.tensor(agent1.values.detach(), requires_grad=True)
        theta2_ = torch.tensor(agent2.theta.detach(), requires_grad=True)
        values2_ = torch.tensor(agent2.values.detach(), requires_grad=True)

        for k in range(n_lookaheads):
            # estimate other's gradients from in_lookahead:
            grad2 = agent1.in_lookahead(theta2_, values2_)
            grad1 = agent2.in_lookahead(theta1_, values1_)
            # update other's theta
            theta2_ = theta2_ - args.actor_lr_inner * grad2
            theta1_ = theta1_ - args.actor_lr_inner * grad1

        # update own parameters from out_lookahead:
        agent1.out_lookahead(theta2_, values2_)
        agent2.out_lookahead(theta1_, values1_)

        # evaluate progress:
        score = evaluate(agent1, agent2, ipd, args)
        joint_scores.append(0.5 * (score[0] + score[1]))

        # print
        if update % 10 == 0:
            p1 = [p.item() for p in torch.sigmoid(agent1.theta)]
            p2 = [p.item() for p in torch.sigmoid(agent2.theta)]
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))

    return joint_scores
