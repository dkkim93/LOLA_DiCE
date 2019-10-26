import torch
import numpy as np


def evaluate(agent1, agent2, ipd, args):
    # just to evaluate progress:
    (s1, s2), _ = ipd.reset()
    score1 = 0
    score2 = 0

    actor1, critic1 = agent1.actor, agent1.critic
    actor2, critic2 = agent2.actor, agent2.critic
    for t in range(args.ep_max_timesteps):
        a1, lp1, v1 = agent1.act(s1, actor1, critic1)
        a2, lp2, v2 = agent2.act(s2, actor2, critic2)
        (s1, s2), (r1, r2), _, _ = ipd.step((a1, a2))
        # cumulate scores
        score1 += np.mean(r1) / float(args.ep_max_timesteps)
        score2 += np.mean(r2) / float(args.ep_max_timesteps)
    return (score1, score2)


def train(agent1, agent2, args, ipd):
    joint_scores = []
    print("start iterations with", args.n_lookahead, "lookaheads:")

    for update in range(200):
        # copy other's parameters:
        actor1_ = torch.tensor(agent1.actor.detach(), requires_grad=True)
        critic1_ = torch.tensor(agent1.critic.detach(), requires_grad=True)
        actor2_ = torch.tensor(agent2.actor.detach(), requires_grad=True)
        critic2_ = torch.tensor(agent2.critic.detach(), requires_grad=True)

        for k in range(args.n_lookahead):
            # estimate other's gradients from in_lookahead:
            grad2 = agent1.in_lookahead(actor2_, critic2_)
            grad1 = agent2.in_lookahead(actor1_, critic1_)

            # update other's actor
            actor2_ = actor2_ - args.actor_lr_inner * grad2
            actor1_ = actor1_ - args.actor_lr_inner * grad1

        # update own parameters from out_lookahead:
        agent1.out_lookahead(actor2_, critic2_)
        agent2.out_lookahead(actor1_, critic1_)

        # evaluate progress:
        score = evaluate(agent1, agent2, ipd, args)
        joint_scores.append(0.5 * (score[0] + score[1]))

        # print
        if update % 10 == 0:
            p1 = [p.item() for p in torch.sigmoid(agent1.actor)]
            p2 = [p.item() for p in torch.sigmoid(agent2.actor)]
            print('update', update, 'score (%.3f,%.3f)' % (score[0], score[1]) , 'policy (agent1) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p1[0], p1[1], p1[2], p1[3], p1[4]),' (agent2) = {%.3f, %.3f, %.3f, %.3f, %.3f}' % (p2[0], p2[1], p2[2], p2[3], p2[4]))

    return joint_scores
