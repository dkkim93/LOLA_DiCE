# coding: utf-8
import torch
import matplotlib.pyplot as plt
from envs import IPD
from trainer import train
from policy.agent import Agent


class Hp():
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.3
        self.lr_v = 0.1
        self.gamma = 0.96
        self.n_update = 200
        self.len_rollout = 150
        self.batch_size = 128
        self.use_baseline = True
        self.seed = 42


if __name__ == "__main__":
    colors = ['b', 'c', 'm', 'r']
    hp = Hp()
    env = IPD(hp.len_rollout, hp.batch_size)

    for i in range(4):
        torch.manual_seed(hp.seed)
        scores = train(
            Agent(env, hp), 
            Agent(env, hp), 
            i, 
            hp, 
            env)
        plt.plot(scores, colors[i], label=str(i) + " lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()
