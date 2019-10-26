import argparse
import torch
import matplotlib.pyplot as plt
from envs import IPD
from trainer import train
from policy.agent import Agent


def main(args):
    colors = ['b', 'c', 'm', 'r']
    env = IPD(args.ep_max_timesteps, args.batch_size)

    for i in range(4):
        torch.manual_seed(args.seed)
        scores = train(
            Agent(env, args), 
            Agent(env, args), 
            i, 
            args, 
            env)
        plt.plot(scores, colors[i], label=str(i) + " lookaheads")

    plt.legend()
    plt.xlabel('rollouts', fontsize=20)
    plt.ylabel('joint score', fontsize=20)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--n-agent", type=int, default=2, 
        help="Number of agent")
    parser.add_argument(
        "--algorithm", type=str, choices=["standard", "dice"],
        default="dice", help="Learning algorithm to train agent")
    parser.add_argument(
        "--batch-size", type=int, default=128, 
        help="Batch size for both actor and critic")
    parser.add_argument(
        "--actor-lr_inner", type=float, default=0.3, 
        help="Learning rate for actor (inner loop)")
    parser.add_argument(
        "--actor-lr_outer", type=float, default=0.2, 
        help="Learning rate for actor (outer loop)")
    parser.add_argument(
        "--critic-lr", type=float, default=0.1, 
        help="Learning rate for critic")
    parser.add_argument(
        "--discount", type=float, default=0.96, 
        help="Discount factor")
    parser.add_argument(
        "--use-baseline", type=bool, default=True, 
        help="Use baseline or not")

    # Env
    parser.add_argument(
        "--env-name", type=str, default="",
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep-max-timesteps", type=int, default=150,
        help="Episode is terminated when max timestep is reached")
    parser.add_argument(
        "--n-action", type=int, default=2,
        help="# of possible actions")

    # Misc
    parser.add_argument(
        "--prefix", type=str, default="", 
        help="Prefix for tb_writer and logging")
    parser.add_argument(
        "--seed", type=int, default=42, 
        help="Sets Gym, PyTorch and Numpy seeds")

    args = parser.parse_args()

    # Set log name
    args.log_name = \
        "env::%s_seed::%s_algorithm:%s_batch_size::%s_actor_lr_inner::%s_actor_lr_outer::%s_" \
        "critic_lr::%s_prefix::%s_log" % (
            args.env_name, args.seed, args.algorithm, args.batch_size, args.actor_lr_inner, args.actor_lr_outer,
            args.critic_lr, args.prefix)

    main(args=args)
