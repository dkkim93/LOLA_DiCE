import os
import argparse
import torch
import random
import numpy as np
from trainer import train
from misc.utils import set_log, make_env
from dice.agent import Agent
from tensorboardX import SummaryWriter


def main(args):
    # Create directories
    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    if not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    # Set logs
    log = set_log(args)
    tb_writer = SummaryWriter('./logs/tb_{0}'.format(args.log_name))

    # Create env
    env = make_env(log, args)

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    # Initialize agents
    agent1 = Agent(env, log, tb_writer, args, name="agent1", i_agent=1)
    agent2 = Agent(env, log, tb_writer, args, name="agent2", i_agent=2)

    # Start train
    train(agent1, agent2, env, log, tb_writer, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    # Algorithm
    parser.add_argument(
        "--opponent-shaping", action="store_true", 
        help="If True, include opponent shaping in optimization")
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
        "env::%s_seed::%s_opponent_shaping::%s_batch_size::%s_actor_lr_inner::%s_actor_lr_outer::%s_" \
        "critic_lr::%s_prefix::%s_log" % (
            args.env_name, args.seed, args.opponent_shaping, args.batch_size, args.actor_lr_inner, args.actor_lr_outer,
            args.critic_lr, args.prefix)

    main(args=args)
