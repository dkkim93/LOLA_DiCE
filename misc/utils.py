import gym
import logging


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    return log


def make_env(args):
    import gym_env  # noqa
    env = gym.make(args.env_name, args=args)
    env._max_episode_steps = args.ep_max_timesteps

    return env


def make_env_fn(args):
    return make_env


def set_policy(env, tb_writer, log, args, name, i_agent):
    if name == "agent":
        from policy.agent import Agent
        policy = Agent(
            env=env, tb_writer=tb_writer, log=log, args=args, name=name + str(i_agent), i_agent=i_agent)
    else:
        raise ValueError("Invalid name")

    return policy
