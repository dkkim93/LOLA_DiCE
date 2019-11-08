import numpy as np
from misc.memory import Memory
from algorithm.agent_base import AgentBase
from misc.utils import make_env


class Sampler(object):
    """Multithread sampler to collect trajectories
    Ref: https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/sampler.py
    """
    def __init__(self, log, args):
        self.log = log
        self.args = args
        
        self._set_envs()

    def _set_envs(self):
        self.envs = []
        for i_task in range(self.args.n_task):
            env = make_env(self.log, self.args)
            self.envs.append(env)

    def collect_trajectories(self, agent1, agent2, use_theta, n_task, phis=None):
        memory = Memory(self.args, n_task)

        for i_task in range(n_task):
            obs1, obs2 = self.envs[i_task].reset()
            done = False
            while done is False:
                # Select policies
                if use_theta:
                    actor1, actor2 = agent1.theta, agent2.theta
                else:
                    actor1, actor2 = phis[0][i_task], phis[1][i_task]

                # Select actions
                action1, logprob1, value1 = AgentBase.act(obs1, actor=actor1, critic=agent1.critic)
                action2, logprob2, value2 = AgentBase.act(obs2, actor=actor2, critic=agent2.critic)
                actions = np.concatenate([
                    np.expand_dims(action1, axis=0), 
                    np.expand_dims(action2, axis=0)], 
                    axis=0)

                # Take actions in env
                (next_obs1, next_obs2), (reward1, reward2), done, _ = self.envs[i_task].step(actions)

                # Save to memory
                memory.add(
                    observations=(obs1, obs2),
                    logprobs=(logprob1, logprob2),
                    values=(value1, value2),
                    rewards=(reward1, reward2),
                    i_task=i_task)

                # For next timestep
                obs1, obs2 = next_obs1, next_obs2

        return memory
