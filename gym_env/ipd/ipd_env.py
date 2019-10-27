import gym
import numpy as np
from gym.spaces import Discrete, Tuple


class IPDEnv(gym.Env):
    """Two-agent environment for the Prisoner's Dilemma game.
    Possible actions for each agent are (C)ooperate and (D)efect.
    Ref: https://github.com/alexis-jacq/LOLA_DiCE/blob/master/envs/prisoners_dilemma.py
    """
    def __init__(self, args):
        super(IPDEnv, self).__init__()
        self.args = args

        self.state_space = [Discrete(5), Discrete(5)]
        self.action_space = Tuple([Discrete(2), Discrete(2)])
        self._set_payout_matrix()
        self._set_states()

    def _set_payout_matrix(self):
        """Payout matrix from agent 0's perspective.
        By default, the payout matrix for agent 1 is 
        the transpose of agent 0's matrix"""
        self.payout_matrix = np.array([
            [-2., 0.],
            [-3., -1.]], dtype=np.float32)

    def _set_states(self):
        self.states = np.array([
            [1, 2],
            [3, 4]], dtype=np.int64)

        self.states_dict = {}
        self.states_dict["S0"] = [np.array((0,), dtype=np.int64), np.array((0,), dtype=np.int64)] 
        self.states_dict["DD"] = [np.array((1,), dtype=np.int64), np.array((1,), dtype=np.int64)]
        self.states_dict["DC"] = [np.array((2,), dtype=np.int64), np.array((3,), dtype=np.int64)] 
        self.states_dict["CD"] = [np.array((3,), dtype=np.int64), np.array((2,), dtype=np.int64)] 
        self.states_dict["CC"] = [np.array((4,), dtype=np.int64), np.array((4,), dtype=np.int64)]  

    def reset(self):
        observations = [
            np.zeros(self.args.batch_size), 
            np.zeros(self.args.batch_size)]
        return observations

    def step(self, action):
        assert len(action) == 2, "Only two agents are supported in this domain"
        ac0, ac1 = action
        r0 = self.payout_matrix[ac0, ac1]
        r1 = self.payout_matrix[ac1, ac0]
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observations = [s0, s1]
        rewards = [r0, r1]
        done = False
        return observations, rewards, done, {}

    def render(self, mode='human', close=False):
        raise NotImplementedError()
