import torch
import numpy as np


class Memory():
    def __init__(self, args, n_task):
        self.args = args
        self.n_task = n_task

        self.storage = {}
        self.keywords = [
            "obs1", "obs2",
            "logprob1", "logprob2",
            "value1", "value2",
            "reward1", "reward2"]
        self.reset()
        self._set_discount_array()

    def reset(self):
        self.storage.clear()
        for i_task in range(self.n_task):
            self.storage[i_task] = {}
            for keyword in self.keywords:
                self.storage[i_task][keyword] = []

    def _set_discount_array(self):
        """Used to form the normalized discounted reward 
        (see LOLA (Foerester et al., 2017) for details)"""
        discount_array = []
        for timestep in range(self.args.ep_max_timesteps):
            discount_array.append(pow(self.args.discount, timestep))
        discount_array = np.expand_dims(np.asarray(discount_array), 0)
        self.discount_array = np.repeat(discount_array, repeats=self.args.batch_size, axis=0)

    def add(self, observations, logprobs, values, rewards, i_task):
        self.storage[i_task]["obs1"].append(torch.from_numpy(observations[0]).long())
        self.storage[i_task]["obs2"].append(torch.from_numpy(observations[1]).long())
        self.storage[i_task]["logprob1"].append(logprobs[0])
        self.storage[i_task]["logprob2"].append(logprobs[1])
        self.storage[i_task]["value1"].append(values[0])
        self.storage[i_task]["value2"].append(values[1])
        self.storage[i_task]["reward1"].append(torch.from_numpy(rewards[0]).float())
        self.storage[i_task]["reward2"].append(torch.from_numpy(rewards[1]).float())

    def sample(self, i_agent, i_task):
        if i_agent == 1:
            return \
                self.storage[i_task]["obs1"], \
                self.storage[i_task]["logprob1"], \
                self.storage[i_task]["logprob2"], \
                self.storage[i_task]["value1"], \
                self.storage[i_task]["reward1"]
        elif i_agent == 2:
            return \
                self.storage[i_task]["obs2"], \
                self.storage[i_task]["logprob2"], \
                self.storage[i_task]["logprob1"], \
                self.storage[i_task]["value2"], \
                self.storage[i_task]["reward2"]
        else:
            raise ValueError()

    def get_average_reward(self, i_task):
        reward1 = torch.stack(self.storage[i_task]["reward1"], dim=1).numpy()
        reward1 = self.discount_array * reward1
        reward1 = (1. - self.args.discount) * np.sum(reward1, axis=1)
        reward1 = np.mean(reward1)

        reward2 = torch.stack(self.storage[i_task]["reward2"], dim=1).numpy()
        reward2 = self.discount_array * reward2
        reward2 = (1. - self.args.discount) * np.sum(reward2, axis=1)
        reward2 = np.mean(reward2)

        return reward1, reward2
