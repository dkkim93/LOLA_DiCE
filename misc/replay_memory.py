class ReplayMemory():
    def __init__(self, args):
        self.args = args

        self.obs1 = []
        self.obs2 = []

        self.logprob1 = []
        self.logprob2 = []

        self.value1 = []
        self.value2 = []

        self.reward1 = []
        self.reward2 = []

    def add(self, obs1, obs2, logprob1, logprob2, value1, value2, reward1, reward2):
        self.obs1.append(obs1)
        self.obs2.append(obs2)

        self.logprob1.append(logprob1)
        self.logprob2.append(logprob2)

        self.value1.append(value1)
        self.value2.append(value2)

        self.reward1.append(reward1)
        self.reward2.append(reward2)

    def sample(self, i_agent):
        if i_agent == 1:
            return self.obs1, self.logprob1, self.logprob2, self.value1, self.reward1
        elif i_agent == 2:
            return self.obs2, self.logprob2, self.logprob1, self.value2, self.reward2
        else:
            raise ValueError()
