class ReplayMemory():
    def __init__(self, args):
        self.args = args

        self.logprob = []
        self.opponent_logprob = []
        self.value = []
        self.reward = []

    def add(self, logprob, opponent_logprob, value, reward):
        self.logprob.append(logprob)
        self.opponent_logprob.append(opponent_logprob)
        self.value.append(value)
        self.reward.append(reward)

    def sample(self):
        return self.logprob, self.opponent_logprob, self.value, self.reward
