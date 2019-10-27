import torch


class ReplayMemory():
    def __init__(self, args):
        self.args = args

        self.self_logprob = []
        self.other_logprob = []
        self.value = []
        self.reward = []

    def add(self, lp, other_lp, v, r):
        self.self_logprob.append(lp)
        self.other_logprob.append(other_lp)
        self.value.append(v)
        self.reward.append(r)

    def sample(self):
        return self.self_logprob, self.other_logprob, self.value, self.reward

    def dice_objective(self):
        self_logprob = torch.stack(self.self_logprob, dim=1)
        other_logprob = torch.stack(self.other_logprob, dim=1)
        value = torch.stack(self.value, dim=1)
        reward = torch.stack(self.reward, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(self.args.discount * torch.ones(*reward.size()), dim=1) / self.args.discount
        discounted_rewards = reward * cum_discount
        discounted_values = value * cum_discount

        # stochastics nodes involved in reward dependencies:
        dependencies = torch.cumsum(self_logprob + other_logprob, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprob + other_logprob

        # dice objective:
        dice_objective = torch.mean(torch.sum(self.magic_box(dependencies) * discounted_rewards, dim=1))

        if self.args.use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - self.magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective

    def magic_box(self, x):
        return torch.exp(x - x.detach())
