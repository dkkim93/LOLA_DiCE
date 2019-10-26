import torch


class ReplayMemory():
    def __init__(self, args):
        self.args = args

        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, lp, other_lp, v, r):
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(self.args.gamma * torch.ones(*rewards.size()), dim=1) / self.args.gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(self.magic_box(dependencies) * discounted_rewards, dim=1))

        if self.args.use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - self.magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective

    def value_loss(self):
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)
        return torch.mean((rewards - values)**2)

    def magic_box(self, x):
        return torch.exp(x - x.detach())
