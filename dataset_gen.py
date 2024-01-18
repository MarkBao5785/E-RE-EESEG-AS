import numpy as np
import pandas as pd
import random

import itertools
import random
import torch

class IRIS:
    def __init__(self):
        self.arm = 3
        self.n_arm = 3
        self.n_arms = 3
        self.dim = 12
        self.data = pd.read_csv('Iris.csv')

    def step(self):
        r = random.randint( 0, 149)
        if  0 <= r <= 49:
            target = 0
        elif 50 <= r <= 99:
            target = 1
        else:
            target = 2
        rd = self.data.loc[r]
        x = np.zeros(4)
        for i in range(1,5):
            x[i-1] = rd[i]
        X_n = []
        for i in range(3):
            front = np.zeros((4 * i))
            back = np.zeros((4 * (2 - i)))
            new_d = np.concatenate((front, x, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)
        reward = np.zeros(self.arm)
        # print(target)
        reward[target] = 1
        return X_n, reward

class Experimental_ContextualBandit():
    def __init__(self,
                 T,
                 n_arms,
                 n_features,
                 h,
                 context,
                 noise_std=1.0,
                 seed=None,
                 ):
        # if not None, freeze seed for reproducibility
        self._seed(seed)

        # number of rounds
        self.T = T*3
        # number of arms
        self.n_arms = n_arms
        # number of features for each arm
        self.n_features = n_features
        # average reward function
        # h : R^d -> R
        self.h = h
        self.context = context
        # standard deviation of Gaussian reward noise
        self.noise_std = noise_std

        # generate random features
        self.reset()

        self.round = 0

        self.dim = n_features
        

    @property
    def arms(self):
        """Return [0, ...,n_arms-1]
        """
        return range(self.n_arms)

    def reset(self):
        """Generate new features and new rewards.
        """
        self.reset_features()
        self.reset_rewards()
        self.round = 0

    def reset_features(self):
        """Generate normalized random N(0,1) features.
        """
        x = np.random.randn(self.T, self.n_arms, self.n_features)
        x /= np.repeat(np.linalg.norm(x, axis=-1, ord=2), self.n_features).reshape(self.T, self.n_arms, self.n_features)
        self.features = x

    def reset_rewards(self):
        """Generate rewards for each arm and each round,
        following the reward function h + Gaussian noise.
        """
        self.rewards = np.array(
            [
                self.h(self.features[t, k]) + self.noise_std*np.random.randn()
                for t, k in itertools.product(range(self.T), range(self.n_arms))
            ]
        ).reshape(self.T, self.n_arms)

        # to be used only to compute regret, NOT by the algorithm itself
        self.best_rewards_oracle = np.max(self.rewards, axis=1)
        self.best_actions_oracle = np.argmax(self.rewards, axis=1)

    def step(self):
        current_reward = self.rewards[self.round]
        current_context = self.features[self.round]
        self.round += 1
        return current_context, current_reward


    def _seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

class Experimental_Setting:
    def __init__(self, T, n_arms, n_features, reward_func, context, noise_std, seed):
        self.T = T
        self.n_arms = n_arms
        self.n_features = n_features
        self.reward_func = reward_func
        self.context = context
        self.noise_std = noise_std
        self.seed = seed
        self.dim = n_features

if __name__ == "__main__":
    T = int(2000)
    n_arms = 4
    n_features = 8
    noise_std = 0.1

    SEED = 42

    a = np.random.randn(n_features)
    a /= np.linalg.norm(a, ord=2)
    reward_func = lambda x: 10*np.dot(a, x)

    bandit = Experimental_ContextualBandit(T, n_arms, n_features, reward_func, a, noise_std=noise_std, seed=SEED)
    bandit.reset_rewards()
