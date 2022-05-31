import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
from pytest import PytestUnhandledThreadExceptionWarning


class ATTradingEnv(gym.Env):
    def __init__(self, config):
        self.day = 0
        self.df = config["df"]
        self.stock_dim = len(self.df.tic.unique())
        self.initial_amount = config["initial_amount"]
        self.transaction_cost_pct = config["transaction_cost_pct"]
        self.state_space_shape = self.stock_dim
        self.action_space_shape = self.stock_dim

        self.tech_indicator_list = config["tech_indicator_list"]

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.action_space_shape,))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(len(self.tech_indicator_list), self.state_space_shape))

        self.data = self.df.loc[self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = np.array([self.data[tech].values.tolist()
                              for tech in self.tech_indicator_list])

        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.weights_memory = [[1]+[0]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []


    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.state = [self.data[tech].values.tolist()
                      for tech in self.tech_indicator_list]
        self.state = np.array(self.state)
        self.portfolio_value = self.initial_amount
        self.portfolio_return_memory = [0]

        self.terminal = False
        self.weights_memory = [[1]+[0]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.transaction_cost_memory = []
        return self.state    