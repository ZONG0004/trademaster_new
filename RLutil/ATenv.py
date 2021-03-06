import numpy as np
from gym.utils import seeding
import gym
from gym import spaces
import pandas as pd
from pytest import PytestUnhandledThreadExceptionWarning
import more_itertools
HMAX_NORMALIZE = 10

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
                                            shape=(((len(self.tech_indicator_list)+1)*self.state_space_shape)+1,))

        self.data = self.df.loc[self.day, :]
        # initially, the self.state's shape is 1+stock_dim*（len(tech_indicator_list)+1）
        self.state = np.array([self.initial_amount] + [0]*self.stock_dim + \
                               list(more_itertools.collapse([self.data[tech].values.tolist() for tech in self.tech_indicator_list])))

        self.terminal = False
        self.stock_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.stock_return_memory = [0]
        self.stock_memory = [[0]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.rewards_memory = []
        self.transaction_cost_memory = []
        self.reward = 0
        self.cost = 0
        self.trades = 0


    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.state = np.array([self.initial_amount] + [0]*self.stock_dim + \
                               list(more_itertools.collapse([self.data[tech].values.tolist() for tech in self.tech_indicator_list])))
        #self.state = np.array(self.state)
        self.stock_value = self.initial_amount
        self.stock_return_memory = [0]

        self.terminal = False
        self.stock_memory = [[0]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.reward_memory = []
        self.transaction_cost_memory = []
        self.cost = 0
        self.trades = 0
        return self.state

    def step(self, actions):
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique())-1
        actions = np.array(actions)

        if self.terminal:
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            print("=================================")
            print("the profit margin is", tr*100, "%")
            print("the sharpe ratio is", sharpe_ratio)
            print("the Volatility is", vol)
            print("the max drawdown is", mdd)
            print("the Calmar Ratio is", cr)
            print("the Sortino Ratio is", sor)
            print("=================================")
            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))   

            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])


            # step into the next time stamp
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # get the state
            self.state =  [self.state[0]] + \
                    list(self.state[1:(self.stock_dim+1)]) + \
                      list(more_itertools.collapse([self.data[tech].values.tolist() for tech in self.tech_indicator_list]))

            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            stock_return = self.reward/begin_total_asset
            self.stock_return_memory.append(stock_return)
            self.date_memory.append(self.data.date.unique()[0])

        return self.state, self.reward, self.terminal, {}

    
    def _sell_stock(self, index, action):
        if self.state[index+self.stock_dim] > 0:
            self.state[0] += self.state[index+self.stock_dim+1]*min(abs(action),self.state[index+1]) * (1- self.transaction_cost_pct)                         
            self.state[index+1] -= min(abs(action), self.state[index+1])
            self.cost += self.state[index+self.stock_dim+1]*min(abs(action),self.state[index+1]) * self.transaction_cost_pct                 
            self.trades+=1
        else:
            pass 

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+self.stock_dim+1]
        # print('available_amount:{}'.format(available_amount))
            
        #update balance
        self.state[0] -= self.state[index+self.stock_dim+1]*min(available_amount, action)*(1+ self.transaction_cost_pct)                      

        self.state[index+1] += min(available_amount, action)
            
        self.cost+=self.state[index+self.stock_dim+1]*min(available_amount, action) * self.transaction_cost_pct
        self.trades+=1


    def save_stock_return_memory(self):
        # a record of return for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.stock_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = df_date.date

        return df_return

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = df_date.date

        return df_value

    def evaualte(self,df):
        # a function to analysis the return & risk using history record
        daily_return = df["daily_return"]
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1]/df["total assets"].values[0]-1
        sharpe_ratio = np.mean(daily_return) / \
            np.std(daily_return)*(len(df)**0.5)
        vol = np.std(daily_return)
        mdd = max((max(df["total assets"]) -
                  df["total assets"])/max(df["total assets"]))
        cr = np.sum(daily_return)/mdd
        sor = np.sum(daily_return)/np.std(neg_ret_lst) / \
            np.sqrt(len(daily_return))
        return tr, sharpe_ratio, vol, mdd, cr, sor

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_stock_return_memory()
        daily_return = df_return.daily_return.values
        df_value = self.save_asset_memory()
        assets = df_value["total assets"].values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        return self.evaualte(df)




class ATTestingEnv(gym.Env):
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
                                            shape=(((len(self.tech_indicator_list)+1)*self.state_space_shape)+1,))

        self.data = self.df.loc[self.day, :]
        # initially, the self.state's shape is stock_dim*len(tech_indicator_list)
        self.state = np.array([self.initial_amount] + [0]*self.stock_dim + \
                               list(more_itertools.collapse([self.data[tech].values.tolist() for tech in self.tech_indicator_list])))

        self.terminal = False
        self.stock_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.stock_return_memory = [0]
        self.stock_memory = [[0]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.rewards_memory = []
        self.transaction_cost_memory = []
        self.reward = 0
        self.cost = 0
        self.trades = 0


    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.state = np.array([self.initial_amount] + [0]*self.stock_dim + \
                               list(more_itertools.collapse([self.data[tech].values.tolist() for tech in self.tech_indicator_list])))
        #self.state = np.array(self.state)
        self.stock_value = self.initial_amount
        self.stock_return_memory = [0]

        self.terminal = False
        self.stock_memory = [[0]*self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
        self.reward_memory = []
        self.transaction_cost_memory = []
        self.cost = 0
        self.trades = 0
        return self.state

    def step(self, actions):
        # make judgement about whether our data is running out
        self.terminal = self.day >= len(self.df.index.unique())-1
        actions = np.array(actions)

        if self.terminal:
            tr, sharpe_ratio, vol, mdd, cr, sor = self.analysis_result()
            print("=================================")
            print("the profit margin is", tr*100, "%")
            print("the sharpe ratio is", sharpe_ratio)
            print("the Volatility is", vol)
            print("the max drawdown is", mdd)
            print("the Calmar Ratio is", cr)
            print("the Sortino Ratio is", sor)
            print("=================================")
            return self.state, self.reward, self.terminal, sharpe_ratio

        else:
            actions = actions * HMAX_NORMALIZE
            #actions = (actions.astype(int))
            begin_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))   

            #print("begin_total_asset:{}".format(begin_total_asset))
            
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                # print('take sell action'.format(actions[index]))
                self._sell_stock(index, actions[index])

            for index in buy_index:
                # print('take buy action: {}'.format(actions[index]))
                self._buy_stock(index, actions[index])


            # step into the next time stamp
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # get the state
            self.state =  [self.state[0]] + \
                    list(self.state[1:(self.stock_dim+1)]) + \
                      list(more_itertools.collapse([self.data[tech].values.tolist() for tech in self.tech_indicator_list]))

            end_total_asset = self.state[0]+ \
            sum(np.array(self.state[1:(self.stock_dim+1)])*np.array(self.state[(self.stock_dim+1):(self.stock_dim*2+1)]))
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset            
            self.rewards_memory.append(self.reward)
            stock_return = self.reward/begin_total_asset
            self.stock_return_memory.append(stock_return)
            self.date_memory.append(self.data.date.unique()[0])

        return self.state, self.reward, self.terminal, {}

    
    def _sell_stock(self, index, action):
        if self.state[index+self.stock_dim] > 0:
            self.state[0] += self.state[index+self.stock_dim+1]*min(abs(action),self.state[index+1]) * (1- self.transaction_cost_pct)                         
            self.state[index+1] -= min(abs(action), self.state[index+1])
            self.cost += self.state[index+self.stock_dim+1]*min(abs(action),self.state[index+1]) * self.transaction_cost_pct                 
            self.trades+=1
        else:
            pass 

    
    def _buy_stock(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+self.stock_dim+1]
        # print('available_amount:{}'.format(available_amount))
            
        #update balance
        self.state[0] -= self.state[index+self.stock_dim+1]*min(available_amount, action)*(1+ self.transaction_cost_pct)                      

        self.state[index+1] += min(available_amount, action)
            
        self.cost+=self.state[index+self.stock_dim+1]*min(available_amount, action) * self.transaction_cost_pct
        self.trades+=1


    def save_stock_return_memory(self):
        # a record of return for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        return_list = self.stock_return_memory
        df_return = pd.DataFrame(return_list)
        df_return.columns = ["daily_return"]
        df_return.index = df_date.date

        return df_return

    def save_asset_memory(self):
        # a record of asset values for each time stamp
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']

        assets_list = self.asset_memory
        df_value = pd.DataFrame(assets_list)
        df_value.columns = ["total assets"]
        df_value.index = df_date.date

        return df_value

    def evaualte(self,df):
        # a function to analysis the return & risk using history record
        daily_return = df["daily_return"]
        neg_ret_lst = df[df["daily_return"] < 0]["daily_return"]
        tr = df["total assets"].values[-1]/df["total assets"].values[0]-1
        sharpe_ratio = np.mean(daily_return) / \
            np.std(daily_return)*(len(df)**0.5)
        vol = np.std(daily_return)
        mdd = max((max(df["total assets"]) -
                  df["total assets"])/max(df["total assets"]))
        cr = np.sum(daily_return)/mdd
        sor = np.sum(daily_return)/np.std(neg_ret_lst) / \
            np.sqrt(len(daily_return))
        return tr, sharpe_ratio, vol, mdd, cr, sor

    def analysis_result(self):
        # A simpler API for the environment to analysis itself when coming to terminal
        df_return = self.save_stock_return_memory()
        daily_return = df_return.daily_return.values
        df_value = self.save_asset_memory()
        assets = df_value["total assets"].values
        df = pd.DataFrame()
        df["daily_return"] = daily_return
        df["total assets"] = assets
        return self.evaualte(df)
           