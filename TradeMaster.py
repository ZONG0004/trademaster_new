from hashlib import algorithms_guaranteed
import sys
sys.path.append("./RLutil/")
sys.path.append("./data/")
from data.data_config import *
from RLutil.agent import *
from RLutil.env import TestingEnv, TradingEnv
from ast import Raise
from logging import raiseExceptions
import numpy
import os
import pandas as pd
from analysis.risk_return import *
class TradeMaster(Dataconfig,agent,Analysis):
    def __init__(self, path, data,algorithms,seed,train_env,test_env):
        self.algorithms=algorithms
        Dataconfig.__init__(self,path, data)
        train,valid,test=self.get_data_config()
        trained_model_path=self.trained_model_dict
        result_path=self.results_dict
        print(result_path)
        agent.__init__(self,algorithms,train,seed,trained_model_path,result_path,valid,test,train_env,test_env)
        Analysis.__init__(self,path=self.result_path)

        

    def get_environment(self):
        train, valid, test = self.get_data_config()
        valid_env = TradingEnv(valid)
        test_env = TradingEnv(test)
    
    def make_dict(self):
        if (self.data in ["dj30", "sz50", "acl18", "futures", "crypto", "exchange"]) == False:
            raiseExceptions("this dataset is not supported yet")
        if not os.path.exists(self.path+"/" + self.data+"/"+"dataset"):
            os.makedirs(self.path+"/" + self.data+"/"+"dataset")
        self.data_dict = self.path+"/" + self.data+"/"+"dataset"
        if not os.path.exists(self.path+"/" + self.data + "/" +self.algorithms+ "/"+ "trained model"):
            os.makedirs(self.path+"/" + self.data + "/" +self.algorithms+ "/"+ "trained model")
        self.trained_model_dict = self.path+"/" + self.data + "/" +self.algorithms+ "/"+ "trained model"
        if not os.path.exists(self.path + "/" + self.data+ "/" +self.algorithms + "/" + "results"):
            os.makedirs(self.path + "/" + self.data+ "/" +self.algorithms + "/" + "results")
        self.results_dict = self.path + "/" + self.data+ "/" +self.algorithms + "/" + "results"
    
    

if __name__=="__main__":
    path = "/mnt/c/Users/DELL/Desktop/code/trademaster_new/experiment"
    for seed in [12345,23451,34512,45123,51234]:
        a = TradeMaster(path=path,data="acl18",algorithms="td3",seed=seed,train_env=TradingEnv,test_env=TestingEnv)
        a.train_with_valid()
        a.test()
    a.get_report()