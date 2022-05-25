from ast import Raise
from logging import raiseExceptions
from operator import index
import numpy
import os
import pandas as pd
import preprocess as p
import sys
sys.path.append('..')


class Dataconfig:
    def __init__(self, path, data="crypto"):
        self.path = path
        self.data = data
        self.make_dict()
        self.download_data()

    def make_dict(self):
        if (self.data in ["dj30", "sz50", "acl18", "futures", "crypto", "exchange"]) == False:
            raiseExceptions("this dataset is not supported yet")
        if not os.path.exists(self.path+"/" + self.data+"/"+"dataset"):
            os.makedirs(self.path+"/" + self.data+"/"+"dataset")
        self.data_dict = self.path+"/" + self.data+"/"+"dataset"
        if not os.path.exists(self.path+"/" + self.data + "/" + "trained model"):
            os.makedirs(self.path + "/" + self.data + "/" + "trained model")
        self.trained_model_dict = self.path + "/" + self.data + "/" + "trained model"
        if not os.path.exists(self.path + "/" + self.data + "/" + "results"):
            os.makedirs(self.path + "/" + self.data + "/" + "results")
        self.results_dict = self.path + "/" + self.data + "/" + "results"

    def download_data(self):
        datasource = " https://raw.githubusercontent.com/qinmoelei/TradeMater-Data/main/"
        command = "wget -P"+self.data_dict + \
            datasource+self.data+".csv"
        if not os.path.exists(self.data_dict+"/"+self.data+".csv"):
            os.system(command)
        data = pd.read_csv(self.data_dict+"/"+self.data+".csv",index_col=0)
        return data

    def get_data_config(self, portion=[0.8, 0.1, 0.1],transaction_cost_pct=0.001,initial_amount=100000,
    tech_indicator_list=["zopen","zhigh","zlow","zadjcp","zclose","zd_5","zd_10","zd_15","zd_20","zd_25","zd_30"],**kwargs):
        self.dataset = self.download_data()
        self.dataset = p.generate_normalized_feature(self.dataset)
        train, valid, test = p.split(self.dataset, portion)
        base_config={"initial_amount":100000, "transaction_cost_pct":transaction_cost_pct,"tech_indicator_list":tech_indicator_list}
        train_config={"df":train}
        train_config.update(base_config)
        valid_config={"df":valid}
        valid_config.update(base_config)
        test_config={"df":test}
        test_config.update(base_config)
        return train_config,valid_config,test_config
    







if __name__ == "__main__":
    path = "/home/sunshuo/qml/trademaster_new"
    a = Dataconfig(path, "crypto")
    data = a.download_data()
    train,valid,test=a.get_data_config()
    print(train)
    
 