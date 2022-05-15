import numpy as np
import pandas as pd
import os
class Analysis:
    def __init__(self,path):
        self.file_dir=path
        self.all_file_list=os.listdir(self.file_dir)
        self.all_result=[]
        for single_file in self.all_file_list:
            single_data_frame=pd.read_csv(os.path.join(self.file_dir,single_file))
            self.all_result.append(single_data_frame)
    
    def get_report(self):
        if len(self.all_result)==1:
            print("there is only one seed's result")
            tr, sharpe_ratio, vol, mdd, cr, sor=self.analysis(self.all_result[0])
            print("the profit margin is", tr*100, "%")
            print("the sharpe ratio is", sharpe_ratio)
            print("the Volatility is", vol)
            print("the max drawdown is", mdd)
            print("the Calmar Ratio is", cr)
            print("the Sortino Ratio is", sor)
        else:
            trs, sharpe_ratios, vols, mdds, crs, sors=[],[],[],[],[],[]
            for result in self.all_result:
                tr, sharpe_ratio, vol, mdd, cr, sor=self.analysis(result)
                trs.append(tr)
                sharpe_ratios.append(sharpe_ratio)
                vols.append(vol)
                mdds.append(mdd)
                crs.append(cr)
                sors.append(sor)
                tr_mean=np.mean(trs)
                sr_mean=np.mean(sharpe_ratios)
                vols_mean=np.mean(vols)
                mdd_mean=np.mean(mdds)
                cr_mean=np.mean(crs)
                sor_mean=np.mean(sors)

                tr_std=np.std(trs)
                sr_std=np.std(sharpe_ratios)
                vols_std=np.std(vols)
                mdd_std=np.std(mdds)
                cr_std=np.std(crs)
                sor_std=np.std(sors)
                print("the mean profit margin is", tr_mean*100, "%","the standard deviation is",tr_std*100, "%")
                print("the mean sharpe ratio is", sr_mean,"the standard deviation is",sr_std)
                print("the mean Volatility is", vols_mean,"the standard deviation is",vols_std)
                print("the mean max drawdown is", mdd_mean,"the standard deviation is",mdd_std)
                print("the mean Calmar Ratio is", cr_mean,"the standard deviation is",cr_std)
                print("the mean Sortino Ratio is", sor_mean,"the standard deviation is",sor_std)
    
    def analysis(self,df):
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



        

        