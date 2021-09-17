# read the data from data source
# save it in the data/raw for further process
import os
import pandas as pd
from get_data import read_params
import argparse
from sklearn.preprocessing import StandardScaler
from pickle import dump
from log_class import getLog


log = getLog("EDA.py")



def eda_process2(df,config_path):
    config = read_params(config_path)
    target = config["base"]["target_col"]

    X = df.drop(target,axis=1)
    scaler=StandardScaler()
    arr=scaler.fit_transform(X)
    log.info("Independent features of the dataset has been normalized")
    y=df[[target]]
    dump(scaler, open('scaler.pkl', 'wb'))
    log.info("Scaler objected has been dumped to scaler.pkl pickle file")
    df2=pd.DataFrame(arr)
    df2[target]=y
    df2.columns=df.columns
    return df2



def eda_process1(df,config_path):
    df['GRE_Score']=df['GRE_Score'].fillna(df['GRE_Score'].mean())
    df['TOEFL_Score']=df['TOEFL_Score'].fillna(df['TOEFL_Score'].mean())
    df['University_Rating']=df['University_Rating'].fillna(df['University_Rating'].mean())
    
    df.drop(columns=['Serial_No'],inplace=True)
    log.info("Missing column values in data set are handled")

    df2=eda_process2(df,config_path)
    
    return df2

def process_df(config_path):
    config = read_params(config_path)
    data_path = config["load_data"]["raw_dataset_csv"]
    raw_data_path=config["process_data"]["raw_dataset_csv"]
    df = pd.read_csv(data_path,sep=',')
    df2=eda_process1(df,config_path)
    print(df2.columns)
    df2.to_csv(raw_data_path, sep=",", index=False)
    



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    process_df(config_path=parsed_args.config)    