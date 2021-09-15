# read the data from data source
# save it in the data/raw for further process
import os
import pandas as pd
from get_data import read_params, get_data
import argparse
from sklearn.preprocessing import StandardScaler


def eda_process2(df,config_path):
    config = read_params(config_path)
    target = config["base"]["target_col"]

    X = df.drop(target,axis=1)

    scaler=StandardScaler()
    arr=scaler.fit_transform(X)
    column_name = [df.columns]
    y=df[[target]]

    df2=pd.DataFrame(arr)
    df2[target]=y
    df2.columns=column_name
    return df2


def eda_process1(df,config_path):
    df['GRE Score']=df['GRE Score'].fillna(df['GRE Score'].mean())
    df['TOEFL Score']=df['TOEFL Score'].fillna(df['TOEFL Score'].mean())
    df['University Rating']=df['University Rating'].fillna(df['University Rating'].mean())
    
    df.drop(columns=['Serial No.'],inplace=True)
    df2=eda_process2(df,config_path)
    
    return df2

def process_df(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col.replace(" ", "_") for col in df.columns]
    raw_data_path = config["process_data"]["raw_dataset_csv"]
    df2=eda_process1(df,config_path)
    df2.to_csv(raw_data_path, sep=",", index=False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    process_df(config_path=parsed_args.config)    