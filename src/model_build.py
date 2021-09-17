# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from get_data import read_params
import argparse
import joblib
import json
from sklearn.linear_model import ElasticNet,ElasticNetCV
from log_class import getLog

log = getLog("model_build.py")


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    model_dir = config["model_dir"]
    

    
    cv = config["estimators"]["ElasticNet"]["params"]["cv"]
    norm =config["estimators"]["ElasticNet"]["params"]["normalize"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    y_train = np.array(train[target])
    y_train.reshape(-1,)
    y_train=y_train.ravel()

    y_test =  np.array(test[target])
    y_test.reshape(-1,)
    y_test=y_test.ravel()

    x_train = train.drop(target, axis=1)
    x_test = test.drop(target, axis=1)
    
   
    elasticNetCV=ElasticNetCV(alphas=None,cv=cv,normalize=norm)
    elasticNetCV.fit(x_train,y_train)

    lr_model=ElasticNet(alpha=elasticNetCV.alpha_,l1_ratio=elasticNetCV.l1_ratio_)
    lr_model.fit(x_train,y_train)

    log.info("Linear Regression Model has been build using ElasticNet")

    predicted_score = lr_model.predict(x_test)
    
    (rmse, mae, r2) = eval_metrics(y_test, predicted_score)

    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (elasticNetCV.alpha_, elasticNetCV.l1_ratio_))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  Adj R2: %s" % r2)


    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "a") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "adj_r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "a") as f:
        params = {
            "alpha": elasticNetCV.alpha_,
            "l1_ratio": elasticNetCV.l1_ratio_,
        }
        json.dump(params, f, indent=4)



    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr_model, model_path)
    log.info("Build Model has saved to saved_models directory as model.joblib")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)