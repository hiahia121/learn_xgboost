# -*- coding: utf-8 -*-
import pandas as pd
import logging
import xgboost as xgb
from xgboost import plot_importance
import pickle
import numpy as np
import matplotlib.pyplot as plt


def set_logging():
    logging.basicConfig(level=logging.DEBUG, 
    format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")


def loadDataset(file_path):
    df = pd.read_csv(filepath_or_buffer=file_path, sep=" ")
    return df


def featureSet(data, flag="train"):
    data_num = len(data)
    x_list = []
    y_list = []
    idx = 0
    for row in range(0, 10000):
        if idx % 1000 == 0:
            logging.debug("deal idx is %s" % idx)
        idx += 1
        tmp_list = []
        tmp_list.append(data.iloc[row]["regDate"])
        tmp_list.append(data.iloc[row]["brand"])
        tmp_list.append(data.iloc[row]["bodyType"])
        tmp_list.append(data.iloc[row]["fuelType"])
        tmp_list.append(data.iloc[row]["gearbox"])
        tmp_list.append(data.iloc[row]["power"])
        tmp_list.append(data.iloc[row]["kilometer"])
        tmp_list.append(data.iloc[row]["notRepairedDamage"])
        tmp_list.append(data.iloc[row]["creatDate"])
        if '-' in tmp_list:
            continue
        x_list.append(tmp_list)
        if flag == "train":
            y = data.iloc[row]['price']
            y_list.append([y])

    if flag == "train":
        return np.array(x_list), np.array(y_list)

    elif flag == "test":
        return np.array(x_list)


def train_(x_train, y_train, model_file):
    model = xgb.XGBRegressor(max_depth=5, learning_rete=0.1, n_estimators=160, silent=False, objective='reg:gamma')
    model.fit(x_train, y_train)
    plot_importance(model)
    plt.show()

    # 保存模型
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    
    
def use_model_predict(model_file, x_test):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    ans = model.predict(x_test)
    print ans
    print len(ans)
    if 1 == 0:
        data_arr = []
        ans_len = len(ans)
        for row in range(0, ans_len):
            pass
        

if __name__ == "__main__":
    set_logging()
    train_file_path = "data/used_car_train_20200313.csv"
    test_file_path = "data/used_car_testA_20200313.csv"
    model_file = "data/model.pickle"
    train_data = loadDataset(train_file_path)
    test_data = loadDataset(test_file_path)
    
    #x_train, y_train = featureSet(train_data)
    #train_(x_train, y_train, model_file)

    x_test = featureSet(test_data, "test")
    use_model_predict(model_file, x_test)