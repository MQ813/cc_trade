import time
import matplotlib.pyplot as plt
from binance.client import Client
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns
from model import preprocessing

def run(model, money=10):
    xy = np.loadtxt("./data/historical/BTCUSDT_15m_klines.csv", delimiter=",", skiprows=1)
    xy = xy[:, [2, 3, 4, 6, 8, 9, 10, 11, 5]]
    l_test = int(len(xy) * 0.7)
    xy = xy[l_test:]
    seqLength = 7  # window size
    dataDim = 9  # 시가, 고가, 저가, 거래량 , 종가
    hiddenDim = 10
    outputDim = 1
    lr = 0.01
    iterations = 500

    trainSize = int(len(xy) * 0.8)
    trainSet = xy[0:trainSize]
    testSet = xy[trainSize - seqLength:]

    trainSet = preprocessing.MinMaxScaler(trainSet)
    testSet = preprocessing.MinMaxScaler(testSet)
    trainX, trainY = preprocessing.buildDataSet(trainSet, seqLength, outputDim)
    testX, testY = preprocessing.buildDataSet(testSet, seqLength, outputDim)
    res = model.evaluate(trainX, trainY, batch_size=16)
    print("train : loss", res[0], "mae", res[1])
    res = model.evaluate(testX, testY, batch_size=16)
    print("test : loss", res[0], "mae", res[1])
    xhat_train = trainX
    xhat = testX
    yhat = model.predict(xhat)
    yhat_train = model.predict(xhat_train)

    # 원래 값으로 되돌리기
    predicts = preprocessing.back_MinMax(xy[trainSize - seqLength:, [-1]], yhat_train)
    actuals = preprocessing.back_MinMax(xy[trainSize - seqLength:, [-1]], trainY)

    x=1
    print(predicts.shape)
    bitcoin = 0
    print(f'money start : {money}')
    print('actuals[idx], actuals[idx - 1], predicts[idx], idx, money')
    for idx, predict in enumerate(predicts):
        if idx > 0:
            # if predicts[idx] > predicts[idx-1]*1.03:
            if predicts[idx] > actuals[idx-1]*1.02 and predicts[idx] > predicts[idx-1]*1.02:
                # print(predicts[idx], predicts[idx-1], idx)
                # print(actuals[idx], actuals[idx-1], idx)

                # buy
                bitcoin = money/actuals[idx-1]
                # sell
                money = bitcoin*actuals[idx]
                print(actuals[idx], actuals[idx - 1], predicts[idx], idx, money)
    print(f'money finish : {money}')

    plt.plot(predicts, 'bo-', label="predict_RNN")
    plt.plot(actuals, 'ro-', label="actual")
    plt.show()


