import time, os
import matplotlib.pyplot as plt
from binance.client import Client
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns
from model import preprocessing

def get_model(read_model=False, model_path=''):
    if read_model:
        model = tf.keras.models.load_model(model_path)
    else:
        from tensorflow import keras
        from tensorflow.keras import layers

        # xy = np.loadtxt("./data/historical/BTCUSDT_1d_klines.csv", delimiter=",", skiprows=1)
        # xy = np.loadtxt("./data/historical/BTCUSDT_1h_klines.csv", delimiter=",", skiprows=1)
        xy = np.loadtxt("./data/historical/BTCUSDT_15m_klines.csv", delimiter=",", skiprows=1)
        # xy = xy[::-1]
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
        # trainSize = int(len(xy))
        trainSet = xy[0:trainSize]
        testSet = xy[trainSize - seqLength:]

        trainSet = preprocessing.MinMaxScaler(trainSet)
        testSet = preprocessing.MinMaxScaler(testSet)

        trainX, trainY = preprocessing.buildDataSet(trainSet, seqLength, outputDim)
        testX, testY = preprocessing.buildDataSet(testSet, seqLength, outputDim)

        model = keras.Sequential()
        model.add(layers.GRU(units=10,
                                   activation='tanh',
                                   input_shape=[seqLength, dataDim]))
        # model.add(layers.SimpleRNN(units=10,
        #                            activation='tanh',
        #                            input_shape=[seqLength, dataDim]))
        model.add(layers.Dense(1))
        model.summary()
        # 모델 학습과정 설정
        model.compile(loss='mse', optimizer='adam', metrics=['mae'])
        # 모델 트레이닝
        hist = model.fit(trainX, trainY, epochs=1000, batch_size=16)
        model.save(model_path)
        # 모델 테스트
        res = model.evaluate(testX, testY, batch_size=16)
        print("loss", res[0], "mae", res[1])
        # 7 모델 사용
        xhat = testX
        yhat = model.predict(xhat)
        print(testY)
        print(yhat)
        print("Evaluate : {}".format(np.average((yhat - testY) ** 2)))
        # 원래 값으로 되돌리기
        predict1 = preprocessing.back_MinMax(xy[trainSize - seqLength:, [-1]], yhat)
        actual = preprocessing.back_MinMax(xy[trainSize - seqLength:, [-1]], testY)
        print("예측값", predict1)
        print("실제값", actual)
        print(predict1.shape)
        print(actual.shape)

        plt.figure()
        plt.plot(predict1[300:], label="predict_RNN")
        plt.plot(actual[300:], label="actual")
        plt.legend(prop={'size': 20})
        plt.show()
    return model
