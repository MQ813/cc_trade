import time
import matplotlib.pyplot as plt
from binance.client import Client
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns


# binance 또는 dump 파일에서 이전 data 가져오기
def get_historical_klines(ticker="BTCUSDT", interval='1d', from_dump=True, save_dump=False, dump_path=''):
    if not dump_path:
        dump_path = f'./data/historical/{ticker}_{interval}_klines.pickle'
    if from_dump:
        with open(dump_path, 'rb') as f_dmp:
            klines = pickle.load(f_dmp)
    else:
        api_key = ""
        api_secret = ""
        client = Client(api_key=api_key, api_secret=api_secret)
        klines = client.get_historical_klines(
            symbol=ticker,
            interval=interval,
            start_str="2018-01-01",
            limit=1000
        )
        if save_dump:
            with open(dump_path, "wb") as f_dmp:
                pickle.dump(klines, f_dmp)
    return klines


def plot_data(data, columns):
    data_np = np.array(data)
    data_df = pd.DataFrame(data_np.reshape(-1, len(columns)), dtype=float, columns=columns)
    data_df['Open Time'] = pd.to_datetime(data_df['Open Time'], unit='ms')
    times = data_df['Open Time']
    trades = data_df['Number of trades']
    close = data_df['Close']

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(times, close, 'b')
    ax2.plot(times, trades, 'r')
    plt.show()


# 정규화 함수
def MinMaxScaler(data):
    denom = np.max(data, 0)-np.min(data, 0)
    nume = data-np.min(data, 0)
    return nume/denom


# 정규화 되돌리기 함수
def back_MinMax(data,value):
    diff = np.max(data,0)-np.min(data,0)
    back = value * diff + np.min(data,0)
    return back



if __name__ == '__main__':
    pass

