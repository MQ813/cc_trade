import time
import matplotlib.pyplot as plt
from binance.client import Client
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import seaborn as sns

klines_columns = (
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
    'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
)
# binance 또는 dump 파일에서 이전 data 가져오기
def get_historical_klines(ticker="BTCUSDT", interval='1d', from_csv=True, save_csv=False, csv_path=''):
    if not csv_path:
        csv_path = f'./data/historical/{ticker}_{interval}_klines.csv'
    if from_csv:
        klines_df = pd.read_csv(csv_path)
    else:
        client = Client(api_key="", api_secret="")
        klines = client.get_historical_klines(
            symbol=ticker,
            interval=interval,
            start_str="2018-01-01",
            limit=1000
        )
        klines_np = np.array(klines)
        klines_df = pd.DataFrame(klines_np.reshape(-1, len(klines_columns)), dtype=float, columns=klines_columns)

    if save_csv:
        klines_df.to_csv(csv_path, sep=',', na_rep='NaN')
    return klines_df


def plot_price_trade(data_df):
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





# 7일간의 5가지 데이터(시가, 종가, 고가, 저가, 거래량)를 받아와서
# 바로 다음 날의 종가를 예측하는 모델로 구성

def buildDataSet(timeSeries, seqLength, outputDim):
    xdata = []
    ydata = []
    for i in range(0, len(timeSeries) - seqLength):
        # tx = timeSeries[i:i + seqLength, :-outputDim]
        tx = timeSeries[i:i + seqLength, :]
        ty = timeSeries[i + seqLength, [-outputDim]]
        xdata.append(tx)
        ydata.append(ty)
    return np.array(xdata), np.array(ydata)








if __name__ == '__main__':
    pass

