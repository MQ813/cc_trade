from model import preprocessing, rnn
from trade import mock

if __name__ == '__main__':
    # klines = preprocessing.get_historical_klines(interval='1d', from_csv=True, save_csv=False)
    # preprocessing.plot_price_trade(klines)

    # model = rnn.get_model(read_model=True, model_path='rnn_model_15m')
    model = rnn.get_model(model_path='rnn_model_15m_gru')
    mock.run(model)