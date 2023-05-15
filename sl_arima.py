
import streamlit as st
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as pyplot


st.title('Daily Mean PM10 Concentration in LA, California')

DATE_COLUMN = "Date"
DATA_URL = "data/LA_pm10_2022.csv"


@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    data = data[[DATE_COLUMN, 'Daily Mean PM10 Concentration']]
    return data

data_load_state = st.text('Loading data...')
data = load_data(1000)
#display data and line char
st.dataframe(data = data)
st.line_chart(data, x = 'Date', y='Daily Mean PM10 Concentration')
data_load_state.text("Done! (using st.cache_data)")

#CO concentration per day

def CO_per_day(data):
    ts = data.groupby(data["Date"])["Daily Mean PM10 Concentration"].sum()
    ts.dropna()
    return ts

ts = CO_per_day(data=data)
st.dataframe(data = ts)
st.bar_chart(data=ts, y="Daily Mean PM10 Concentration")

def seasonal_decomp(data):
    dec_mul = seasonal_decompose(data, model = 'multiplicative', extrapolate_trend='freq')
    plot = dec_mul.plot()
    plt.rcParams.update({'figure.figsize': (10,10)})
    st.pyplot(plot)
seasonal_decomp(ts)

def build_arima(ts):
    # split into train and test sets
    X = ts.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    data = pd.DataFrame(
    {'predictions': predictions,
     'test': test
    })
    df = pd.DataFrame(data, columns=['predictions', 'test'])
    st.line_chart(df)
    return st.text('Test RMSE: %.3f' % rmse)

build_arima(ts)
