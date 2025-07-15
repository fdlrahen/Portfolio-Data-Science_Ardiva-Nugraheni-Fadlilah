import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input, GRU
from keras.utils import plot_model

# Load data
url = 'https://raw.githubusercontent.com/MainakRepositor/Datasets/master/Cryptocurrency/bitcoin.csv'
hist = pd.read_csv(url)

# Set 'Date' as index and convert to datetime
hist = hist.set_index('Date')
hist.index = pd.to_datetime(hist.index)

# Convert relevant columns to numeric and handle NaN
relevant_cols = ['Open', 'High', 'Low', 'Close']
hist[relevant_cols] = hist[relevant_cols].apply(pd.to_numeric, errors='coerce')
hist = hist.ffill()

target_col = 'Close'

def custom_train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_lstm(train_data, window_len, lstm_neurons, dropout, loss, optimizer, epochs, batch_size):
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data[relevant_cols])
    X_train, y_train = create_sequences(train_data_scaled, window_len)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(relevant_cols)))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], len(relevant_cols))))
    model.add(LSTM(units=lstm_neurons, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=lstm_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # Print model summary and save plot
    st.text("LSTM Model Summary:")
    model.summary(print_fn=lambda x: st.text(x))
    plot_model(model, to_file='model_plot_lstm.png', show_shapes=True, show_layer_names=True)
    st.image('model_plot_lstm.png')
    return model, scaler

def train_gru(train_data, window_len, gru_neurons, dropout, loss, optimizer, epochs, batch_size):
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data[relevant_cols])
    X_train, y_train = create_sequences(train_data_scaled, window_len)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], len(relevant_cols)))

    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], len(relevant_cols))))
    model.add(GRU(units=gru_neurons, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(GRU(units=gru_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))

    model.compile(optimizer=optimizer, loss=loss)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    # Print model summary and save plot
    st.text("GRU Model Summary:")
    model.summary(print_fn=lambda x: st.text(x))
    plot_model(model, to_file='model_plot_gru.png', show_shapes=True, show_layer_names=True)
    st.image('model_plot_gru.png')
    return model, scaler

def evaluate_model(model, test_data, window_len, scaler):
    test_data_scaled = scaler.transform(test_data[relevant_cols])
    X_test, y_test = create_sequences(test_data_scaled, window_len)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], len(relevant_cols)))
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    temp_df = pd.DataFrame(test_data_scaled, columns=relevant_cols)
    temp_df[target_col] = np.nan
    temp_df.loc[temp_df.index[-len(predictions):], target_col] = predictions.flatten()

    predictions_actual = scaler.inverse_transform(temp_df)[:, relevant_cols.index(target_col)]
    y_test_actual = scaler.inverse_transform(temp_df)[:, relevant_cols.index(target_col)]
    
    # Remove NaN and inf values
    mask = ~np.isnan(y_test_actual) & ~np.isinf(y_test_actual) & ~np.isnan(predictions_actual) & ~np.isinf(predictions_actual)
    y_test_actual = y_test_actual[mask]
    predictions_actual = predictions_actual[mask]

    mae = mean_absolute_error(y_test_actual[-len(predictions_actual):], predictions_actual)
    return predictions_actual, y_test_actual[-len(predictions_actual):], mae

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('Price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)
    return fig

def dual_line_plot(line1, line2, label1=None, label2=None, lw=2):
    import matplotlib.dates as mdates
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_ylabel('Daily Returns', fontsize=14)
    ax.legend(loc='best', fontsize=18)
    return fig

def main():
    st.title('Bitcoin Price Prediction')
    st.sidebar.title('Parameters')
    test_size = st.sidebar.slider('Test Size', 0.05, 0.3, 0.1, step=0.01)
    window_len = st.sidebar.slider('Window Length', 5, 30, 7)
    lstm_neurons = st.sidebar.slider('LSTM Neurons', 10, 100, 20)
    gru_neurons = st.sidebar.slider('GRU Neurons', 10, 100, 20)
    dropout = st.sidebar.slider('Dropout Rate', 0.1, 0.5, 0.25)
    epochs = st.sidebar.slider('Epochs', 10, 200, 100)
    batch_size = st.sidebar.slider('Batch Size', 1, 32, 4)
    loss = st.sidebar.selectbox('Loss Function', ['mae', 'mse'])
    optimizer = st.sidebar.selectbox('Optimizer', ['adam', 'rmsprop'])
    
    # Define n_points here to ensure it's available
    n_points = 30

    train_data, test_data = custom_train_test_split(hist, test_size=test_size)
    st.subheader('Column Names')
    st.write(hist.columns)

    st.subheader('Bitcoin Historical Data')
    st.write(hist)

    st.subheader('Descriptive Statistics')
    st.write(hist.describe())

    st.subheader('Relevant Columns')
    st.write(relevant_cols)

    st.subheader('Bitcoin Price Over Time')
    fig = line_plot(train_data[target_col], test_data[target_col], 'Training', 'Test', title='BTC')
    st.pyplot(fig)

    st.subheader('Model Training and Evaluation')

    # Define targets here to ensure they are available for returns comparison
    targets = test_data[target_col][window_len:]
    actual_returns = targets.pct_change()[1:]

    if st.button('Train LSTM Model'):
        model_lstm, scaler_lstm = train_lstm(train_data, window_len, lstm_neurons, dropout, loss, optimizer, epochs, batch_size)
        st.success('LSTM model trained successfully!')

        st.subheader('Evaluation Results for LSTM')
        predictions_lstm, y_test_actual_lstm, mae_lstm = evaluate_model(model_lstm, test_data, window_len, scaler_lstm)
        st.write('LSTM Mean Absolute Error:', mae_lstm)

        preds_lstm = pd.Series(index=targets.index, data=predictions_lstm.squeeze())
        fig = line_plot(targets, preds_lstm, 'actual', 'LSTM prediction', title='Actual vs. Predicted BTC Prices (LSTM)', lw=3)
        st.pyplot(fig)

    if st.button('Train GRU Model'):
        model_gru, scaler_gru = train_gru(train_data, window_len, gru_neurons, dropout, loss, optimizer, epochs, batch_size)
        st.success('GRU model trained successfully!')

        st.subheader('Evaluation Results for GRU')
        predictions_gru, y_test_actual_gru, mae_gru = evaluate_model(model_gru, test_data, window_len, scaler_gru)
        st.write('GRU Mean Absolute Error:', mae_gru)

        preds_gru = pd.Series(index=targets.index, data=predictions_gru.squeeze())
        fig = line_plot(targets, preds_gru, 'actual', 'GRU prediction', title='Actual vs. Predicted BTC Prices (GRU)', lw=3)
        st.pyplot(fig)

    st.subheader("Returns Comparison for LSTM")
    if 'preds_lstm' in locals():
        predicted_returns_lstm = preds_lstm.pct_change()[1:]
        dual_fig_lstm = dual_line_plot(actual_returns[-n_points:], predicted_returns_lstm[-n_points:], 'actual returns', 'LSTM predicted returns', lw=3)
        st.pyplot(dual_fig_lstm)

    st.subheader("Returns Comparison for GRU")
    if 'preds_gru' in locals():
        predicted_returns_gru = preds_gru.pct_change()[1:]
        dual_fig_gru = dual_line_plot(actual_returns[-n_points:], predicted_returns_gru[-n_points:], 'actual returns', 'GRU predicted returns', lw=3)
        st.pyplot(dual_fig_gru)

if __name__ == '__main__':
    main()
