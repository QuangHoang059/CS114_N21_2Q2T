from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from init import parameter


# def create_model(actions):
#     model = Sequential()
#     model.add(LSTM(64, return_sequences=True,
#               activation='relu', input_shape=(30, 1662)))
#     model.add(LSTM(128, return_sequences=True, activation='relu'))
#     model.add(LSTM(64, return_sequences=False, activation='relu'))
#     model.add(Dense(64, activation='relu'))
#     model.add(Dense(32, activation='relu'))
#     model.add(Dense(actions.shape[0], activation='softmax'))
#     return model
def create_model(actions):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True,
              activation='relu', input_shape=(parameter["FPS"], 126+33*4)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    return model
