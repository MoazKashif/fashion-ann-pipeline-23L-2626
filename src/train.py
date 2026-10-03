import os, yaml
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['train']

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(params['dense_units'], activation='relu'),
    Dropout(params['dropout_rate']),
    Dense(10, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    epochs=params['epochs'], batch_size=params['batch_size'])

os.makedirs('models', exist_ok=True)
model.save('models/model.h5')
pd.DataFrame(history.history).to_csv('models/history.csv', index=False)