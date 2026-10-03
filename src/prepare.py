import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

os.makedirs('data/raw', exist_ok=True)
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
np.save('data/raw/X_train.npy', X_train)
np.save('data/raw/y_train.npy', y_train)
np.save('data/raw/X_test.npy', X_test)
np.save('data/raw/y_test.npy', y_test)