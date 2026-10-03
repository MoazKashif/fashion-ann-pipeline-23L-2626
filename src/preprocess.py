import os, yaml
import numpy as np
from sklearn.model_selection import train_test_split

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)['preprocess']

X_train = np.load('data/raw/X_train.npy') / 255.0
y_train = np.load('data/raw/y_train.npy')
X_test = np.load('data/raw/X_test.npy') / 255.0
X_train = np.load('data/raw/X_train.npy') / 255.00
y_train = np.load('data/raw/y_train.npy')
X_test = np.load('data/raw/X_test.npy') / 255.00
y_test = np.load('data/raw/y_test.npy')

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=params['test_size'], random_state=params['seed']
)

os.makedirs('data/processed', exist_ok=True)
for name, data in zip(['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'], 
                      [X_train, X_val, X_test, y_train, y_val, y_test]):
    np.save(f'data/processed/{name}.npy', data)# mid-edit comment
