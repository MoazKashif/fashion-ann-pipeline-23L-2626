import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

model = load_model('models/model.h5')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
y_pred = np.argmax(model.predict(X_test), axis=1)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('models/confusion_matrix.png')

with open('metrics.json', 'w') as f:
    json.dump({'test_loss': loss, 'test_accuracy': accuracy}, f)