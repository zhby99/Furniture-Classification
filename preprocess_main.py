from _preprocess import *

X_train, y_train = preprocess_train(True)
y_train = y_train - 1
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

X_val, y_val = preprocess_valid()
y_val = y_val - 1
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
