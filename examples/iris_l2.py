import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from Neuralnet.model import Sequential
from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU, Softmax
from Neuralnet.losses import SoftmaxCategoricalCrossEntropy
from Neuralnet.optimizers import Adam
from Neuralnet.metrics import Accuracy
from Neuralnet.regularizers import L2


# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print(X.shape)
print(y.shape)


# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Normalize Features
X_train = (
    X_train - X_train.mean(axis=0)
) / X_train.std(axis=0)

X_test = (
    X_test - X_test.mean(axis=0)
) / X_test.std(axis=0)


# Build Model with L2 Regularization
model = Sequential()

model.add(Dense(4, 16, kernel_regularizer=L2(lambda_=0.001)))
model.add(ReLU())

model.add(Dense(16, 3, kernel_regularizer=L2(lambda_=0.001)))
model.add(Softmax())


# Train
model.compile(
    loss=SoftmaxCategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

model.fit(
    X_train,
    y_train,
    epochs=5000
)


# Evaluate
predictions = model.predict(X_test)

predicted_classes = np.argmax(
    predictions,
    axis=1
)

accuracy = np.mean(
    predicted_classes == y_test
)

print(
    f"Test Accuracy: {accuracy:.4f}"
)