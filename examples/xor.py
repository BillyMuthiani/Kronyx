import numpy as np

from Neuralnet.model import Sequential
from Neuralnet.layers import Dense
from Neuralnet.activations import ReLU, Sigmoid
from Neuralnet.losses import BinaryCrossEntropy
from Neuralnet.optimizers import SGD
from Neuralnet.metrics import Accuracy

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0]).reshape(-1, 1)

print(f"Input shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Build model
model = Sequential()
model.add(Dense(2, 4))
model.add(ReLU())
model.add(Dense(4, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=SGD(learning_rate=0.1),
    metric=Accuracy()
)

# Train
model.fit(X, y, epochs=10000)

# Evaluate
predictions = model.predict(X)
print(f"\nPredictions:\n{predictions}")
print(f"Targets:\n{y}")