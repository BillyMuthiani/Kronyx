import numpy as np

from Neuralnet.layers import Dense
from Neuralnet.activations import Sigmoid, Tanh
from Neuralnet.losses import BinaryCrossEntropy
from Neuralnet.metrics import Accuracy
from Neuralnet.optimizers import Adam

from Neuralnet.model import Sequential


X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

y = np.array([
    [0],
    [1],
    [1],
    [0]
])


model = Sequential()

model.add(Dense(2, 8))
model.add(Tanh())

model.add(Dense(8, 1))
model.add(Sigmoid())

model.compile(
    loss=BinaryCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    metric=Accuracy()
)

model.fit(
    X,
    y,
    epochs=5000
)

predictions = model.predict(X)

print(predictions)